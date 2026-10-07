//! `dzk nnbench` (per-decision cost of the network: observation build, encode, forward, measured apart, on
//! the real decisions of played games, single thread) and `dzk parity` (Rust forward vs PyTorch outputs on
//! dumped decisions).

use crate::bots::BotSpec;
use crate::data;
use crate::encode::{self, encode_game, Encoded};
use crate::game::Game;
use crate::nn::{Model, Scratch};
use crate::rng::bot_seed;
use crate::runner::{load_decks, read_pairs};
use mtg_kernel::rl::observe_policy_v5_without_projection_hash_v1;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct NnBenchArgs {
    pub decks_dir: PathBuf,
    pub pairs_file: PathBuf,
    pub net: Option<PathBuf>,
    pub bot: String,
    pub games: usize,
    pub seed: u64,
}

fn stats_us(mut xs: Vec<f64>) -> Value {
    if xs.is_empty() {
        return Value::Null;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| xs[((xs.len() - 1) as f64 * p).round() as usize];
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let r = |x: f64| (x * 100.0).round() / 100.0;
    json!({"n": xs.len(), "mean": r(mean), "p50": r(q(0.5)), "p90": r(q(0.9)), "p99": r(q(0.99)), "max": r(q(1.0))})
}

pub fn run_nnbench(a: &NnBenchArgs) -> Result<Value, String> {
    let spec = BotSpec::parse(&a.bot)?;
    let pairs = read_pairs(&a.pairs_file)?;
    let decks = load_decks(&a.decks_dir, pairs.iter().flat_map(|(x, y)| [x.clone(), y.clone()]))?;
    let model = a.net.as_ref().map(|p| Model::load(p)).transpose()?;
    let mut enc = Encoded::default();
    let mut sc = Scratch::default();
    let (mut t_obs, mut t_enc, mut t_fwd, mut t_all) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut t_fwd_hot, mut n_unique) = (Vec::new(), Vec::new());
    let (mut n_obj, mut n_act) = (Vec::new(), Vec::new());
    let mut kinds: BTreeMap<String, u64> = BTreeMap::new();
    let mut families: BTreeMap<String, u64> = BTreeMap::new();
    let mut zones = [0u64; encode::NZ];
    let (mut unlinked, mut dropped, mut obs_errors, mut nondeterministic) = (0u64, 0u64, 0u64, 0u64);
    let mut kind_names: BTreeMap<u8, String> = BTreeMap::new();
    for g in 0..a.games {
        let (d1, d2) = &pairs[g % pairs.len()];
        let env_seed = a.seed.wrapping_mul(1000).wrapping_add(g as u64);
        let mut game = Game::new(&decks[d1], &decks[d2], env_seed, env_seed, false)?;
        let mut bots = [spec.build(bot_seed(env_seed, 0)), spec.build(bot_seed(env_seed, 1))];
        while !game.is_terminal() {
            let actor = game.actor().unwrap();
            // the observation, rebuilt as the session builds it at every step (its cost is paid by every bot)
            let at = game.decision_coords().unwrap();
            let t0 = Instant::now();
            let obs = observe_policy_v5_without_projection_hash_v1(
                game.state(),
                game.session().policy_surface_v1(),
                actor,
                at.step,
                at.physical_decision_id,
                at.substep_index,
                at.substep_count,
            );
            let dt_obs = t0.elapsed().as_secs_f64() * 1e6;
            if obs.is_err() {
                obs_errors += 1;
            }
            drop(obs);
            let t1 = Instant::now();
            encode_game(&game, &mut enc).map_err(|e| format!("{e:?}"))?;
            let dt_enc = t1.elapsed().as_secs_f64() * 1e6;
            let mut dt_fwd = 0.0;
            if let Some(m) = &model {
                let t2 = Instant::now();
                m.forward(&enc, &mut sc);
                dt_fwd = t2.elapsed().as_secs_f64() * 1e6;
                let (l, v) = (sc.logits.clone(), sc.value);
                // hot-cache repeats (the same decision again): the best of three
                let mut hot = f64::INFINITY;
                for _ in 0..3 {
                    let t3 = Instant::now();
                    m.forward(&enc, &mut sc);
                    hot = hot.min(t3.elapsed().as_secs_f64() * 1e6);
                    if l != sc.logits || v != sc.value {
                        nondeterministic += 1;
                    }
                }
                t_fwd_hot.push(hot);
                n_unique.push(sc.unique_objects as f64);
                t_fwd.push(dt_fwd);
            }
            t_obs.push(dt_obs);
            t_enc.push(dt_enc);
            t_all.push(dt_obs + dt_enc + dt_fwd);
            n_obj.push(enc.n_obj() as f64);
            n_act.push(enc.n_act() as f64);
            unlinked += u64::from(enc.unlinked_refs);
            dropped += u64::from(enc.dropped_objects);
            for &z in &enc.obj_zone {
                zones[z as usize] += 1;
            }
            for k in 0..game.num_choices() {
                let s = &game.choice(k).semantic;
                let id = encode::action_kind(s);
                let name = kind_names.entry(id).or_insert_with(|| {
                    serde_json::to_value(s)
                        .ok()
                        .and_then(|v| v["action_kind"].as_str().map(String::from))
                        .unwrap_or_default()
                });
                *kinds.entry(name.clone()).or_default() += 1;
            }
            let fam = encode::decision_family((game.num_choices() > 0).then(|| &game.choice(0).semantic));
            *families.entry(encode::FAMILIES[fam].to_string()).or_default() += 1;
            let k = bots[actor.index()].choose(&game);
            game.play(k)?;
        }
    }
    Ok(json!({
        "bot": a.bot,
        "games": a.games,
        "net": a.net,
        "net_hash": model.as_ref().map(|m| m.hash.clone()),
        "net_dims": model.as_ref().map(|m| format!("{:?}", m.dims)),
        "decisions": t_enc.len(),
        "microseconds": {
            "observation_build": stats_us(t_obs),
            "encode": stats_us(t_enc),
            "forward": stats_us(t_fwd),
            "forward_hot_cache": stats_us(t_fwd_hot),
            "total": stats_us(t_all),
        },
        "objects_per_decision": stats_us(n_obj),
        "unique_objects_per_decision": stats_us(n_unique),
        "fma_kernels": crate::nn::uses_fma(),
        "actions_per_decision": stats_us(n_act),
        "object_tokens_by_zone": zones.to_vec(),
        "actions_by_kind": kinds,
        "decisions_by_family": families,
        "unlinked_action_refs": unlinked,
        "dropped_objects": dropped,
        "observation_errors": obs_errors,
        "nondeterministic_forwards": nondeterministic,
        "threads": 1,
        "load_average": crate::runner::load_average(),
    }))
}

/// Compare the Rust forward with PyTorch's outputs (`py/parity.py`): `expected` lists, in shard order, each
/// record's `value` and `logits`.
pub fn run_parity(net: &Path, shard: &Path, expected: &Path) -> Result<Value, String> {
    let model = Model::load(net)?;
    let (members, torn) = data::read_shard(shard)?;
    let exp: Value =
        serde_json::from_str(&std::fs::read_to_string(expected).map_err(|e| format!("{}: {e}", expected.display()))?)
            .map_err(|e| format!("{}: {e}", expected.display()))?;
    let want = exp["decisions"].as_array().ok_or("expected.decisions missing")?;
    let mut sc = Scratch::default();
    let (mut max_dl, mut max_dv, mut n, mut n_logits) = (0.0f64, 0.0f64, 0usize, 0usize);
    let mut worst = Value::Null;
    for m in &members {
        for r in &m.records {
            let w = want.get(n).ok_or_else(|| format!("expected has {} decisions, the shard more", want.len()))?;
            model.forward(&r.enc, &mut sc);
            let wl = w["logits"].as_array().ok_or("logits")?;
            if wl.len() != sc.logits.len() {
                return Err(format!("decision {n}: {} logits expected, {} computed", wl.len(), sc.logits.len()));
            }
            for (a, b) in sc.logits.iter().zip(wl) {
                let d = (f64::from(*a) - b.as_f64().unwrap_or(f64::NAN)).abs();
                if d > max_dl || d.is_nan() {
                    max_dl = if d.is_nan() { f64::INFINITY } else { d };
                    worst = json!({"decision": n, "n_act": r.enc.n_act(), "n_obj": r.enc.n_obj()});
                }
            }
            let dv = (f64::from(sc.value) - w["value"].as_f64().unwrap_or(f64::NAN)).abs();
            max_dv = max_dv.max(if dv.is_nan() { f64::INFINITY } else { dv });
            n += 1;
            n_logits += wl.len();
        }
    }
    if n != want.len() {
        return Err(format!("expected has {} decisions, the shard {n}", want.len()));
    }
    Ok(json!({
        "net": net, "net_hash": model.hash, "shard": shard, "torn_member": torn,
        "decisions": n, "logits": n_logits,
        "max_abs_delta_logit": max_dl, "max_abs_delta_value": max_dv, "worst_logit_at": worst,
    }))
}
