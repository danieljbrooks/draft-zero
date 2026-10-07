//! Stage-2 tests: the decision encoder, the network (parity with PyTorch on the committed fixture), the network
//! bots inside `dzk play`'s paired-game runner, and the training data (`--data-out`).

use dzk::bots::mcts::{add_dirichlet, gamma, sample_visits};
use dzk::bots::BotSpec;
use dzk::data::{self, read_shard, shards_in, FLAG_INVALID};
use dzk::deck::{load_deck, Deck};
use dzk::determinize;
use dzk::encode::{self, encode_game, Encoded, ENC_VERSION};
use dzk::game::Game;
use dzk::nn::{Model, Scratch};
use dzk::nntools::run_parity;
use dzk::record::{play_game_data, GameLimits, GameTask};
use dzk::rng::{bot_seed, SplitMix64};
use dzk::runner::{run_play, PlayArgs};
use mtg_kernel::card_def::CARD_DEFS;
use mtg_kernel::rl::{observe_policy_v5_without_projection_hash_v1, ActionSemanticV1, LegalActionV5, ObservationV5};
use mtg_kernel::state::GameState;
use std::path::{Path, PathBuf};

const UG: &str = "FDN_top_04956_UG";
const WG: &str = "FDN_top_20626_WG";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn decks_dir() -> PathBuf {
    root().join("../assets/sample/decks")
}

fn fixture() -> (Deck, Deck) {
    (load_deck(&decks_dir(), UG).unwrap(), load_deck(&decks_dir(), WG).unwrap())
}

fn fixture_net() -> PathBuf {
    root().join("tests/fixtures/parity_v2/net.dzkn")
}

fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("dzk-stage2-{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Every real decision of a few games (random and `mcts:10` play), encoded and checked.
#[test]
fn encodings_are_well_formed_on_real_decisions() {
    assert!(CARD_DEFS.len() < encode::VOCAB - 1, "registry ids fit the vocabulary without the overflow bucket");
    let (ug, wg) = fixture();
    let mut enc = Encoded::default();
    let (mut n, mut dropped) = (0usize, 0u64);
    // (mcts:10 with env_seed 113 is avoided: its Homunculus Horde copies double every turn, 236 tokens by round
    // 16 and a 400-second game; the object cap handles it, but too slowly for a test)
    for (g, spec) in ["random,mull=random", "mcts:10", "random"].iter().enumerate() {
        let spec = BotSpec::parse(spec).unwrap();
        for seed in 0..3u64 {
            let s = 100 + 10 * g as u64 + seed;
            let mut game = Game::new(&ug, &wg, s, s, false).unwrap();
            let mut bots = [spec.build(bot_seed(s, 0)), spec.build(bot_seed(s, 1))];
            while !game.is_terminal() {
                encode_game(&game, &mut enc).unwrap();
                n += 1;
                assert_eq!(enc.global.len(), encode::G);
                assert_eq!(enc.n_act(), game.num_choices());
                assert_eq!(enc.obj_feat.len(), enc.n_obj() * encode::F);
                assert_eq!(enc.act_feat.len(), enc.n_act() * encode::A);
                // only a degenerate random game (a hundred tokens) reaches the object cap
                dropped += u64::from(enc.dropped_objects > 0);
                if enc.dropped_objects == 0 {
                    assert_eq!(enc.unlinked_refs, 0, "every action reference links to an object token");
                }
                // every cap group the observation has objects in keeps tokens (my hand, stack, my battlefield,
                // the opponent's battlefield; the cap is per group)
                let o = game.observation().unwrap();
                let surf = &o.projection.surface;
                let me = o.acting_player;
                let on_bf = |mine: bool| {
                    surf.battlefield.iter().flatten().filter(|c| (c.stable.controller == me) == mine).count()
                };
                let want = [o.own_hand.len(), surf.stack.len(), on_bf(true), on_bf(false)];
                let mut have = [0usize; encode::CAP_GROUPS];
                for &z in &enc.obj_zone {
                    have[encode::cap_group(z)] += 1;
                }
                assert!(enc.n_obj() <= encode::MAX_OBJECTS);
                for g in 0..4 {
                    assert!(want[g] == 0 || have[g] > 0, "group {g}: {} objects, no token", want[g]);
                    assert!(have[g] <= want[g]);
                }
                assert!(enc.global.iter().chain(&enc.obj_feat).chain(&enc.act_feat).all(|x| x.is_finite()));
                for (part, xs, w) in [
                    ("global", &enc.global, encode::G),
                    ("obj", &enc.obj_feat, encode::F),
                    ("act", &enc.act_feat, encode::A),
                ] {
                    if let Some(i) = xs.iter().position(|x| x.abs() > encode::FEATURE_CLAMP) {
                        panic!("{part} feature {} = {}", i % w, xs[i]);
                    }
                }
                assert_eq!(enc.global[56..66].iter().sum::<f32>(), 1.0, "one decision family");
                for k in 0..enc.n_act() {
                    for idx in [enc.act_src[k], enc.act_tgt[k]] {
                        assert!(idx >= -1 && (idx as i64) < enc.n_obj() as i64);
                    }
                    // an action names the card token of its object, not a stack item sharing its arena id
                    let src_zone = (enc.act_src[k] >= 0).then(|| enc.obj_zone[enc.act_src[k] as usize]);
                    match &game.choice(k).semantic {
                        ActionSemanticV1::PlayLand { .. } => assert_eq!(src_zone, Some(encode::Z_MY_HAND)),
                        ActionSemanticV1::ActivateAbility { .. } | ActionSemanticV1::ChooseAttackerInclusion { .. } => {
                            assert_eq!(src_zone, Some(encode::Z_MY_BF))
                        }
                        ActionSemanticV1::ChooseBlockerInclusion { .. } => {
                            assert_eq!(src_zone, Some(encode::Z_MY_BF));
                            assert_eq!(enc.obj_zone[enc.act_tgt[k] as usize], encode::Z_OPP_BF);
                        }
                        _ => {}
                    }
                    assert!((enc.act_kind[k] as usize) < encode::NK && enc.act_tgt_player[k] < 3);
                }
                assert!(enc.obj_card.iter().all(|&c| c > 0 && (c as usize) < encode::VOCAB));
                assert!(enc.obj_zone.iter().all(|&z| (z as usize) < 12));
                let actor = game.actor().unwrap();
                let k = bots[actor.index()].choose(&game);
                game.play(k).unwrap();
            }
        }
    }
    eprintln!("{n} decisions encoded, {dropped} at the object cap");
    assert!(n > 400 && dropped * 50 < n as u64);
}

/// The actor's observation, rebuilt from `state` at `game`'s current decision (as the session builds it), with
/// the stored observation's projection hash (a pure function of the other fields) copied over.
fn rebuild_observation(game: &Game, state: &GameState) -> ObservationV5 {
    let at = game.decision_coords().unwrap();
    let mut o = observe_policy_v5_without_projection_hash_v1(
        state,
        game.session().policy_surface_v1(),
        game.actor().unwrap(),
        at.step,
        at.physical_decision_id,
        at.substep_index,
        at.substep_count,
    )
    .unwrap();
    o.visible_projection_hash = game.observation().unwrap().visible_projection_hash;
    o
}

/// The encoding depends on the actor's view only. At real decisions of mulliganing random play and of
/// `mcts:8,mull=own` play, the state's hidden cards (the opponent's hand, every library position the actor does
/// not know) are re-dealt, the actor's observation is REBUILT from the re-dealt state (the session's stored
/// observation would be the real one), and encoded with the same masked actions: it must equal the real
/// encoding. The bots' own world builder (`determinize::redeal`) must never be refused for a changed observation.
#[test]
fn encoding_ignores_hidden_cards() {
    let gen = root().join("decks/gen_v1");
    let mut deckpairs = vec![fixture()];
    for (a, b) in dzk::runner::read_pairs(&gen.join("pairs_train.tsv")).unwrap().iter().take(3) {
        deckpairs.push((load_deck(&gen, a).unwrap(), load_deck(&gen, b).unwrap()));
    }
    let (mut real, mut redealt) = (Encoded::default(), Encoded::default());
    let (mut checked, mut opp_hand_changed, mut own_library_changed, mut worlds) = (0, 0, 0, 0);
    for (g, spec) in ["random,mull=random", "mcts:8,mull=own"].iter().enumerate() {
        let spec = BotSpec::parse(spec).unwrap();
        for (pi, (d1, d2)) in deckpairs.iter().enumerate() {
            let s = 300 + 10 * g as u64 + pi as u64;
            let mut game = Game::new(d1, d2, s, s, false).unwrap();
            let mut bots = [spec.build(bot_seed(s, 0)), spec.build(bot_seed(s, 1))];
            let mut rng = SplitMix64::new(s);
            let mut step = 0;
            while !game.is_terminal() && step < 300 {
                if step % 2 == 0 {
                    let viewer = game.actor().unwrap();
                    let actions: Vec<&LegalActionV5> = (0..game.num_choices()).map(|k| game.choice(k)).collect();
                    encode_game(&game, &mut real).unwrap();
                    assert_eq!(&rebuild_observation(&game, game.state()), game.observation().unwrap());
                    for _ in 0..2 {
                        let mut st = game.state().clone();
                        determinize::redeal_state(&mut st, viewer, &mut rng).unwrap();
                        let o = rebuild_observation(&game, &st);
                        assert_eq!(&o, game.observation().unwrap(), "observation changed: seed {s}, step {step}");
                        encode::encode(&o, &actions, &mut redealt);
                        assert_eq!(real, redealt, "encoding changed: seed {s}, step {step}");
                        let cards = |st: &GameState, p: usize, hand: bool| {
                            let ps = &st.players[p];
                            let ids = if hand { &ps.hand } else { &ps.library };
                            ids.iter().map(|&id| st.objects.get(id).card_def).collect::<Vec<_>>()
                        };
                        let (me, opp) = (viewer.index(), viewer.opponent().index());
                        opp_hand_changed += usize::from(cards(&st, opp, true) != cards(game.state(), opp, true));
                        own_library_changed += usize::from(cards(&st, me, false) != cards(game.state(), me, false));
                        checked += 1;
                    }
                    match determinize::redeal(&game, &mut rng) {
                        Ok(_) => worlds += 1,
                        Err(e) => panic!("redeal refused ({e:?}): seed {s}, step {step}"),
                    }
                }
                let k = bots[game.actor().unwrap().index()].choose(&game);
                game.play(k).unwrap();
                step += 1;
            }
        }
    }
    eprintln!(
        "{checked} re-deals compared ({opp_hand_changed} changed the opponent's hand, {own_library_changed} the \
         actor's library order); {worlds} worlds admitted"
    );
    assert!(checked > 400 && opp_hand_changed > checked / 2 && own_library_changed > checked / 2);
}

/// The Rust forward matches PyTorch on the committed fixture: py/parity.py's stratified sample of real decisions
/// (gen-0 `mcts:100` self-play, `random,mull=random` and `mcts:10,mull=own` play, a runaway token game): every
/// decision family, 20 action kinds, menus up to 5,040 actions.
#[test]
fn nn_parity_matches_pytorch() {
    let d = root().join("tests/fixtures/parity_v2");
    let r = run_parity(&d.join("net.dzkn"), &d.join("decisions.dzd.gz"), &d.join("expected.json")).unwrap();
    eprintln!("{}", serde_json::to_string(&r).unwrap());
    let exp: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("expected.json")).unwrap()).unwrap();
    for f in encode::FAMILIES {
        assert!(exp["families"][f].as_u64().unwrap_or(0) > 0, "the fixture has no {f} decision");
    }
    assert!(exp["max_actions"].as_u64().unwrap() >= 720);
    assert!(r["decisions"].as_u64().unwrap() >= 200);
    assert!(r["max_abs_delta_logit"].as_f64().unwrap() < 1e-4);
    assert!(r["max_abs_delta_value"].as_f64().unwrap() < 1e-4);
}

#[test]
fn nn_file_is_checked() {
    let bytes = std::fs::read(fixture_net()).unwrap();
    let m = Model::from_bytes(&bytes, "fixture").unwrap();
    assert_eq!(m.dims.h, 128);
    // a flipped blob byte fails the hash; another encoder version is refused
    let mut bad = bytes.clone();
    let n = bad.len();
    bad[n - 3] ^= 1;
    assert!(Model::from_bytes(&bad, "x").unwrap_err().contains("sha256"));
    let text = String::from_utf8_lossy(&bytes[16..200]).to_string();
    assert!(text.contains(ENC_VERSION) || String::from_utf8_lossy(&bytes).contains(ENC_VERSION));
    let swapped: Vec<u8> = {
        let s = bytes.clone();
        let pos = s.windows(ENC_VERSION.len()).position(|w| w == ENC_VERSION.as_bytes()).unwrap();
        let mut s2 = s.clone();
        s2[pos + ENC_VERSION.len() - 1] = b'9';
        s2
    };
    assert!(Model::from_bytes(&swapped, "x").unwrap_err().contains("encoder"));
    assert!(Model::from_bytes(b"not a net", "x").is_err());
}

/// The forward is deterministic, and computing identical object tokens once changes nothing.
#[test]
fn nn_forward_is_deterministic() {
    let m = Model::load(&fixture_net()).unwrap();
    let (members, _) = read_shard(&root().join("tests/fixtures/parity_v2/decisions.dzd.gz")).unwrap();
    let (mut s1, mut s2) = (Scratch::default(), Scratch::default());
    let mut dups = 0;
    for r in members.iter().flat_map(|m| &m.records) {
        m.forward(&r.enc, &mut s1);
        m.forward(&r.enc, &mut s2);
        assert_eq!((s1.value, &s1.logits), (s2.value, &s2.logits));
        dups += r.enc.n_obj() - s1.unique_objects;
        assert!(s1.value > -1.0 && s1.value < 1.0 && s1.logits.iter().all(|x| x.is_finite()));
    }
    assert!(dups > 0, "the fixture has identical object tokens (dedup is exercised)");
}

#[test]
fn network_specs_parse() {
    let p = fixture_net().display().to_string();
    let BotSpec::Net(c) = BotSpec::parse(&format!("net:path={p}")).unwrap() else { panic!() };
    assert_eq!(c.temperature, 0.0);
    let BotSpec::Net(c) = BotSpec::parse(&format!("net,path={p},t=1,mull=xmage")).unwrap() else { panic!() };
    assert_eq!(c.temperature, 1.0);
    assert_eq!(BotSpec::parse(&format!("net:path={p}")).unwrap(), BotSpec::parse(&format!("net,path={p}")).unwrap());
    let BotSpec::Mcts(c) = BotSpec::parse(&format!(
        "pmcts:50,net={p},pt=0.5,leaf=mix:0.25,noise=0.25,alpha=0.3,temp_moves=8,worlds=2,fpu=0.2"
    ))
    .unwrap() else {
        panic!()
    };
    let n = c.net.as_ref().unwrap();
    assert_eq!((c.budget, c.worlds, n.prior_temp, n.temp_moves), (50, 2, 0.5, 8));
    assert_eq!(n.leaf, dzk::bots::mcts::Leaf::Mix(0.25));
    assert_eq!(n.fpu, Some(0.2));
    assert!(BotSpec::parse(&format!("pmcts:10,net={p},fpu=-1")).is_err());
    assert!(BotSpec::parse("net").is_err(), "net needs path=");
    assert!(BotSpec::parse("pmcts:10").is_err(), "pmcts needs net=");
    assert!(BotSpec::parse(&format!("pmcts:10,net={p},leaf=mix:2")).is_err());
    assert!(BotSpec::parse(&format!("mcts:10,net={p}")).is_err(), "mcts takes no network");
    assert!(BotSpec::parse("net:path=/nonexistent.dzkn").is_err());
    assert_ne!(
        BotSpec::parse(&format!("pmcts:10,net={p}")).unwrap(),
        BotSpec::parse(&format!("pmcts:10,net={p},noise=0.25")).unwrap()
    );
}

#[test]
fn dirichlet_and_visit_sampling() {
    let mut rng = SplitMix64::new(9);
    for alpha in [0.3, 1.0, 2.5] {
        let n = 20000;
        let mean = (0..n).map(|_| gamma(alpha, &mut rng)).sum::<f64>() / n as f64;
        assert!((mean - alpha).abs() < 0.05 * alpha.max(1.0), "gamma({alpha}) mean {mean}");
    }
    let mut p = vec![0.5, 0.25, 0.25];
    add_dirichlet(&mut p, 0.25, 0.3, &mut rng);
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9 && p.iter().all(|&x| x > 0.0));
    let mut counts = [0u32; 3];
    for _ in 0..30000 {
        counts[sample_visits(&[60, 30, 10], 1.0, &mut rng)] += 1;
    }
    let f: Vec<f64> = counts.iter().map(|&c| c as f64 / 30000.0).collect();
    assert!((f[0] - 0.6).abs() < 0.02 && (f[1] - 0.3).abs() < 0.02 && (f[2] - 0.1).abs() < 0.02, "{f:?}");
    assert_eq!(sample_visits(&[0, 0], 1.0, &mut rng), 0);
}

/// `net`, `net:t=1` and `pmcts` play inside the paired-game runner; their records are like the other bots'
/// (bot counters in `search`), deterministic, and the data holds every real decision with its result.
#[test]
fn network_bots_play_paired_games_with_data() {
    let dir = scratch("netplay");
    std::fs::write(dir.join("pairs.tsv"), format!("{UG}\t{WG}\n{WG}\t{UG}\n")).unwrap();
    let p = fixture_net().display().to_string();
    let cases = [
        (format!("net:path={p}"), "random".to_string()),
        (format!("net:path={p},t=1"), "mcts:10".to_string()),
        (format!("pmcts:12,net={p},noise=0.25,temp_moves=4"), "mcts:12".to_string()),
        (format!("pmcts:12,net={p},leaf=mix:0.5,worlds=2"), format!("pmcts:12,net={p},leaf=heuristic")),
    ];
    for (i, (b1, b2)) in cases.iter().enumerate() {
        let mut lines = Vec::new();
        for rep in 0..2 {
            let sub = dir.join(format!("c{i}r{rep}"));
            let args = PlayArgs {
                decks_dir: decks_dir(),
                pairs_file: dir.join("pairs.tsv"),
                bot1: b1.clone(),
                bot2: b2.clone(),
                seed: 21,
                threads: 2,
                out: sub.join("games.jsonl"),
                n_pairs: Some(2),
                pair_range: None,
                shard: None,
                resume: false,
                data_out: Some(sub.join("data")),
                limits: Default::default(),
            };
            let s = run_play(&args).unwrap();
            assert_eq!(s["games"], 4);
            assert_eq!(s["errors"], 0, "{b1} vs {b2}: {s}");
            let text = std::fs::read_to_string(sub.join("games.jsonl")).unwrap();
            let mut recs: Vec<serde_json::Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
            recs.sort_by_key(|r| (r["pair"].as_u64(), r["swap"].as_bool()));
            // the data: one record per real decision of either seat, results from each record's seat
            let mut n_data = 0u64;
            let mut games = 0;
            for shard in shards_in(&sub.join("data")).unwrap() {
                let (members, torn) = read_shard(&shard).unwrap();
                assert!(!torn);
                for m in members {
                    games += 1;
                    let rec =
                        recs.iter().find(|r| r["pair"] == m.header["pair"] && r["swap"] == m.header["swap"]).unwrap();
                    let real =
                        rec["decisions"]["real_A"].as_u64().unwrap() + rec["decisions"]["real_B"].as_u64().unwrap();
                    assert_eq!(m.records.len() as u64, real);
                    for r in &m.records {
                        n_data += 1;
                        let seat = if r.seat == 0 { "A" } else { "B" };
                        let want = match rec["winner"].as_str() {
                            Some(w) if w == seat => 1,
                            Some(_) => -1,
                            None => 0,
                        };
                        assert_eq!(r.z, want);
                        assert_eq!(r.flags & FLAG_INVALID, 0);
                        assert!((r.target.iter().sum::<f32>() - 1.0).abs() < 1e-5);
                        assert!((r.chosen as usize) < r.enc.n_act());
                        let bot = rec[if seat == "A" { "botA" } else { "botB" }].as_str().unwrap();
                        let source = data::SOURCE_NAMES[r.source as usize];
                        let expect = if bot.starts_with("net") {
                            ["net", "rule"].as_slice()
                        } else if bot.contains("mcts") {
                            ["search", "rule"].as_slice()
                        } else {
                            ["random", "rule"].as_slice()
                        };
                        assert!(expect.contains(&source), "{bot}: source {source}");
                    }
                }
            }
            assert_eq!(games, 4);
            assert!(n_data > 0);
            let meta: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(sub.join("data/meta.json")).unwrap()).unwrap();
            assert_eq!(meta["totals"]["records"].as_u64(), Some(n_data));
            assert_eq!(meta["encoder"], ENC_VERSION);
            for r in &recs {
                // the networks' hashes ride on the record (resume and the data's game keys use them)
                for (role, spec) in [("bot1", b1), ("bot2", b2)] {
                    let want = BotSpec::parse(spec).unwrap().net_hash();
                    assert_eq!(r[format!("{role}_net")].as_str().map(String::from), want, "{role} {spec}");
                }
                for seat in ["A", "B"] {
                    let bot = r[if seat == "A" { "botA" } else { "botB" }].as_str().unwrap();
                    let st = &r["search"][seat];
                    if bot.starts_with("pmcts") {
                        assert!(st["net_evals"].as_u64().unwrap() > 0 && st["simulations"].as_u64().unwrap() > 0);
                        assert_eq!(st["encode_failures"], 0);
                    } else if bot.starts_with("net") {
                        assert!(st["net_decisions"].as_u64().unwrap() > 0);
                    }
                }
            }
            lines.push(
                recs.iter()
                    .map(|r| {
                        let mut r = r.clone();
                        r["seconds"] = serde_json::json!(0);
                        r.to_string()
                    })
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(lines[0], lines[1], "{b1} vs {b2}: records are deterministic");
    }
}

/// A halted game's data is kept for the policy but flagged as no value target (z = 0).
#[test]
fn data_of_a_game_without_a_valid_result_is_flagged() {
    let (ug, wg) = fixture();
    let random = BotSpec::parse("random,mull=random").unwrap();
    // env_seed 1746, UG vs WG with random,mull=random halts (stage 1's Ward observation fail-close)
    let t = GameTask { pair: 0, swap: false, deck1: UG.into(), deck2: WG.into(), game_seed: 1746, env_seed: 1746 };
    let (rec, _, d) = play_game_data(&t, &ug, &wg, ("r", &random), ("r", &random), true, &Default::default());
    let d = d.unwrap();
    assert!(rec.error.as_deref().is_some_and(|e| e.starts_with("halted:")), "{:?}", rec.error);
    assert!(!d.records.is_empty());
    assert!(d.records.iter().all(|r| r.flags & FLAG_INVALID != 0 && r.z == 0));
}

/// A torn last game member (a killed run) is dropped and reported; the complete ones read.
#[test]
fn a_torn_shard_keeps_its_complete_games() {
    let src = root().join("tests/fixtures/parity_v2/decisions.dzd.gz");
    let (full, torn) = read_shard(&src).unwrap();
    assert!(!torn && full.len() >= 2);
    let bytes = std::fs::read(&src).unwrap();
    let dir = scratch("torn");
    let cut = dir.join("cut.dzd.gz");
    std::fs::write(&cut, &bytes[..bytes.len() - 10]).unwrap(); // inside the last member's trailer
    let (part, torn) = read_shard(&cut).unwrap();
    assert!(torn);
    assert_eq!(part.len(), full.len() - 1);
    // the counting reader (a killed run's recount) agrees without decoding the records
    for (shard, members, torn) in [(&src, &full, false), (&cut, &part, true)] {
        let (g, r, by_source, t) = data::count_shard(shard).unwrap();
        let want_r: usize = members.iter().map(|m| m.records.len()).sum();
        assert_eq!((g as usize, r as usize, t, by_source.iter().sum::<u64>()), (members.len(), want_r, torn, r));
        let search = members.iter().flat_map(|m| &m.records).filter(|x| x.source == 0).count();
        assert_eq!(by_source[0] as usize, search);
    }
    for (a, b) in part.iter().zip(&full) {
        assert_eq!(a.records.len(), b.records.len());
        for (x, y) in a.records.iter().zip(&b.records) {
            assert_eq!((&x.enc, &x.target, x.chosen, x.z), (&y.enc, &y.target, y.chosen, y.z));
            // root_q may be NaN
        }
    }
}

/// A copy of the fixture network with one weight changed (a valid dzk-net-v1 file with another hash).
fn altered_net(to: &Path) {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(fixture_net()).unwrap();
    let hl = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[16..16 + hl]).unwrap();
    let mut blob = bytes[16 + hl..].to_vec();
    let n = blob.len();
    let last = f32::from_le_bytes(blob[n - 4..].try_into().unwrap()) + 0.5; // the value head's bias
    blob[n - 4..].copy_from_slice(&last.to_le_bytes());
    header["blob_sha256"] = serde_json::json!(format!("{:x}", Sha256::digest(&blob)));
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = bytes[..8].to_vec();
    out.extend_from_slice(&(h.len() as u64).to_le_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(&blob);
    std::fs::write(to, out).unwrap();
}

/// A network file replaced under the same path: `--resume` refuses the games file (it would mix two networks'
/// games), and the data's game keys carry the network's hash.
#[test]
fn resume_refuses_another_network_under_the_same_path() {
    let dir = scratch("netswap");
    std::fs::write(dir.join("pairs.tsv"), format!("{UG}\t{WG}\n")).unwrap();
    let net = dir.join("net.dzkn");
    std::fs::copy(fixture_net(), &net).unwrap();
    let bot1 = format!("net:path={}", net.display());
    let args = |n_pairs: usize, resume: bool| PlayArgs {
        decks_dir: decks_dir(),
        pairs_file: dir.join("pairs.tsv"),
        bot1: bot1.clone(),
        bot2: "random".into(),
        seed: 3,
        threads: 1,
        out: dir.join("games.jsonl"),
        n_pairs: Some(n_pairs),
        pair_range: None,
        shard: None,
        resume,
        data_out: Some(dir.join("data")),
        limits: Default::default(),
    };
    run_play(&args(1, false)).unwrap();
    let old_hash = BotSpec::parse(&bot1).unwrap().net_hash().unwrap();
    for shard in shards_in(&dir.join("data")).unwrap() {
        for m in read_shard(&shard).unwrap().0 {
            assert!(m.header["game_key"].as_str().unwrap().ends_with(&format!("|{old_hash}|")), "{}", m.header);
        }
    }
    altered_net(&net);
    let new_hash = BotSpec::parse(&bot1).unwrap().net_hash().unwrap();
    assert_ne!(old_hash, new_hash, "the rewritten file is loaded again");
    let err = run_play(&args(2, true)).unwrap_err();
    assert!(err.contains("networks") && err.contains(&new_hash) && err.contains(&old_hash), "{err}");
    // the original network back: the run resumes and adds pair 1
    std::fs::copy(fixture_net(), &net).unwrap();
    let s = run_play(&args(2, true)).unwrap();
    assert_eq!((s["games"].as_u64(), s["bot1_net"].as_str()), (Some(4), Some(old_hash.as_str())));
}

/// A game that reaches the permanent limit ends as truncated: no winner, flagged data (z = 0).
#[test]
fn game_limits_truncate() {
    let (ug, wg) = fixture();
    let random = BotSpec::parse("random").unwrap();
    let t = GameTask { pair: 0, swap: false, deck1: UG.into(), deck2: WG.into(), game_seed: 5, env_seed: 5 };
    let limits = GameLimits { max_permanents: 6, max_seconds: 0.0 };
    let (rec, _, d) = play_game_data(&t, &ug, &wg, ("r", &random), ("r", &random), true, &limits);
    assert_eq!(rec.terminal.classification.as_deref(), Some("truncated"));
    assert_eq!(rec.terminal.code.as_deref(), Some("dzk_permanent_cap"));
    assert!(rec.winner.is_none() && rec.error.is_none());
    let d = d.unwrap();
    assert!(!d.records.is_empty() && d.records.iter().all(|r| r.flags == data::FLAG_TRUNCATED && r.z == 0));
    let s = dzk::runner::summarize(&[rec]);
    assert_eq!(s["terminals"]["truncated_by_code"]["dzk_permanent_cap"], 1);
    assert_eq!(s["bot1_score"], 0.5);
}
