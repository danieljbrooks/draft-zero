//! `net:path=FILE[,t=T][,mull=P]`: play the network's policy. `t = 0` (the default) is greedy (argmax logit,
//! ties to the lower index; draft-zero's rule: strength evaluations play the policy greedy); `t > 0` samples
//! `softmax(logits / t)` from the bot's own generator. `net,path=FILE,...` is the same spec.
//!
//! The bot sees only the acting player's observation and the masked actions (`encode::encode_game`).
//! Mulligans: `mull=` as for the other bots, default `keep` (the gen-0 search targets for the mulligan
//! announcement are degenerate: the searcher's own `mull=keep` answers them by rule, so no policy head was
//! trained on them; `mull=own` lets the network answer). If a decision cannot be encoded (no observation;
//! not seen in practice), the bot picks uniformly and counts it (`encode_failures`).

use super::{Bot, MulliganPolicy, MulliganStats};
use crate::data::{PolicyTarget, Source};
use crate::encode::{encode_game, Encoded};
use crate::game::Game;
use crate::nn::{argmax, softmax, NetRef, Scratch};
use crate::rng::SplitMix64;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct NetConfig {
    pub net: NetRef,
    pub temperature: f32,
    pub mulligan: MulliganPolicy,
}

impl NetConfig {
    pub fn parse<'a>(opts: impl Iterator<Item = &'a str>, spec: &str) -> Result<NetConfig, String> {
        let mut path = None;
        let mut temperature = 0.0f32;
        let mut mulligan = MulliganPolicy::Keep;
        for kv in opts {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("bad option {kv:?} in {spec:?}"))?;
            let v = v.trim();
            match k.trim() {
                "path" => path = Some(v.to_string()),
                "t" => {
                    temperature = v
                        .parse()
                        .ok()
                        .filter(|t: &f32| t.is_finite() && *t >= 0.0)
                        .ok_or_else(|| format!("bad value for t in {spec:?}"))?
                }
                "mull" => {
                    mulligan = MulliganPolicy::parse(v).ok_or_else(|| format!("bad value for mull in {spec:?}"))?
                }
                other => return Err(format!("unknown net option {other:?} in {spec:?}")),
            }
        }
        let path = path.ok_or_else(|| format!("{spec:?}: the net bot needs path=FILE"))?;
        Ok(NetConfig { net: NetRef::load(&path)?, temperature, mulligan })
    }
}

/// Sample an index from `probs` (sums to 1) with one uniform draw.
pub fn sample(probs: &[f32], rng: &mut SplitMix64) -> usize {
    let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let mut acc = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        acc += f64::from(p);
        if u < acc {
            return i;
        }
    }
    probs.len() - 1
}

pub struct NetBot {
    cfg: NetConfig,
    rng: SplitMix64,
    mull: MulliganStats,
    enc: Encoded,
    sc: Scratch,
    probs: Vec<f32>,
    decisions: u64,
    encode_failures: u64,
    last: Option<Source>,
}

impl NetBot {
    pub fn new(cfg: NetConfig, seed: u64) -> Self {
        NetBot {
            cfg,
            rng: SplitMix64::new(seed),
            mull: MulliganStats::default(),
            enc: Encoded::default(),
            sc: Scratch::default(),
            probs: Vec::new(),
            decisions: 0,
            encode_failures: 0,
            last: None,
        }
    }
}

impl Bot for NetBot {
    fn choose(&mut self, game: &Game) -> usize {
        if let Some(k) = self.mull.apply(self.cfg.mulligan, game) {
            self.last = Some(Source::Rule);
            return k;
        }
        self.last = Some(Source::Net);
        if encode_game(game, &mut self.enc).is_err() {
            self.encode_failures += 1;
            return self.rng.below(game.num_choices() as u64) as usize;
        }
        self.cfg.net.model.forward(&self.enc, &mut self.sc);
        self.decisions += 1;
        if self.cfg.temperature <= 0.0 {
            argmax(&self.sc.logits)
        } else {
            softmax(&self.sc.logits, self.cfg.temperature, &mut self.probs);
            sample(&self.probs, &mut self.rng)
        }
    }

    fn stats(&self) -> Value {
        json!({
            "net": self.cfg.net.model.hash,
            "net_decisions": self.decisions,
            "encode_failures": self.encode_failures,
            "mulligan_decisions": self.mull.decisions,
            "mulligans_taken": self.mull.taken,
        })
    }

    fn take_target(&mut self) -> Option<PolicyTarget> {
        self.last.take().map(PolicyTarget::onehot)
    }
}
