//! Bots: a bot is consulted only at a real decision of its own seat (2+ masked actions) and returns an index
//! into the masked legal list (`Game::choice(k)`).
//!
//! Specs: `first`, `random[,mull=P]`, `greedy1[,mull=P]`,
//! `mcts:N[,worlds=K][,c=C][,discount=D][,cache=M][,mull=P]` (`heuristic@N` is accepted as an alias of
//! `mcts:N`, draft-zero's name for the same search), and the network bots (stage 2)
//! `net:path=FILE[,t=T][,mull=P]` (greedy at t = 0, the default; sampled at t > 0) and
//! `pmcts:N,net=FILE[,pt=T][,leaf=net|heuristic|mix:L][,noise=E][,alpha=A][,temp=T][,temp_moves=K][,fpu=R]` plus the
//! `mcts:N` options. Every bot gets its own SplitMix64 seeded from the game seed and its seat
//! (`rng::bot_seed`), independent of the deck shuffle.
//!
//! Hidden information is a convention, not enforced: `choose` gets the whole `Game` (its `state()` shows
//! every hidden card). The search bots (`greedy1`, `mcts:N`) only look ahead on re-dealt worlds
//! (`determinize::redeal`); a bot that reads hidden state directly is cheating.

pub mod mcts;
pub mod net;

use crate::data::{PolicyTarget, Source};
use crate::determinize;
use crate::eval;
use crate::game::Game;
use crate::rng::SplitMix64;
use mtg_kernel::card_def::CARD_DEFS;
use mtg_kernel::ids::PlayerId;
use mtg_kernel::rl::ActionSemanticV1;
use serde_json::{json, Value};

pub trait Bot: Send {
    /// An index into the masked legal list of `game`'s current decision (which is this bot's).
    fn choose(&mut self, game: &Game) -> usize;
    /// Deterministic per-game counters for the record (no timings).
    fn stats(&self) -> Value {
        Value::Null
    }
    /// The policy target of the last `choose` (for `dzk play --data-out`), taken once. `None`: one-hot of
    /// the chosen action, source "other".
    fn take_target(&mut self) -> Option<PolicyTarget> {
        None
    }
}

/// How a bot answers its own London mulligan announcements (`mull=`).
///
/// The Java baseline never mulligans: draft-zero's XMage games run in test mode with
/// `allowMulligans = false`, so `ComputerPlayer.chooseMulligan` always keeps. `keep` (every bot's default)
/// does the same. `xmage` is that method's rule outside test mode: mulligan a hand of 6+ cards with fewer
/// than 2 lands or more than size - 2. XMage's `LondonMulligan` bottoms right after each redraw, as
/// mtg-kernel does, so the rule sees the hand after bottoming (6 cards after one mulligan, which it then
/// always keeps). `own` (also spelled `search` or `random`) leaves the announcement to the bot like any
/// decision: uniform for `random`, a search for `mcts:N`, the one-ply score for `greedy1`. Bottom choices
/// are always the bot's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulliganPolicy {
    Keep,
    Xmage,
    Own,
}

impl MulliganPolicy {
    fn parse(v: &str) -> Option<Self> {
        match v.trim() {
            "keep" => Some(Self::Keep),
            "xmage" => Some(Self::Xmage),
            "own" | "search" | "random" => Some(Self::Own),
            _ => None,
        }
    }
}

/// The masked index of the `mulligan: want` answer of a London mulligan announcement, if this is one.
fn mulligan_answer(game: &Game, want: bool) -> Option<usize> {
    (0..game.num_choices()).find(|&k| {
        matches!(game.choice(k).semantic, ActionSemanticV1::ChooseLondonMulligan { mulligan, .. } if mulligan == want)
    })
}

/// XMage `ComputerPlayer.chooseMulligan` without test mode: mulligan a 6+ card hand with < 2 or > size - 2 lands.
fn xmage_mulligan(game: &Game, me: PlayerId) -> bool {
    let s = game.state();
    let hand = &s.players[me.index()].hand;
    let lands = hand.iter().filter(|&&id| CARD_DEFS[s.objects.get(id).card_def as usize].is_land).count();
    hand.len() >= 6 && (lands < 2 || lands + 2 > hand.len())
}

/// The policy's answer when `game`'s current decision is `me`'s London mulligan announcement and the policy
/// is not `Own`; `None` otherwise. Reads only `me`'s own hand.
pub fn mulligan_override(policy: MulliganPolicy, game: &Game, me: PlayerId) -> Option<usize> {
    if policy == MulliganPolicy::Own || game.actor() != Some(me) {
        return None;
    }
    let keep = mulligan_answer(game, false)?;
    let mull = mulligan_answer(game, true)?;
    Some(if policy == MulliganPolicy::Xmage && xmage_mulligan(game, me) { mull } else { keep })
}

/// Mulligan announcements answered by the policy, and how many of them were mulligans.
#[derive(Debug, Clone, Default)]
pub struct MulliganStats {
    pub decisions: u64,
    pub taken: u64,
}

impl MulliganStats {
    fn apply(&mut self, policy: MulliganPolicy, game: &Game) -> Option<usize> {
        let me = game.actor()?;
        let k = mulligan_override(policy, game, me)?;
        self.decisions += 1;
        self.taken += u64::from(mulligan_answer(game, true) == Some(k));
        Some(k)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BotSpec {
    Random {
        mulligan: MulliganPolicy,
    },
    First,
    Greedy1 {
        mulligan: MulliganPolicy,
    },
    /// `mcts:N` (no network) and `pmcts:N` (`cfg.net` set).
    Mcts(mcts::MctsConfig),
    Net(net::NetConfig),
}

const SPECS: &str =
    "first | random[,mull=P] | greedy1[,mull=P] | mcts:N[,worlds=K][,c=C][,discount=D][,cache=M][,mull=P] \
     | net:path=FILE[,t=T][,mull=P] | pmcts:N,net=FILE[,pt=T][,leaf=net|heuristic|mix:L][,noise=E][,alpha=A]\
     [,temp=T][,temp_moves=K][,fpu=R][,worlds=K][,c=C][,discount=D][,cache=M][,mull=P] (P = keep | xmage | own)";

impl BotSpec {
    pub fn parse(spec: &str) -> Result<BotSpec, String> {
        let spec = spec.trim();
        if spec == "first" {
            return Ok(BotSpec::First);
        }
        let mut parts = spec.split(',');
        let head = parts.next().unwrap_or_default().trim();
        let bad = |k: &str| format!("bad value for {k} in {spec:?}");
        let mulligan_only = |parts: std::str::Split<'_, char>| -> Result<MulliganPolicy, String> {
            let mut mulligan = MulliganPolicy::Keep;
            for kv in parts {
                let (k, v) = kv.split_once('=').ok_or_else(|| format!("bad option {kv:?} in {spec:?}"))?;
                match k.trim() {
                    "mull" => mulligan = MulliganPolicy::parse(v).ok_or_else(|| bad(k))?,
                    other => return Err(format!("unknown option {other:?} in {spec:?}")),
                }
            }
            Ok(mulligan)
        };
        match head {
            "random" => return Ok(BotSpec::Random { mulligan: mulligan_only(parts)? }),
            "greedy1" => return Ok(BotSpec::Greedy1 { mulligan: mulligan_only(parts)? }),
            _ => {}
        }
        if head == "net" || head.starts_with("net:") {
            // `net:path=F,t=1` and `net,path=F,t=1` are the same bot
            let first = head.strip_prefix("net:").filter(|r| !r.trim().is_empty());
            return Ok(BotSpec::Net(net::NetConfig::parse(first.into_iter().chain(parts), spec)?));
        }
        let (budget, with_net) = if let Some(r) = head.strip_prefix("mcts:") {
            (r, false)
        } else if let Some(r) = head.strip_prefix("heuristic@") {
            (r, false)
        } else if let Some(r) = head.strip_prefix("pmcts:") {
            (r, true)
        } else {
            return Err(format!("unknown bot {spec:?}: {SPECS}"));
        };
        let budget: u32 = budget
            .trim()
            .parse()
            .ok()
            .filter(|&b| b >= 1)
            .ok_or_else(|| format!("bad simulation budget in {spec:?}"))?;
        let mut cfg = mcts::MctsConfig { budget, ..mcts::MctsConfig::default() };
        let mut ns = mcts::NetSearchOptions::default();
        for kv in parts {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("bad option {kv:?} in {spec:?}"))?;
            if with_net && ns.set(k.trim(), v.trim()).map_err(|e| format!("{e} in {spec:?}"))? {
                continue;
            }
            match k.trim() {
                "worlds" => cfg.worlds = v.trim().parse().ok().filter(|&w| w >= 1).ok_or_else(|| bad(k))?,
                "c" => {
                    cfg.c_puct =
                        v.trim().parse().ok().filter(|c: &f64| c.is_finite() && *c >= 0.0).ok_or_else(|| bad(k))?
                }
                "discount" => {
                    cfg.discount =
                        v.trim().parse().ok().filter(|d: &f64| *d > 0.0 && *d <= 1.0).ok_or_else(|| bad(k))?
                }
                "cache" => cfg.max_cached = v.trim().parse().ok().ok_or_else(|| bad(k))?,
                "mull" => cfg.mulligan = MulliganPolicy::parse(v).ok_or_else(|| bad(k))?,
                other => return Err(format!("unknown mcts option {other:?} in {spec:?}")),
            }
        }
        if with_net {
            cfg.net = Some(ns.finish().map_err(|e| format!("{e} in {spec:?}"))?);
        }
        Ok(BotSpec::Mcts(cfg))
    }

    /// The first 16 hex digits of the network's sha256, for the network bots (`net`, `pmcts`).
    pub fn net_hash(&self) -> Option<String> {
        match self {
            BotSpec::Net(c) => Some(c.net.model.hash.clone()),
            BotSpec::Mcts(c) => c.net.as_ref().map(|n| n.net.model.hash.clone()),
            _ => None,
        }
    }

    pub fn build(&self, seed: u64) -> Box<dyn Bot> {
        match self {
            BotSpec::Random { mulligan } => Box::new(RandomBot {
                rng: SplitMix64::new(seed),
                mulligan: *mulligan,
                mull: MulliganStats::default(),
                last: None,
            }),
            BotSpec::First => Box::new(FirstBot),
            BotSpec::Greedy1 { mulligan } => Box::new(Greedy1Bot::new(seed, *mulligan)),
            BotSpec::Mcts(cfg) => Box::new(mcts::MctsBot::new(cfg.clone(), seed)),
            BotSpec::Net(cfg) => Box::new(net::NetBot::new(cfg.clone(), seed)),
        }
    }
}

/// Uniform over the masked legal actions (mulligan announcements by `mull=`, default keep).
pub struct RandomBot {
    rng: SplitMix64,
    mulligan: MulliganPolicy,
    mull: MulliganStats,
    last: Option<Source>,
}

impl Bot for RandomBot {
    fn choose(&mut self, game: &Game) -> usize {
        if let Some(k) = self.mull.apply(self.mulligan, game) {
            self.last = Some(Source::Rule);
            return k;
        }
        self.last = Some(Source::Random);
        self.rng.below(game.num_choices() as u64) as usize
    }

    fn stats(&self) -> Value {
        json!({ "mulligan_decisions": self.mull.decisions, "mulligans_taken": self.mull.taken })
    }

    fn take_target(&mut self) -> Option<PolicyTarget> {
        self.last.take().map(PolicyTarget::onehot)
    }
}

/// Always the first masked action (a sanity baseline, close to "pass first"; mulligans included).
pub struct FirstBot;

impl Bot for FirstBot {
    fn choose(&mut self, _game: &Game) -> usize {
        0
    }
}

/// How often a world is re-dealt before a search gives that world up (see `determinize::redeal`).
pub const WORLD_ATTEMPTS: u32 = 4;

/// A determinized copy of `game` from its actor's view: up to `WORLD_ATTEMPTS` re-deals. On refusal, `mcts:N` /
/// `pmcts:N` skip the world (counted; they never search the real game, which shows the hidden cards); `greedy1`
/// (stage 1) still falls back to the real game (counted).
pub fn world_for(game: &Game, rng: &mut SplitMix64) -> Result<Game, (determinize::RedealRefusal, u32)> {
    let mut last = determinize::RedealRefusal::NoDecision;
    for _ in 0..WORLD_ATTEMPTS {
        match determinize::redeal(game, &mut SplitMix64::new(rng.next_u64())) {
            Ok(world) => return Ok(world),
            Err(reason) => last = reason,
        }
    }
    Err((last, WORLD_ATTEMPTS))
}

/// One-ply lookahead on one re-dealt world: each masked action, auto-advanced to the next real decision,
/// scored by GSE3 from the actor's seat; ties to the lower index.
pub struct Greedy1Bot {
    rng: SplitMix64,
    mulligan: MulliganPolicy,
    mull: MulliganStats,
    errors: u64,
    world_refusals: u64,
    world_fallbacks: u64,
}

impl Greedy1Bot {
    pub fn new(seed: u64, mulligan: MulliganPolicy) -> Self {
        Greedy1Bot {
            rng: SplitMix64::new(seed),
            mulligan,
            mull: MulliganStats::default(),
            errors: 0,
            world_refusals: 0,
            world_fallbacks: 0,
        }
    }
}

impl Bot for Greedy1Bot {
    fn choose(&mut self, game: &Game) -> usize {
        if let Some(k) = self.mull.apply(self.mulligan, game) {
            return k;
        }
        let me = game.actor().expect("a decision");
        let world = match world_for(game, &mut self.rng) {
            Ok(w) => w,
            Err((_, refused)) => {
                self.world_refusals += u64::from(refused);
                self.world_fallbacks += 1;
                game.clone_untracked()
            }
        };
        let mut best = (f64::NEG_INFINITY, 0usize);
        for k in 0..world.num_choices() {
            let mut g = world.clone_untracked();
            let v = match g.play(k) {
                Ok(()) if !g.is_halted() => eval::evaluate(&g, me),
                _ => {
                    self.errors += 1;
                    continue;
                }
            };
            if v > best.0 {
                best = (v, k);
            }
        }
        best.1
    }

    fn stats(&self) -> Value {
        json!({
            "errors": self.errors,
            "world_refusals": self.world_refusals,
            "world_fallbacks": self.world_fallbacks,
            "mulligan_decisions": self.mull.decisions,
            "mulligans_taken": self.mull.taken,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_parse() {
        assert_eq!(BotSpec::parse("random").unwrap(), BotSpec::Random { mulligan: MulliganPolicy::Keep });
        assert_eq!(BotSpec::parse("random,mull=random").unwrap(), BotSpec::Random { mulligan: MulliganPolicy::Own });
        assert_eq!(BotSpec::parse("greedy1,mull=xmage").unwrap(), BotSpec::Greedy1 { mulligan: MulliganPolicy::Xmage });
        let BotSpec::Mcts(c) = BotSpec::parse("mcts:1000,worlds=4").unwrap() else { panic!() };
        assert_eq!((c.budget, c.worlds), (1000, 4));
        let BotSpec::Mcts(c) = BotSpec::parse("heuristic@100").unwrap() else { panic!() };
        assert_eq!((c.budget, c.worlds, c.c_puct, c.discount), (100, 1, 1.0, 0.99));
        assert_eq!(c.mulligan, MulliganPolicy::Keep);
        let BotSpec::Mcts(c) = BotSpec::parse("mcts:10,mull=search").unwrap() else { panic!() };
        assert_eq!(c.mulligan, MulliganPolicy::Own);
        // an alias is the same bot (mirror detection compares parsed specs)
        assert_eq!(BotSpec::parse("mcts:3").unwrap(), BotSpec::parse(" heuristic@3 ").unwrap());
        assert_ne!(BotSpec::parse("mcts:3").unwrap(), BotSpec::parse("mcts:3,worlds=2").unwrap());
        assert!(BotSpec::parse("mcts:10,mull=maybe").is_err());
        assert!(BotSpec::parse("random,worlds=2").is_err());
        assert!(BotSpec::parse("mcts:0").is_err());
        assert!(BotSpec::parse("mcts:10,foo=1").is_err());
        assert!(BotSpec::parse("nope").is_err());
    }
}
