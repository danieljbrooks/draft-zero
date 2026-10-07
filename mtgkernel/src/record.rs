//! One game, played to the end, and its JSONL record (readable by draft-zero's
//! `tools/imitation_scale/gih.py` and `card_stats.py`, which read `winner`, `error`, `deck1`/`deck2` and
//! `seats.<A|B>.inHand`).
//!
//! Seen in hand, as 17lands' game data counts it: `seats.X.openingHand` is the player's last seven-card deal
//! (before the London mulligan's bottoming; 17lands' `opening_hand` is seven cards whatever the mulligan
//! count), `seats.X.drawn` the cards drawn after the mulligans, `seats.X.inHand` their sum by name (17lands'
//! GIH: in the opening hand or drawn; copies = opening + drawn). Also `seats.X.keptHand` (the hand after
//! bottoming) and `seats.X.tutored` (other library-to-hand moves, 17lands' `tutored`, not in `inHand`).
//! Names are the registry's, which are the FDN reference names (tests check this).
//!
//! `turns` counts individual turns, as XMage's `getTurnNum` (draft-zero's Java records); `rounds` is
//! mtg-kernel's own counter (both players' turns are one round).

use crate::bots::{Bot, BotSpec};
use crate::data::GameData;
use crate::deck::Deck;
use crate::game::{seat_letter, Game};
use crate::rng::bot_seed;
use mtg_kernel::ids::PlayerId;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

pub const METHOD: &str = "mtgkernel";

/// One game to play: deck1 sits in seat A (P0, plays first); bot1 sits in A unless `swap`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameTask {
    pub pair: usize,
    pub swap: bool,
    pub deck1: String,
    pub deck2: String,
    pub game_seed: u64,
    pub env_seed: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TerminalRec {
    pub outcome: Option<String>,
    pub classification: Option<String>,
    pub code: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DecisionsRec {
    /// Engine physical decisions (a combat declaration scanned creature by creature is one).
    pub physical: u64,
    /// Engine policy steps (every answer the session took, forced ones included).
    pub policy_steps: u64,
    #[serde(rename = "real_A")]
    pub real_a: u64,
    #[serde(rename = "real_B")]
    pub real_b: u64,
    /// Steps taken without a bot (one action left after masking).
    pub forced: u64,
    /// The most masked actions any decision offered (bot-consulted or not).
    pub max_choices: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SeatRec {
    /// openingHand + drawn (17lands' "games in hand").
    #[serde(rename = "inHand")]
    pub in_hand: BTreeMap<String, u32>,
    /// The last seven-card deal, bottomed cards included (17lands' `opening_hand`).
    #[serde(rename = "openingHand")]
    pub opening_hand: BTreeMap<String, u32>,
    /// The hand kept after bottoming (7 - mulligans cards).
    #[serde(rename = "keptHand")]
    pub kept_hand: BTreeMap<String, u32>,
    pub drawn: BTreeMap<String, u32>,
    pub tutored: BTreeMap<String, u32>,
}

/// The step that ended a halted game: whose it was and whether it was forced (one action after masking).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HaltStep {
    pub seat: String,
    pub role: String,
    pub forced: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GameRecord {
    pub pair: usize,
    pub swap: bool,
    pub deck1: String,
    pub deck2: String,
    /// The first 16 hex digits of each deck's mtg-kernel content identity (sha256 of its registry ids).
    pub deck1_hash: String,
    pub deck2_hash: String,
    #[serde(rename = "botA")]
    pub bot_a: String,
    #[serde(rename = "botB")]
    pub bot_b: String,
    pub bot1: String,
    pub bot2: String,
    /// The first 16 hex digits of bot1's / bot2's network sha256 (network bots only): resume and the data's
    /// game keys tell two networks under one path apart.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bot1_net: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bot2_net: Option<String>,
    pub winner: Option<String>,
    pub winner_role: Option<String>,
    pub winner_bot: Option<String>,
    /// The seat that played first (always A in this session).
    pub first: String,
    /// Individual turns (XMage's `getTurnNum`, as draft-zero's Java records count them).
    pub turns: u32,
    /// mtg-kernel's turn counter (rounds: both players' turns count once).
    pub rounds: u32,
    pub seconds: f64,
    pub game_seed: u64,
    pub env_seed: u64,
    pub method: String,
    pub error: Option<String>,
    /// Set for a halted game only.
    pub halt_step: Option<HaltStep>,
    pub terminal: TerminalRec,
    pub decisions: DecisionsRec,
    pub mulligans: BTreeMap<String, u8>,
    pub seats: BTreeMap<String, SeatRec>,
    /// Each seat's bot counters (search statistics for mcts); deterministic.
    pub search: BTreeMap<String, Value>,
}

fn snake(v: impl Serialize) -> Option<String> {
    serde_json::to_value(v).ok().and_then(|v| v.as_str().map(str::to_string))
}

fn count_names(defs: &[u16]) -> BTreeMap<String, u32> {
    let mut m = BTreeMap::new();
    for &d in defs {
        *m.entry(Game::card_name(d).to_string()).or_insert(0) += 1;
    }
    m
}

/// Per-game limits beyond the engine's 20,000-step caps. A game that reaches one ends at that decision as
/// truncated (no winner; `terminal.code` says which limit; its data is flagged truncated, z = 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GameLimits {
    /// Permanents on both battlefields together (0: none). Normal play peaks below 40 (4,320 `mcts:100` and
    /// `pmcts:100` self-play games); a runaway token board (Homunculus Horde with a looter doubles every turn)
    /// otherwise makes a game cost tens of minutes under search.
    pub max_permanents: usize,
    /// Wall-clock seconds (0: none). A safety net only: a game it ends is not reproducible.
    pub max_seconds: f64,
}

pub const DEFAULT_MAX_PERMANENTS: usize = 100;

impl Default for GameLimits {
    fn default() -> Self {
        GameLimits { max_permanents: DEFAULT_MAX_PERMANENTS, max_seconds: 0.0 }
    }
}

impl GameLimits {
    /// The limit `game` has reached (terminal code, reason), checked at each real decision.
    fn reached(&self, game: &Game, t0: &Instant) -> Option<(&'static str, String)> {
        if self.max_permanents > 0 {
            let n: usize = game.state().players.iter().map(|p| p.battlefield.len()).sum();
            if n >= self.max_permanents {
                return Some(("dzk_permanent_cap", format!("{n} permanents (limit {})", self.max_permanents)));
            }
        }
        if self.max_seconds > 0.0 && t0.elapsed().as_secs_f64() >= self.max_seconds {
            return Some((
                "dzk_game_seconds",
                format!("{:.0} s (limit {})", t0.elapsed().as_secs_f64(), self.max_seconds),
            ));
        }
        None
    }
}

/// Totals for the bench (and the summary) beyond the record.
#[derive(Debug, Clone, Default)]
pub struct GameTotals {
    pub policy_steps: u64,
    pub physical: u64,
    pub real: u64,
    pub simulations: u64,
}

/// Play one game to its end. Never panics: an engine error, a halted (fail-closed) game, or a panic becomes
/// the record's `error`.
pub fn play_game(
    task: &GameTask,
    deck1: &Deck,
    deck2: &Deck,
    bot1: (&str, &BotSpec),
    bot2: (&str, &BotSpec),
) -> (GameRecord, GameTotals) {
    let (rec, totals, _) = play_game_data(task, deck1, deck2, bot1, bot2, false, &GameLimits::default());
    (rec, totals)
}

/// `play_game` under `limits`, and with `collect` the game's training data (`data.rs`): every real decision of
/// either seat, encoded from its actor's seat before it is played, with the bot's policy target, results filled
/// in at the end. A panicking game returns no data.
pub fn play_game_data(
    task: &GameTask,
    deck1: &Deck,
    deck2: &Deck,
    bot1: (&str, &BotSpec),
    bot2: (&str, &BotSpec),
    collect: bool,
    limits: &GameLimits,
) -> (GameRecord, GameTotals, Option<GameData>) {
    let ((name_a, spec_a), (name_b, spec_b)) = if task.swap { (bot2, bot1) } else { (bot1, bot2) };
    let mut rec = GameRecord {
        pair: task.pair,
        swap: task.swap,
        deck1: task.deck1.clone(),
        deck2: task.deck2.clone(),
        deck1_hash: deck1.short_hash(),
        deck2_hash: deck2.short_hash(),
        bot_a: name_a.to_string(),
        bot_b: name_b.to_string(),
        bot1: bot1.0.to_string(),
        bot2: bot2.0.to_string(),
        bot1_net: bot1.1.net_hash(),
        bot2_net: bot2.1.net_hash(),
        winner: None,
        winner_role: None,
        winner_bot: None,
        first: "A".to_string(),
        turns: 0,
        rounds: 0,
        seconds: 0.0,
        game_seed: task.game_seed,
        env_seed: task.env_seed,
        method: METHOD.to_string(),
        error: None,
        halt_step: None,
        terminal: TerminalRec::default(),
        decisions: DecisionsRec::default(),
        mulligans: BTreeMap::new(),
        seats: BTreeMap::new(),
        search: BTreeMap::new(),
    };
    let mut totals = GameTotals::default();
    let t0 = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let mut bots: [Box<dyn Bot>; 2] =
            [spec_a.build(bot_seed(task.game_seed, 0)), spec_b.build(bot_seed(task.game_seed, 1))];
        let mut game = Game::new(deck1, deck2, task.env_seed, task.game_seed, true)?;
        let mut err: Option<String> = None;
        let mut cut = None;
        let mut data = collect.then(GameData::default);
        while !game.is_terminal() {
            if let Some(c) = limits.reached(&game, &t0) {
                cut = Some(c);
                break;
            }
            let actor = game.actor().expect("nonterminal game has an actor");
            let k = bots[actor.index()].choose(&game);
            if k >= game.num_choices() {
                err = Some(format!("bot chose {k} of {} options", game.num_choices()));
                break;
            }
            if let Some(d) = data.as_mut() {
                let target = bots[actor.index()].take_target();
                d.push(&game, k, target);
            }
            if let Err(e) = game.play(k) {
                err = Some(e);
                break;
            }
        }
        Ok::<_, String>((game, bots, err, data, cut))
    }));
    rec.seconds = (t0.elapsed().as_secs_f64() * 1000.0).round() / 1000.0;
    let (game, bots, err, mut data, cut) = match outcome {
        Ok(Ok(x)) => x,
        Ok(Err(e)) => {
            rec.error = Some(e);
            return (rec, totals, None);
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".into());
            rec.error = Some(format!("panic: {msg}"));
            return (rec, totals, None);
        }
    };
    rec.turns = game.turns();
    rec.rounds = game.rounds();
    rec.decisions = DecisionsRec {
        physical: game.physical_decisions(),
        policy_steps: game.policy_steps(),
        real_a: game.counters.real[0],
        real_b: game.counters.real[1],
        forced: game.counters.forced,
        max_choices: game.counters.max_choices,
    };
    let m = game.mulligans();
    rec.mulligans.insert("A".into(), m[0]);
    rec.mulligans.insert("B".into(), m[1]);
    if let Some(t) = game.tracker() {
        for p in [PlayerId::P0, PlayerId::P1] {
            let i = p.index();
            let kept = t.kept.as_ref().map(|o| o[i].clone()).unwrap_or_default();
            let all: Vec<u16> = t.deal[i].iter().chain(t.drawn[i].iter()).copied().collect();
            rec.seats.insert(
                seat_letter(p).to_string(),
                SeatRec {
                    in_hand: count_names(&all),
                    opening_hand: count_names(&t.deal[i]),
                    kept_hand: count_names(&kept),
                    drawn: count_names(&t.drawn[i]),
                    tutored: count_names(&t.tutored[i]),
                },
            );
        }
    }
    rec.search.insert("A".into(), bots[0].stats());
    rec.search.insert("B".into(), bots[1].stats());
    totals.policy_steps = rec.decisions.policy_steps;
    totals.physical = rec.decisions.physical;
    totals.real = rec.decisions.real_a + rec.decisions.real_b;
    totals.simulations =
        ["A", "B"].iter().filter_map(|s| rec.search[*s].get("simulations").and_then(Value::as_u64)).sum();
    if let Some(t) = game.terminal() {
        rec.terminal = TerminalRec {
            outcome: snake(t.terminal_outcome),
            classification: snake(t.terminal_classification),
            code: snake(t.terminal_code),
            reason: Some(t.terminal_reason.clone()),
        };
    }
    if let Some(e) = err {
        rec.error = Some(e);
        return finish_data(rec, totals, data.take());
    }
    if let Some((code, reason)) = cut {
        rec.terminal = TerminalRec {
            outcome: Some("truncated".into()),
            classification: Some("truncated".into()),
            code: Some(code.into()),
            reason: Some(reason),
        };
        return finish_data(rec, totals, data);
    }
    if game.is_halted() {
        if let Some(seat) = game.counters.last_actor {
            let seat = if seat == 0 { "A" } else { "B" };
            let bot1 = (seat == "A") != task.swap;
            rec.halt_step = Some(HaltStep {
                seat: seat.to_string(),
                role: if bot1 { "bot1" } else { "bot2" }.to_string(),
                forced: game.counters.last_forced,
            });
        }
        rec.error = Some(format!(
            "halted: {}: {}",
            rec.terminal.code.clone().unwrap_or_default(),
            rec.terminal.reason.clone().unwrap_or_default()
        ));
        return finish_data(rec, totals, data.take());
    }
    if let Some(w) = game.winner() {
        let seat = seat_letter(w);
        rec.winner = Some(seat.to_string());
        let bot1_won = (seat == "A") != task.swap;
        rec.winner_role = Some(if bot1_won { "bot1" } else { "bot2" }.to_string());
        rec.winner_bot = Some(if bot1_won { bot1.0 } else { bot2.0 }.to_string());
    }
    finish_data(rec, totals, data)
}

fn finish_data(
    rec: GameRecord,
    totals: GameTotals,
    data: Option<GameData>,
) -> (GameRecord, GameTotals, Option<GameData>) {
    let data = data.map(|mut d| {
        d.finish(&rec);
        d
    });
    (rec, totals, data)
}
