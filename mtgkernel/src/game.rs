//! One mtg-kernel game under schema-4 Limited rules, with draft-zero's decision loop on top.
//!
//! Construction mirrors `limited_session_v1::LimitedJsonlServerV1::new_with_london_mulligans_v1` resetting a
//! game: `RlEpisodeSessionV1::reset_with_custom_decks_and_london_v1(episode_id, env_seed, caps, deck_ids,
//! decks, PriorityModeV1::EngineWindowsV1, foundations_combat = true, london_mulligans = true)`; the session
//! enables Limited incremental trigger ordering itself when Foundations combat is on.
//!
//! Seats: seat A = P0 (deck1), seat B = P1 (deck2). P0 always plays first in this session (the reset builds
//! the state with `starting_player = P0`); the record says so (`first`).
//!
//! Decision loop. At each engine decision the legal list is masked: `activate_mana_ability` actions are
//! dropped whenever at least one other action remains. Casting a spell or activating a non-mana ability pays
//! its mana itself (engine.rs: the cast's payment stage calls `mana::can_pay`, mtg-kernel's exact
//! backtracking payment solver, and `pay_plan` taps the chosen sources), so manual mana activation is never
//! needed to cast. After masking, a decision with exactly one action left is stepped without consulting a
//! bot (a "forced" decision). The game therefore always rests at a real decision (2+ masked actions) or at
//! a terminal.
//!
//! Throughput (patch 0002): the session observes without the visible-projection hash (the observation is
//! otherwise the JSONL server's), steps with `step_quiet_v1` (no response copy), and `Game` reads the current
//! decision through the session's borrowing accessors instead of keeping its own copy.

use crate::deck::Deck;
use mtg_kernel::card_def::CARD_DEFS;
use mtg_kernel::event::CommittedEvent;
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::rl::{
    ActionSemanticV1, LegalActionV5, ObservationV5, PlayerSeatV1, TerminalClassificationV1, TerminalOutcomeV1,
};
use mtg_kernel::rl_session::{RlEpisodeSessionV1, RlSessionTerminalV1};
use mtg_kernel::state::{GameState, Zone};
use mtg_kernel::surface_v2::PriorityModeV1;

pub const MAX_PHYSICAL_DECISIONS: u64 = 20_000;
pub const MAX_POLICY_STEPS: u64 = 20_000;

pub fn player_of(seat: PlayerSeatV1) -> PlayerId {
    match seat {
        PlayerSeatV1::P0 => PlayerId::P0,
        PlayerSeatV1::P1 => PlayerId::P1,
    }
}

/// "A" for P0, "B" for P1.
pub fn seat_letter(p: PlayerId) -> &'static str {
    if p == PlayerId::P0 {
        "A"
    } else {
        "B"
    }
}

pub fn is_mana_ability(a: &LegalActionV5) -> bool {
    matches!(a.semantic, ActionSemanticV1::ActivateManaAbility { .. })
}

/// Decisions taken in this game (copied along with clones, so a search clone keeps counting).
#[derive(Debug, Clone, Default)]
pub struct Counters {
    /// Decisions a bot was consulted on, by seat (A, B).
    pub real: [u64; 2],
    /// Decisions stepped without a bot (one action after masking).
    pub forced: u64,
    /// The most masked actions any decision of this game offered.
    pub max_choices: u64,
    /// Who took the last engine step (P0 = 0, P1 = 1) and whether it was forced: names the step that ended
    /// a game (e.g. a fail-closed halt).
    pub last_actor: Option<u8>,
    pub last_forced: bool,
}

/// Seen-in-hand bookkeeping for the game record (the live game only; search clones drop it).
///
/// mtg-kernel's London mulligan runs in rounds: every announcement, then each mulliganing player's redraw
/// of seven, then that player's bottom choices (as many as its mulligan count), then the next round's
/// announcements on the smaller hand (XMage's `LondonMulligan` has the same order). The step that answers a
/// round's last announcement does the redraws and returns at the first bottom choice, so the hand right after
/// the step that raised a player's mulligan count is its new seven-card deal.
#[derive(Debug, Clone, Default)]
pub struct HandTracker {
    /// Each player's last seven-card deal (the reset's, or the last mulligan's redraw), before any card went
    /// to the bottom: 17lands' `opening_hand` (seven cards whatever the mulligan count).
    pub deal: [Vec<u16>; 2],
    deal_counts: Option<[u8; 2]>,
    /// The hands kept when the mulligan phase ended (the deal minus the bottomed cards).
    pub kept: Option<[Vec<u16>; 2]>,
    /// Cards drawn after that point (`Draw` events): 17lands' `drawn`.
    pub drawn: [Vec<u16>; 2],
    /// Other library-to-hand moves after that point (tutors): 17lands' `tutored`, not part of "in hand".
    pub tutored: [Vec<u16>; 2],
}

/// A library-to-hand move committed at or after a cursor.
enum HandMove {
    Draw(usize, ObjectId),
    Tutor(usize, ObjectId),
}

impl HandTracker {
    fn mulligans_done(state: &GameState) -> bool {
        state.london_mulligans_v1.as_ref().is_none_or(|m| m.is_complete())
    }

    /// Each library-to-hand move committed at or after `cursor`. A `Draw` event is one draw (a new
    /// incarnation of the object, no separate zone change), so nothing is counted twice.
    fn moves_since(state: &GameState, cursor: usize) -> Vec<HandMove> {
        let mut out = Vec::new();
        for ev in state.engine.event_history.iter().skip(cursor) {
            match ev {
                CommittedEvent::Draw { player, object: Some(id) } => out.push(HandMove::Draw(player.index(), *id)),
                CommittedEvent::ZoneChange { object, from: Zone::Library, to: Zone::Hand, .. } => {
                    out.push(HandMove::Tutor(state.objects.get(*object).owner.index(), *object))
                }
                _ => {}
            }
        }
        out
    }

    fn hand_defs(state: &GameState, p: usize) -> Vec<u16> {
        state.players[p].hand.iter().map(|&id| state.objects.get(id).card_def).collect()
    }

    fn record(&mut self, state: &GameState, moves: Vec<HandMove>) {
        for m in moves {
            match m {
                HandMove::Draw(p, id) => self.drawn[p].push(state.objects.get(id).card_def),
                HandMove::Tutor(p, id) => self.tutored[p].push(state.objects.get(id).card_def),
            }
        }
    }

    fn observe(&mut self, state: &GameState, cursor: usize) {
        if self.kept.is_some() {
            let moves = Self::moves_since(state, cursor);
            self.record(state, moves);
            return;
        }
        let counts = state.london_mulligans_v1.as_ref().map_or([0, 0], |m| m.counts());
        for p in 0..2 {
            if self.deal_counts.is_none_or(|c| c[p] != counts[p]) {
                self.deal[p] = Self::hand_defs(state, p);
            }
        }
        self.deal_counts = Some(counts);
        if !Self::mulligans_done(state) {
            return;
        }
        // The step that ended the mulligans was a keep (or the last bottom); no redraw happens in it, so every
        // library-to-hand move committed during it came after the kept hand.
        let moves = Self::moves_since(state, cursor);
        let mut kept = [Vec::new(), Vec::new()];
        for (p, hand) in kept.iter_mut().enumerate() {
            for &id in &state.players[p].hand {
                let moved = moves
                    .iter()
                    .any(|m| matches!(*m, HandMove::Draw(q, o) | HandMove::Tutor(q, o) if q == p && o == id));
                if !moved {
                    hand.push(state.objects.get(id).card_def);
                }
            }
        }
        self.kept = Some(kept);
        self.record(state, moves);
    }
}

/// Where the current decision sits: what `observe_policy_v5` needs to rebuild its observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionCoords {
    pub step: u64,
    pub physical_decision_id: u64,
    pub substep_index: u32,
    pub substep_count: u32,
}

#[derive(Clone)]
pub struct Game {
    session: RlEpisodeSessionV1,
    episode_id: u64,
    /// Original indices (into the current legal list) of the masked actions; empty at a terminal.
    masked: Vec<u32>,
    pub counters: Counters,
    tracker: Option<Box<HandTracker>>,
}

impl Game {
    /// A fresh game, auto-advanced to its first real decision. `track` keeps the seen-in-hand record.
    pub fn new(deck_a: &Deck, deck_b: &Deck, env_seed: u64, episode_id: u64, track: bool) -> Result<Game, String> {
        let mut session = RlEpisodeSessionV1::reset_with_custom_decks_and_london_v1(
            episode_id,
            env_seed,
            MAX_PHYSICAL_DECISIONS,
            MAX_POLICY_STEPS,
            [deck_a.content_id.clone(), deck_b.content_id.clone()],
            [&deck_a.ids, &deck_b.ids],
            PriorityModeV1::EngineWindowsV1,
            true,
            true,
        )
        .map_err(|e| format!("reset: {e}"))?;
        // The reset's own first observation carries the hash; every later one is built without it.
        session.set_skip_visible_projection_hash_v1(true);
        let mut game = Game {
            session,
            episode_id,
            masked: Vec::new(),
            counters: Counters::default(),
            tracker: track.then(Box::<HandTracker>::default),
        };
        game.refresh();
        if let Some(t) = game.tracker.as_mut() {
            // Before any step: the reset's own deal is not a draw after the opening hand.
            let cursor = game.session.game_state().engine.event_history.len();
            t.observe(game.session.game_state(), cursor);
        }
        game.auto_advance()?;
        Ok(game)
    }

    /// A copy without the seen-in-hand tracker, for search.
    pub fn clone_untracked(&self) -> Game {
        Game {
            session: self.session.clone(),
            episode_id: self.episode_id,
            masked: self.masked.clone(),
            counters: self.counters.clone(),
            tracker: None,
        }
    }

    /// Re-read the session's current decision: the masked list (empty at a terminal).
    fn refresh(&mut self) {
        self.masked.clear();
        let n = self.session.current_legal_action_count_v1();
        for i in 0..n {
            if !is_mana_ability(self.legal_action(i)) {
                self.masked.push(i as u32);
            }
        }
        if self.masked.is_empty() {
            self.masked.extend(0..n as u32);
        }
        self.counters.max_choices = self.counters.max_choices.max(self.masked.len() as u64);
    }

    /// One engine step on the action at `original` (an index into the unmasked legal list).
    fn raw_step(&mut self, original: u32) -> Result<(), String> {
        if self.is_terminal() {
            return Err("step on a terminal game".into());
        }
        let a = self.session.current_legal_action_v1(original as usize).ok_or("action index out of range")?;
        let (selected, id) = (a.selected_index, a.stable_id.clone());
        self.counters.last_actor = self.actor().map(|p| p.index() as u8);
        let cursor = self.session.game_state().engine.event_history.len();
        let expected = self.session.policy_step_count();
        self.session.step_quiet_v1(self.episode_id, expected, selected, &id).map_err(|e| format!("step: {e}"))?;
        self.refresh();
        if let Some(t) = self.tracker.as_mut() {
            t.observe(self.session.game_state(), cursor);
        }
        Ok(())
    }

    /// Step every forced decision until a real decision or a terminal.
    pub fn auto_advance(&mut self) -> Result<(), String> {
        while self.masked.len() == 1 {
            let only = self.masked[0];
            self.raw_step(only)?;
            self.counters.forced += 1;
            self.counters.last_forced = true;
        }
        Ok(())
    }

    /// Play the `k`-th masked action of a real decision (a bot's choice), then auto-advance.
    pub fn play(&mut self, k: usize) -> Result<(), String> {
        let actor = self.actor().ok_or("play on a terminal game")?;
        let original = *self.masked.get(k).ok_or("choice out of range")?;
        self.raw_step(original)?;
        self.counters.real[actor.index()] += 1;
        self.counters.last_forced = false;
        self.auto_advance()
    }

    // ---------------------------------------------------------------- what bots read

    /// The current decision's observation (the acting player's view; `visible_projection_hash` is 0 after the
    /// first decision, see the module docs).
    pub fn observation(&self) -> Option<&ObservationV5> {
        self.session.current_observation_v1()
    }

    pub fn decision_coords(&self) -> Option<DecisionCoords> {
        self.session.current_substep_v1().map(|(physical_decision_id, substep_index, substep_count)| DecisionCoords {
            step: self.session.policy_step_count(),
            physical_decision_id,
            substep_index,
            substep_count,
        })
    }

    /// The full (unmasked) legal list's length at the current decision (0 at a terminal).
    pub fn num_legal(&self) -> usize {
        self.session.current_legal_action_count_v1()
    }

    /// The `i`-th action of the full (unmasked) legal list.
    pub fn legal_action(&self, i: usize) -> &LegalActionV5 {
        self.session.current_legal_action_v1(i).expect("legal action index in range")
    }

    /// Original indices of the masked actions (what `choose` indexes).
    pub fn masked(&self) -> &[u32] {
        &self.masked
    }

    pub fn num_choices(&self) -> usize {
        self.masked.len()
    }

    /// The `k`-th masked action.
    pub fn choice(&self, k: usize) -> &LegalActionV5 {
        self.legal_action(self.masked[k] as usize)
    }

    pub fn actor(&self) -> Option<PlayerId> {
        self.session.current_actor_v1()
    }

    pub fn is_terminal(&self) -> bool {
        self.session.terminal_v1().is_some()
    }

    pub fn terminal(&self) -> Option<&RlSessionTerminalV1> {
        self.session.terminal_v1()
    }

    /// The winner of a natural terminal.
    pub fn winner(&self) -> Option<PlayerId> {
        self.terminal().and_then(|t| t.winner).map(player_of)
    }

    pub fn is_halted(&self) -> bool {
        self.terminal().is_some_and(|t| t.terminal_classification == TerminalClassificationV1::Halted)
    }

    pub fn is_truncated(&self) -> bool {
        self.terminal().is_some_and(|t| t.terminal_classification == TerminalClassificationV1::Truncated)
    }

    pub fn is_draw(&self) -> bool {
        self.terminal().is_some_and(|t| t.terminal_outcome == TerminalOutcomeV1::Draw)
    }

    pub fn state(&self) -> &GameState {
        self.session.game_state()
    }

    pub fn session(&self) -> &RlEpisodeSessionV1 {
        &self.session
    }

    /// Mutable session access, for determinization of a search copy only.
    pub(crate) fn session_mut(&mut self) -> &mut RlEpisodeSessionV1 {
        &mut self.session
    }

    pub fn policy_steps(&self) -> u64 {
        self.session.policy_step_count()
    }

    pub fn physical_decisions(&self) -> u64 {
        self.session.physical_decision_count()
    }

    pub fn tracker(&self) -> Option<&HandTracker> {
        self.tracker.as_deref()
    }

    /// Mulligans taken by P0 and P1.
    pub fn mulligans(&self) -> [u8; 2] {
        self.state().london_mulligans_v1.as_ref().map_or([0, 0], |m| m.counts())
    }

    /// The engine's turn counter, which counts rounds (both players' turns are one).
    pub fn rounds(&self) -> u32 {
        self.state().turn
    }

    /// Individual turns taken (XMage's `getTurnNum`): two a round, one in the current round's first half.
    pub fn turns(&self) -> u32 {
        let s = self.state();
        let first_half = s.active_player == s.starting_player;
        s.turn.saturating_sub(1) * 2 + if first_half { 1 } else { 2 }
    }

    pub fn card_name(def: u16) -> &'static str {
        CARD_DEFS[def as usize].name
    }
}
