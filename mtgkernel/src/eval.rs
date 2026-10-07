//! XMage's GameStateEvaluator3 (Mage.Player.AI `score/GameStateEvaluator3.java`), on mtg-kernel state.
//!
//! Terminal: +1 a win, -1 a loss, 0 a draw (or a truncated game inside a search; the search drops halted
//! leaves before scoring). Otherwise
//! `tanh((S_me - S_opp) / 15)` with `S = 0.6 max(0, life) + H(cards in hand) + sum over the permanents the
//! player controls of (1 + mana value)`, H the harmonic number. Mana value is the card definition's
//! `mana_value`; a token's definition carries its own (a copy token keeps the copied card's, e.g. Homunculus
//! Horde's token is 4, as in XMage). The Java constant UNTAPPED_BONUS is 0, so tapped state does not matter.

use crate::game::Game;
use mtg_kernel::card_def::CARD_DEFS;
use mtg_kernel::ids::PlayerId;
use mtg_kernel::state::GameState;

const LIFE_TOTAL_VALUE: f64 = 0.6;
const HAND_CARD_VALUE: f64 = 1.0;
const PERM_BASE_VALUE: f64 = 1.0;
const NORMALIZATION_FACTOR: f64 = 15.0;

pub fn harmonic(n: usize) -> f64 {
    (1..=n).map(|i| 1.0 / i as f64).sum()
}

pub fn resources(state: &GameState, p: PlayerId) -> f64 {
    let ps = &state.players[p.index()];
    let mut score = LIFE_TOTAL_VALUE * f64::from(ps.life.max(0));
    score += HAND_CARD_VALUE * harmonic(ps.hand.len());
    // Battlefield lists are per player; count by controller so a stolen permanent scores for its controller.
    for side in &state.players {
        for &id in &side.battlefield {
            let o = state.objects.get(id);
            if o.controller == p {
                score += PERM_BASE_VALUE + f64::from(CARD_DEFS[o.card_def as usize].mana_value);
            }
        }
    }
    score
}

/// GSE3 from `me`'s seat, for a nonterminal state.
pub fn evaluate_state(state: &GameState, me: PlayerId) -> f64 {
    ((resources(state, me) - resources(state, me.opponent())) / NORMALIZATION_FACTOR).tanh()
}

/// GSE3 of a game from `me`'s seat, terminals included.
pub fn evaluate(game: &Game, me: PlayerId) -> f64 {
    if game.is_terminal() {
        return match game.winner() {
            Some(w) if w == me => 1.0,
            Some(_) => -1.0,
            None => 0.0,
        };
    }
    evaluate_state(game.state(), me)
}

#[cfg(test)]
mod tests {
    #[test]
    fn harmonic_numbers() {
        assert_eq!(super::harmonic(0), 0.0);
        assert!((super::harmonic(3) - (1.0 + 0.5 + 1.0 / 3.0)).abs() < 1e-12);
    }
}
