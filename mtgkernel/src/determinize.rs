//! PIMC worlds: re-deal the hidden cards of a game copy from one player's point of view.
//!
//! A re-implementation of mtg-kernel's `kernel_native_search_opponent_v1::redeterminize_hidden_zones_v1`
//! (`pub(crate)` there) through the visibility patch's `game_state_mut_v1`. For each owner, the unknown
//! slots are the owner's hand cards the viewer does not know (only for the viewer's opponent) and every
//! library position the viewer does not know; the card definitions in those slots are sorted, permuted
//! (Fisher-Yates, unbiased draws from our SplitMix64) and written back, with each object's name and
//! per-definition `ObjectStateV4` reset exactly as the kernel does.
//!
//! This is an OPEN-DECKLIST assumption: the searcher knows the opponent's 40 cards, though not their order
//! or which are in hand. A closed-decklist belief sampler is later work.
//!
//! The future RNG. A world's engine randomness (future library shuffles: in this card pool, only the London
//! mulligan's) is re-seeded from the searcher's own generator through mtg-kernel's search-only
//! `GameState::resample_future_randomness_for_search_v3` (made public by patch 0001), so a search no longer
//! sees the real future shuffle stream (which, at a mulligan, told it exactly which cards come back).
//!
//! Objects an action names. An object in an unknown slot that one of the root's legal actions refers to (a
//! `CardStableRefV1`: arena id + card definition) is known to the actor through that action, so it keeps its
//! definition (it is left out of the slots). Every action's semantics and stable id are then the same in the
//! world, by construction: the re-deal changes nothing but the other slots' definitions.
//!
//! Admission check (approximate; the kernel's FastActor path regenerates the candidate list and compares
//! semantics, which this session cannot do because it keeps no regenerable decision): the actor's
//! observation, rebuilt on the re-dealt state, must equal the root decision's. A root action that is legal
//! only because of a hidden card it does not name would go unnoticed (none is known in this pool). A refused
//! world is retried by the caller with a new seed (`bots::world_for`); after `WORLD_ATTEMPTS` refusals `mcts`/`pmcts`
//! skip that world and `greedy1` uses the real game (both counted).

use crate::game::Game;
use crate::rng::SplitMix64;
use mtg_kernel::card_def::CARD_DEFS;
use mtg_kernel::ids::{ObjectId, PlayerId};
use mtg_kernel::rl::observe_policy_v5_without_projection_hash_v1;
use mtg_kernel::state::{GameState, ObjectStateV4, Zone};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedealRefusal {
    NoDecision,
    HiddenStateContract(&'static str),
    ObservationChanged,
}

/// The unknown slots of each owner, from `viewer`'s point of view, in the kernel's order (hand, then library).
pub fn unknown_slots(state: &GameState, viewer: PlayerId) -> [Vec<ObjectId>; 2] {
    let mut out = [Vec::new(), Vec::new()];
    for owner in [PlayerId::P0, PlayerId::P1] {
        let slots = &mut out[owner.index()];
        if owner != viewer {
            let known = state.known_hand_cards(viewer, owner);
            for &id in &state.players[owner.index()].hand {
                let zcc = state.objects.get(id).zone_change_count;
                if !known.iter().any(|e| e.object == id && e.zone_change_count == zcc) {
                    slots.push(id);
                }
            }
        }
        let known = state.known_library_cards(viewer, owner);
        for (position, &id) in state.players[owner.index()].library.iter().enumerate() {
            let zcc = state.objects.get(id).zone_change_count;
            if !known.iter().any(|e| {
                usize::try_from(e.position).ok() == Some(position) && e.object == id && e.zone_change_count == zcc
            }) {
                slots.push(id);
            }
        }
    }
    out
}

/// Re-deal `state`'s hidden cards from `viewer`'s point of view (every unknown slot; no admission check).
pub fn redeal_state(state: &mut GameState, viewer: PlayerId, rng: &mut SplitMix64) -> Result<(), RedealRefusal> {
    let slots = unknown_slots(state, viewer);
    redeal_slots(state, slots, rng)
}

/// Permute the card definitions over each owner's `slots`.
fn redeal_slots(state: &mut GameState, slots: [Vec<ObjectId>; 2], rng: &mut SplitMix64) -> Result<(), RedealRefusal> {
    for owner_slots in slots {
        let mut defs: Vec<u16> = owner_slots.iter().map(|&id| state.objects.get(id).card_def).collect();
        defs.sort_unstable();
        for i in (1..defs.len()).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            defs.swap(i, j);
        }
        for (id, def) in owner_slots.into_iter().zip(defs) {
            let o = state.objects.get(id);
            if !matches!(o.zone, Zone::Hand | Zone::Library)
                || o.spell_copy_origin.is_some()
                || !o.attachments.is_empty()
            {
                return Err(RedealRefusal::HiddenStateContract("unknown slot is not a plain hand/library card"));
            }
            // Written even when the definition is unchanged, as the kernel does (it also resets `v4`).
            let o = state.objects.get_mut(id);
            o.card_def = def;
            o.name = CARD_DEFS[def as usize].name.to_string();
            o.v4 = ObjectStateV4::from_card_def(def);
        }
    }
    Ok(())
}

/// Every object a JSON value refers to through a `CardStableRefV1` (an object with `arena_id` and `card_db_id`).
fn referenced_objects(v: &serde_json::Value, out: &mut BTreeSet<u32>) {
    match v {
        serde_json::Value::Object(m) => {
            if let (Some(id), true) = (m.get("arena_id").and_then(|x| x.as_u64()), m.contains_key("card_db_id")) {
                out.insert(id as u32);
            }
            for x in m.values() {
                referenced_objects(x, out);
            }
        }
        serde_json::Value::Array(a) => {
            for x in a {
                referenced_objects(x, out);
            }
        }
        _ => {}
    }
}

/// A determinized copy of `root` (at a decision) from its acting player's point of view: hidden cards re-dealt
/// (objects a legal action names kept), the future RNG re-seeded, admitted only if the actor's observation
/// is unchanged.
pub fn redeal(root: &Game, rng: &mut SplitMix64) -> Result<Game, RedealRefusal> {
    let viewer = root.actor().ok_or(RedealRefusal::NoDecision)?;
    let observation = root.observation().ok_or(RedealRefusal::NoDecision)?;
    let at = root.decision_coords().ok_or(RedealRefusal::NoDecision)?;
    // An object a legal action names is known to the actor: it keeps its definition.
    let mut refs = BTreeSet::new();
    for i in 0..root.num_legal() {
        referenced_objects(
            &serde_json::to_value(&root.legal_action(i).semantic).expect("semantics serialize"),
            &mut refs,
        );
    }
    let mut slots = unknown_slots(root.state(), viewer);
    for owner_slots in &mut slots {
        owner_slots.retain(|id| !refs.contains(&id.0));
    }
    let mut world = root.clone_untracked();
    let state = world.session_mut().game_state_mut_v1();
    redeal_slots(state, slots, rng)?;
    state.resample_future_randomness_for_search_v3(rng.next_u64());
    // Rebuilt without the projection hash (a pure function of the other fields), so the stored one is copied
    // over before the comparison.
    let mut again = observe_policy_v5_without_projection_hash_v1(
        world.state(),
        world.session().policy_surface_v1(),
        viewer,
        at.step,
        at.physical_decision_id,
        at.substep_index,
        at.substep_count,
    )
    .map_err(|_| RedealRefusal::ObservationChanged)?;
    again.visible_projection_hash = observation.visible_projection_hash;
    if again != *observation {
        return Err(RedealRefusal::ObservationChanged);
    }
    Ok(world)
}

/// Per owner, the sorted multiset of card definitions over hand + library (a re-deal must preserve it).
pub fn hidden_zone_multiset(state: &GameState) -> [Vec<u16>; 2] {
    state.players.each_ref().map(|ps| {
        let mut defs: Vec<u16> =
            ps.hand.iter().chain(ps.library.iter()).map(|&id| state.objects.get(id).card_def).collect();
        defs.sort_unstable();
        defs
    })
}
