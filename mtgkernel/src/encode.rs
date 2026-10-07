//! `dzk-enc-v2`: a fixed-schema numeric encoding of one real decision, from the acting player's seat
//! (`docs/encoding.md` is the reference; keep the two in step and bump `ENC_VERSION` on any change).
//!
//! Inputs: the acting player's redacted observation (`ObservationV5`: own hand, public zones, counts, the
//! actor's known cards) and the decision's masked legal actions, plus static card-registry facts
//! (`CARD_DEFS`: mana value, types, colours, base P/T, printed keywords). `encode` takes nothing else: it
//! never sees the `GameState`, so it cannot read the opponent's hand or the library order. `arena_id` is
//! never a feature (it leaks how deep in the library a card started); it only links an action to the
//! object tokens of the same decision.
//!
//! Everything is actor-relative ("me" = the acting player). Three parts:
//! - `global` f32[G];
//! - one token per visible object (my hand, both battlefields, both graveyards, exile, stack items, the
//!   actor's known cards in the opponent's hand and in either library): `card` (vocabulary id), `zone`
//!   (category) and f32[F];
//! - one token per masked legal action, in the masked order: `kind`, `src`/`tgt` (object token index or
//!   -1), `src_card`/`tgt_card` (vocabulary ids), `tgt_player` (0 none, 1 me, 2 opponent) and f32[A].

use mtg_kernel::card_def::{CardDef, CardType, Keywords, CARD_DEFS};
use mtg_kernel::engine::{CastMode, CostKind, OptionalCostChoice};
use mtg_kernel::mana::ManaColor;
use mtg_kernel::policy_surface_v5::PolicySurfaceStageV5;
use mtg_kernel::rl::{
    ActionSemanticV1, CardCharacteristicsV2, CardPublicV2, CardStableRefV1, LegalActionV5, ObjectRelationPublicV4,
    ObservationV5, PendingEffectChoiceSemanticV4, PlayerSeatV1, StackItemKindV2, TargetRefV1, ZoneIndependentStepV1,
};
use mtg_kernel::state::Zone;

pub const ENC_VERSION: &str = "dzk-enc-v2";

/// Card vocabulary: id 0 = no card; registry id `d` is `d + 1`; ids past `VOCAB - 2` share `VOCAB - 1`.
pub const VOCAB: usize = 512;
/// Zone categories (one-hot width; 12 used).
pub const NZ: usize = 16;
/// Action kinds (one-hot width; 35 used, 39 = other).
pub const NK: usize = 40;
/// `tgt_player` one-hot width (none, me, opponent).
pub const NP: usize = 3;
pub const G: usize = 96;
pub const F: usize = 72;
pub const A: usize = 32;
/// At most this many object tokens; past it, tokens are dropped by per-group quotas (`cap_quotas`), duplicates
/// first, and counted. Reached only by runaway token boards (Homunculus Horde with a looter).
pub const MAX_OBJECTS: usize = 255;
/// Token groups the cap gives quotas to: my hand, the stack, my battlefield, the opponent's battlefield, the rest.
pub const CAP_GROUPS: usize = 5;
/// Every feature is clamped to [-FEATURE_CLAMP, FEATURE_CLAMP].
pub const FEATURE_CLAMP: f32 = 8.0;

// ---- zone categories
pub const Z_MY_HAND: u8 = 0;
pub const Z_MY_BF: u8 = 1;
pub const Z_OPP_BF: u8 = 2;
pub const Z_MY_GY: u8 = 3;
pub const Z_OPP_GY: u8 = 4;
pub const Z_MY_EXILE: u8 = 5;
pub const Z_OPP_EXILE: u8 = 6;
pub const Z_STACK_MINE: u8 = 7;
pub const Z_STACK_OPP: u8 = 8;
pub const Z_OPP_HAND_KNOWN: u8 = 9;
pub const Z_MY_LIB_KNOWN: u8 = 10;
pub const Z_OPP_LIB_KNOWN: u8 = 11;

/// The pooling group of a zone category: 0 my hand, 1 my battlefield, 2 opponent's battlefield, 3 the rest.
pub fn zone_group(zone: u8) -> usize {
    match zone {
        Z_MY_HAND => 0,
        Z_MY_BF => 1,
        Z_OPP_BF => 2,
        _ => 3,
    }
}

// ---- decision families (global one-hot at G_FAMILY)
pub const FAMILIES: [&str; 10] =
    ["priority", "target", "attack", "block", "damage", "mulligan", "bottom", "trigger_order", "legend", "other"];

// ---- global feature offsets
const G_LIFE: usize = 0; // me, opp (/20)
const G_HAND: usize = 2; // (/10)
const G_LIBRARY: usize = 4; // (/40)
const G_GRAVE: usize = 6; // (/20)
const G_EXILE: usize = 8; // owned cards in exile (/10)
const G_TURN: usize = 10; // engine rounds (/30)
const G_ACTIVE: usize = 11;
const G_PRIORITY: usize = 12;
const G_PHASE: usize = 13; // 12 steps
const G_POOL_ME: usize = 25; // W U B R G C (/5)
const G_POOL_OPP: usize = 31;
const G_LANDS: usize = 37; // me, opp (/10)
const G_UNTAPPED_LANDS: usize = 39;
const G_CREATURES: usize = 41;
const G_UNTAPPED_CREATURES: usize = 43;
const G_POWER: usize = 45; // total effective power of creatures (/20)
const G_LANDS_PLAYED: usize = 47;
const G_SPELLS_CAST: usize = 48; // me, opp (/5)
const G_DRAWS: usize = 50; // me (/5)
const G_MULL_ACTIVE: usize = 51;
const G_MULL_COUNT: usize = 52; // me, opp (/7)
const G_MULL_KEPT: usize = 54; // me, opp
const G_FAMILY: usize = 56; // 10
const G_STACK: usize = 66; // (/5)
const G_ATTACKERS_DECLARED: usize = 67;
const G_BLOCKERS_DECLARED: usize = 68;
const G_N_ATTACKERS: usize = 69; // (/5)
const G_STAGE_ATTACK: usize = 70;
const G_STAGE_BLOCK: usize = 71;
const G_SCAN_PROGRESS: usize = 72; // candidate_index / candidate_count
const G_SCAN_COUNT: usize = 73; // (/10)
const G_SCAN_SELECTED: usize = 74; // (/5)
const G_N_ACTIONS: usize = 75; // ln(1 + n) / 5
const G_ON_THE_PLAY: usize = 76;
const G_PENDING_CAST: usize = 77;
const G_PENDING_EFFECT: usize = 78;
const G_PENDING_MIN: usize = 79; // (/5)
const G_PENDING_MAX: usize = 80;
const G_PENDING_SELECTED: usize = 81;
const G_DAMAGE_ASSIGNMENT: usize = 82;
const G_OPP_KNOWN_HAND: usize = 83; // (/5)
const G_STACK_TOP_MINE: usize = 84;
const G_STACK_TOP_OPP: usize = 85;
const G_PASSED: usize = 86; // me, opp
const G_PENDING_TARGETS_PLAYER: usize = 88; // me, opp (a player chosen as a target of the pending cast/effect)
const G_HAS_LOST: usize = 90; // me, opp
                              // 92..96 reserved (zero)

// ---- object feature offsets
const O_TAPPED: usize = 0;
const O_SICK: usize = 1;
const O_DAMAGE: usize = 2;
const O_POWER: usize = 3;
const O_TOUGHNESS: usize = 4;
const O_BASE_POWER: usize = 5;
const O_BASE_TOUGHNESS: usize = 6;
const O_PLUS1: usize = 7;
const O_MINUS1: usize = 8;
const O_OTHER_COUNTERS: usize = 9;
const O_ATTACKING: usize = 10;
const O_BLOCKING: usize = 11;
const O_BLOCKED: usize = 12;
const O_ATTACHED: usize = 13;
const O_ATTACHED_TO_MINE: usize = 14;
const O_N_ATTACHMENTS: usize = 15;
const O_TOKEN: usize = 16;
const O_MANA_VALUE: usize = 17;
const O_TYPES: usize = 18; // creature land instant sorcery enchantment artifact planeswalker
const O_COLORS: usize = 25; // W U B R G, colourless
const O_KEYWORDS: usize = 31; // 14 flags
const O_WARD: usize = 45;
const O_MIN_BLOCKERS: usize = 46;
const O_LANDWALK: usize = 47;
const O_FLASH: usize = 48;
const O_UNBLOCKABLE: usize = 49;
const O_ENTERED_THIS_TURN: usize = 50;
const O_SCAN_SELECTED: usize = 51;
const O_SCAN_CURRENT: usize = 52;
const O_SCAN_REMAINING: usize = 53;
const O_SCAN_ATTACKER: usize = 54;
const O_PENDING_TARGET: usize = 55;
const O_TARGETED_BY_MINE: usize = 56;
const O_TARGETED_BY_OPP: usize = 57;
const O_STACK_SPELL: usize = 58;
const O_STACK_ACTIVATED: usize = 59;
const O_STACK_TRIGGERED: usize = 60;
const O_STACK_DEPTH: usize = 61;
const O_STACK_N_TARGETS: usize = 62;
const O_STACK_KICKED: usize = 63;
const O_STACK_COPY: usize = 64;
const O_STACK_TARGETS_ME: usize = 65;
const O_STACK_TARGETS_OPP: usize = 66;
const O_LIBRARY_POSITION: usize = 67;
const O_ABILITY_USED: usize = 68;
const O_SKIP_UNTAP: usize = 69;
const O_MANA_SOURCE: usize = 70;
const O_HAS_ACTIVATED: usize = 71;

// ---- action feature offsets
const X_MULLIGAN: usize = 0;
const X_MULLIGAN_COUNT: usize = 1;
const X_INCLUDE: usize = 2;
const X_DMG_MIN: usize = 3;
const X_DMG_MAX: usize = 4;
const X_DMG_SPLIT: usize = 5;
const X_DMG_UPPER: usize = 6;
const X_BOOL: usize = 7;
const X_MODE: usize = 8;
const X_MODE_COUNT: usize = 9;
const X_ABILITY: usize = 10;
const X_PREFIX: usize = 11;
const X_PENDING_COUNT: usize = 12;
const X_TRIGGER_INDEX: usize = 13;
const X_REMAINING: usize = 14;
const X_SELECTED: usize = 15;
const X_MIN_TARGETS: usize = 16;
const X_MAX_TARGETS: usize = 17;
const X_NUMBER: usize = 18;
const X_NUMBER_MIN: usize = 19;
const X_NUMBER_MAX: usize = 20;
const X_OPTION: usize = 21;
const X_OPTION_COUNT: usize = 22;
const X_N_OBJECTS: usize = 23;
const X_COLOR: usize = 24; // W U B R G C
const X_ALT_COST: usize = 30;
const X_COST_KIND: usize = 31;

/// The vocabulary id of a registry card id.
pub fn vocab_id(card_db_id: u16) -> u16 {
    (card_db_id as usize + 1).min(VOCAB - 1) as u16
}

/// One encoded decision (buffers reused across calls).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Encoded {
    pub global: Vec<f32>,
    pub obj_card: Vec<u16>,
    pub obj_zone: Vec<u8>,
    pub obj_feat: Vec<f32>,
    pub act_kind: Vec<u8>,
    pub act_src: Vec<i16>,
    pub act_src_card: Vec<u16>,
    pub act_tgt: Vec<i16>,
    pub act_tgt_card: Vec<u16>,
    pub act_tgt_player: Vec<u8>,
    pub act_feat: Vec<f32>,
    /// Object tokens dropped past `MAX_OBJECTS`.
    pub dropped_objects: u32,
    /// Action object references with no object token (the action keeps the card id, `src`/`tgt` = -1).
    pub unlinked_refs: u32,
}

impl Encoded {
    pub fn n_obj(&self) -> usize {
        self.obj_card.len()
    }

    pub fn n_act(&self) -> usize {
        self.act_kind.len()
    }

    pub fn clear(&mut self) {
        self.global.clear();
        self.obj_card.clear();
        self.obj_zone.clear();
        self.obj_feat.clear();
        self.act_kind.clear();
        self.act_src.clear();
        self.act_src_card.clear();
        self.act_tgt.clear();
        self.act_tgt_card.clear();
        self.act_tgt_player.clear();
        self.act_feat.clear();
        self.dropped_objects = 0;
        self.unlinked_refs = 0;
    }

    /// A new object token (uncapped: `cap_objects` enforces `MAX_OBJECTS` once every token is built).
    fn push_object(&mut self, card_db_id: u16, zone: u8) -> usize {
        self.obj_card.push(vocab_id(card_db_id));
        self.obj_zone.push(zone);
        self.obj_feat.extend_from_slice(&[0.0; F]);
        self.n_obj() - 1
    }

    fn of(&mut self, i: usize) -> &mut [f32] {
        &mut self.obj_feat[i * F..(i + 1) * F]
    }
}

fn def(card_db_id: u16) -> Option<&'static CardDef> {
    CARD_DEFS.get(card_db_id as usize)
}

fn color_index(c: ManaColor) -> usize {
    match c {
        ManaColor::W => 0,
        ManaColor::U => 1,
        ManaColor::B => 2,
        ManaColor::R => 3,
        ManaColor::G => 4,
        ManaColor::C => 5,
    }
}

fn step_index(s: ZoneIndependentStepV1) -> usize {
    use ZoneIndependentStepV1::*;
    match s {
        Untap => 0,
        Upkeep => 1,
        Draw => 2,
        Main1 => 3,
        BeginCombat => 4,
        DeclareAttackers => 5,
        DeclareBlockers => 6,
        CombatDamage => 7,
        EndCombat => 8,
        Main2 => 9,
        End => 10,
        Cleanup => 11,
    }
}

/// The action kind id (one-hot index) of a semantic.
pub fn action_kind(s: &ActionSemanticV1) -> u8 {
    use ActionSemanticV1::*;
    match s {
        Pass { .. } => 0,
        PlayLand { .. } => 1,
        CastSpell { .. } => 2,
        ActivateManaAbility { .. } => 3,
        ActivateAbility { .. } => 4,
        PlotSpell { .. } => 5,
        ChooseTarget { .. } => 6,
        ChooseCostTarget { .. } => 7,
        ChooseCastMode { .. } => 8,
        ChooseKicker { .. } => 9,
        ChooseSpellMode { .. } => 10,
        ChooseEffectOption { .. } => 11,
        ChooseEffectTarget { .. } => 12,
        FinishEffectSelection { .. } => 13,
        ChooseEffectColor { .. } => 14,
        ChooseEffectNumber { .. } => 15,
        ChooseEffectBoolean { .. } => 16,
        FinishTargetSelection { .. } => 17,
        ChooseOptionalCostUse { .. } => 18,
        ChooseOptionalCostWhich { .. } => 19,
        ChooseSpellCopyPayment { .. } => 20,
        ChooseSpellCopyRetarget { .. } => 21,
        ChooseMadnessCast { .. } => 22,
        Discard { .. } => 23,
        DeclareAttackers { .. } => 24,
        DeclareBlockersForAttacker { .. } => 25,
        ChooseAttackerInclusion { .. } => 26,
        ChooseBlockerInclusion { .. } => 27,
        OrderTriggers { .. } => 28,
        Ambiguous { .. } => 39,
        ChooseCombatDamageRange { .. } => 30,
        ChooseLegendPermanent { .. } => 31,
        ChooseTriggerOrderNext { .. } => 32,
        ChooseLondonMulligan { .. } => 33,
        ChooseLondonBottom { .. } => 34,
    }
}

/// The decision family (index into `FAMILIES`) of a masked action list, from its first action.
pub fn decision_family(first: Option<&ActionSemanticV1>) -> usize {
    use ActionSemanticV1::*;
    match first {
        Some(Pass { .. } | PlayLand { .. } | CastSpell { .. } | ActivateAbility { .. } | PlotSpell { .. })
        | Some(ActivateManaAbility { .. }) => 0,
        Some(
            ChooseTarget { .. }
            | ChooseCostTarget { .. }
            | ChooseEffectTarget { .. }
            | FinishEffectSelection { .. }
            | FinishTargetSelection { .. },
        ) => 1,
        Some(ChooseAttackerInclusion { .. } | DeclareAttackers { .. }) => 2,
        Some(ChooseBlockerInclusion { .. } | DeclareBlockersForAttacker { .. }) => 3,
        Some(ChooseCombatDamageRange { .. }) => 4,
        Some(ChooseLondonMulligan { .. }) => 5,
        Some(ChooseLondonBottom { .. }) => 6,
        Some(OrderTriggers { .. } | ChooseTriggerOrderNext { .. }) => 7,
        Some(ChooseLegendPermanent { .. }) => 8,
        _ => 9,
    }
}

/// A dropped token's entry in `ArenaMap` (no identical kept token to stand in for it).
const DROPPED: u32 = u32::MAX;

/// Object tokens by arena id, for linking action references (exact zone first).
struct ArenaMap {
    /// (arena id, zone, token index or `DROPPED`, is a stack item)
    entries: Vec<(u32, Zone, u32, bool)>,
}

impl ArenaMap {
    fn find(&self, r: &CardStableRefV1) -> Option<usize> {
        // The card token of the same object in the same zone; else a stack item with that source and zone (a
        // spell on the stack; an ability's source is its permanent, whose own token wins); else any card token
        // with the arena id; else any token. A dropped token stays the match (it links nowhere), so a
        // reference never falls through to another object with its arena id.
        let pick = |zone: bool, stack: bool| {
            self.entries.iter().find(|e| e.0 == r.arena_id && (!zone || e.1 == r.zone) && e.3 == stack)
        };
        pick(true, false)
            .or_else(|| pick(true, true))
            .or_else(|| pick(false, false))
            .or_else(|| pick(false, true))
            .and_then(|e| (e.2 != DROPPED).then_some(e.2 as usize))
    }
}

/// The cap group of a zone category (index into `CAP_GROUPS`).
pub fn cap_group(zone: u8) -> usize {
    match zone {
        Z_MY_HAND => 0,
        Z_STACK_MINE | Z_STACK_OPP => 1,
        Z_MY_BF => 2,
        Z_OPP_BF => 3,
        _ => 4,
    }
}

/// Token quotas per cap group, summing to at most `cap`: groups are served smallest first, each up to its count
/// or an equal share of what is left, so a small group is kept whole and every non-empty group keeps at least
/// one token (`cap >= CAP_GROUPS`).
pub fn cap_quotas(counts: [usize; CAP_GROUPS], cap: usize) -> [usize; CAP_GROUPS] {
    let mut order: Vec<usize> = (0..CAP_GROUPS).filter(|&g| counts[g] > 0).collect();
    order.sort_by_key(|&g| (counts[g], g));
    let mut quota = [0; CAP_GROUPS];
    let mut left = cap;
    for (i, &g) in order.iter().enumerate() {
        quota[g] = counts[g].min(left / (order.len() - i));
        left -= quota[g];
    }
    quota
}

/// Enforce `MAX_OBJECTS` on a fully built token list: per-group quotas (`cap_quotas`); within an over-quota group
/// the first token of each distinct (card, zone, features) is kept before any duplicate, in token order. A
/// dropped token with an identical kept token links its action references to that one (the network computes
/// the same vector for both); one without links nowhere (`unlinked_refs`).
fn cap_objects(out: &mut Encoded, map: &mut ArenaMap) {
    let n = out.n_obj();
    if n <= MAX_OBJECTS {
        return;
    }
    let mut counts = [0usize; CAP_GROUPS];
    for &z in &out.obj_zone {
        counts[cap_group(z)] += 1;
    }
    let quota = cap_quotas(counts, MAX_OBJECTS);
    let key = |i: usize| -> (u16, u8, Vec<u32>) {
        (out.obj_card[i], out.obj_zone[i], out.obj_feat[i * F..(i + 1) * F].iter().map(|x| x.to_bits()).collect())
    };
    let keys: Vec<(u16, u8, Vec<u32>)> = (0..n).map(key).collect();
    let mut keep = vec![false; n];
    let mut used = [0usize; CAP_GROUPS];
    let mut seen = std::collections::HashSet::new();
    for i in 0..n {
        let g = cap_group(out.obj_zone[i]);
        if seen.insert(&keys[i]) && used[g] < quota[g] {
            keep[i] = true;
            used[g] += 1;
        }
    }
    for (k, &z) in keep.iter_mut().zip(&out.obj_zone) {
        let g = cap_group(z);
        if !*k && used[g] < quota[g] {
            *k = true;
            used[g] += 1;
        }
    }
    let mut new_index = vec![DROPPED; n];
    let mut kept_as = std::collections::HashMap::new();
    let mut m = 0u32;
    for i in 0..n {
        if keep[i] {
            new_index[i] = m;
            kept_as.entry(&keys[i]).or_insert(m);
            m += 1;
        }
    }
    for i in 0..n {
        if !keep[i] {
            new_index[i] = kept_as.get(&keys[i]).copied().unwrap_or(DROPPED);
        }
    }
    let (mut card, mut zone, mut feat) = (Vec::with_capacity(MAX_OBJECTS), Vec::with_capacity(MAX_OBJECTS), Vec::new());
    for i in (0..n).filter(|&i| keep[i]) {
        card.push(out.obj_card[i]);
        zone.push(out.obj_zone[i]);
        feat.extend_from_slice(&out.obj_feat[i * F..(i + 1) * F]);
    }
    out.obj_card = card;
    out.obj_zone = zone;
    out.obj_feat = feat;
    out.dropped_objects = (n - m as usize) as u32;
    for e in &mut map.entries {
        e.2 = new_index[e.2 as usize];
    }
}

/// A count feature that never saturates: ln(1 + x) / ln(8) (0 → 0, 7 → 1, 255 → 2.67).
fn log_count(x: usize) -> f32 {
    (1.0 + x as f32).ln() / 8f32.ln()
}

fn contains(refs: &[CardStableRefV1], arena: u32) -> bool {
    refs.iter().any(|r| r.arena_id == arena)
}

/// Static facts of a card definition into an object token's features (types, colours, keywords, base P/T,
/// mana value, token, mana source, activated abilities). Public characteristics override some of them.
fn static_facts(f: &mut [f32], d: &CardDef) {
    f[O_MANA_VALUE] = f32::from(d.mana_value) / 10.0;
    f[O_TOKEN] = f32::from(u8::from(d.is_token));
    for t in d.types {
        let k = match t {
            CardType::Creature => 0,
            CardType::Land => 1,
            CardType::Instant => 2,
            CardType::Sorcery => 3,
            CardType::Enchantment => 4,
            CardType::Artifact => 5,
            CardType::Planeswalker => 6,
        };
        f[O_TYPES + k] = 1.0;
    }
    let mut mask = 0u8;
    for &c in d.colors {
        if c != ManaColor::C {
            mask |= 1 << color_index(c);
        }
    }
    set_colors(f, mask);
    let p = d.power.map_or(0.0, f32::from) / 10.0;
    let t = d.toughness.map_or(0.0, f32::from) / 10.0;
    f[O_POWER] = p;
    f[O_TOUGHNESS] = t;
    f[O_BASE_POWER] = p;
    f[O_BASE_TOUGHNESS] = t;
    let kw = d.keywords;
    let flags = [
        Keywords::FLYING,
        Keywords::REACH,
        Keywords::HASTE,
        Keywords::VIGILANCE,
        Keywords::TRAMPLE,
        Keywords::FIRST_STRIKE,
        Keywords::DOUBLE_STRIKE,
        Keywords::DEATHTOUCH,
        Keywords::MENACE,
        Keywords::DEFENDER,
        Keywords::LIFELINK,
        Keywords::HEXPROOF,
        Keywords::INDESTRUCTIBLE,
        Keywords::PROTECTION_FROM_MONOCOLORED,
    ];
    for (i, k) in flags.iter().enumerate() {
        f[O_KEYWORDS + i] = f32::from(u8::from(kw.has(*k)));
    }
    f[O_LANDWALK] = f32::from(u8::from(kw.has(Keywords::ISLANDWALK)));
    f[O_FLASH] = f32::from(u8::from(kw.has(Keywords::FLASH)));
    f[O_UNBLOCKABLE] = f32::from(u8::from(kw.has(Keywords::CANT_BE_BLOCKED)));
    f[O_WARD] = d.ward_cost.map_or(0.0, |_| 1.0 / 3.0);
    f[O_MIN_BLOCKERS] = f32::from(u8::from(d.minimum_blockers > 1));
    f[O_MANA_SOURCE] = f32::from(u8::from(!d.mana_ability_choices.is_empty() || !d.produces_mana.is_empty()));
    f[O_HAS_ACTIVATED] = f32::from(u8::from(!d.activated_abilities.is_empty()));
}

fn set_colors(f: &mut [f32], mask: u8) {
    for i in 0..5 {
        f[O_COLORS + i] = f32::from((mask >> i) & 1);
    }
    f[O_COLORS + 5] = f32::from(u8::from(mask & 0x1f == 0));
}

/// Public (effective) characteristics over the static ones.
fn public_facts(f: &mut [f32], c: &CardCharacteristicsV2) {
    let tf = &c.type_flags;
    for (i, b) in [tf.creature, tf.land, tf.instant, tf.sorcery, tf.enchantment, tf.artifact].iter().enumerate() {
        f[O_TYPES + i] = f32::from(u8::from(*b));
    }
    set_colors(f, c.effective_color_mask);
    let pt = |x: Option<i32>| x.map_or(0.0, |v| v as f32 / 10.0);
    f[O_BASE_POWER] = pt(c.base_power);
    f[O_BASE_TOUGHNESS] = pt(c.base_toughness);
    f[O_POWER] = pt(c.effective_power.or(c.base_power));
    f[O_TOUGHNESS] = pt(c.effective_toughness.or(c.base_toughness));
    let k = &c.effective_keywords;
    let flags = [
        k.flying,
        k.reach,
        k.haste,
        k.vigilance,
        k.trample,
        k.first_strike,
        k.double_strike,
        k.deathtouch,
        k.menace,
        k.defender,
        k.lifelink,
        k.hexproof,
        k.indestructible,
        k.protection_from_monocolored,
    ];
    for (i, b) in flags.iter().enumerate() {
        f[O_KEYWORDS + i] = f32::from(u8::from(*b));
    }
    f[O_WARD] = f32::from(k.ward_generic) / 3.0;
    f[O_MIN_BLOCKERS] = f32::from(u8::from(k.minimum_blockers > 1));
    f[O_LANDWALK] = f32::from(u8::from(k.landwalk_mask != 0));
}

/// Encode one decision: the actor's observation and the masked legal actions (in the masked order).
pub fn encode(obs: &ObservationV5, actions: &[&LegalActionV5], out: &mut Encoded) {
    out.clear();
    let me = obs.acting_player;
    let mi = if me == PlayerSeatV1::P0 { 0 } else { 1 };
    let oi = 1 - mi;
    let is_me = |s: PlayerSeatV1| s == me;
    let proj = &obs.projection;
    let surf = &proj.surface;
    let turn = surf.turn;

    // ---------------------------------------------------------------- object tokens
    let mut map = ArenaMap { entries: Vec::with_capacity(64) };
    let add = |out: &mut Encoded, map: &mut ArenaMap, r: &CardStableRefV1, zone: u8, stack: bool| -> usize {
        let i = out.push_object(r.card_db_id, zone);
        if let Some(d) = def(r.card_db_id) {
            static_facts(out.of(i), d);
        }
        map.entries.push((r.arena_id, r.zone, i as u32, stack));
        i
    };
    // Token order (the network is permutation invariant; only which duplicates the cap keeps depends on it): my
    // hand, the stack (top first), the battlefields, the actor's known cards, then graveyards and exile.
    for c in &obs.own_hand {
        add(out, &mut map, &c.stable, Z_MY_HAND, false);
    }
    let n_stack = surf.stack.len();
    for (depth, item) in surf.stack.iter().rev().enumerate() {
        let zone = if is_me(item.controller) { Z_STACK_MINE } else { Z_STACK_OPP };
        let i = add(out, &mut map, &item.source, zone, true);
        let f = out.of(i);
        match item.stack_item_kind {
            StackItemKindV2::Spell => f[O_STACK_SPELL] = 1.0,
            StackItemKindV2::ActivatedAbility => f[O_STACK_ACTIVATED] = 1.0,
            StackItemKindV2::TriggeredAbility | StackItemKindV2::MadnessOffer => f[O_STACK_TRIGGERED] = 1.0,
        }
        f[O_STACK_DEPTH] = depth as f32 / 5.0;
        f[O_STACK_N_TARGETS] = item.targets.len() as f32 / 3.0;
        f[O_STACK_KICKED] = f32::from(u8::from(item.kicked));
        f[O_STACK_COPY] = f32::from(u8::from(item.is_copy));
        for t in &item.targets {
            if let TargetRefV1::Player { player } = t {
                if is_me(*player) {
                    f[O_STACK_TARGETS_ME] = 1.0;
                } else {
                    f[O_STACK_TARGETS_OPP] = 1.0;
                }
            }
        }
    }
    let combat = &surf.combat;
    let scan = proj.policy_surface_context.private_combat_selection.as_ref();
    let add_public = |out: &mut Encoded, map: &mut ArenaMap, c: &CardPublicV2, zone: u8| {
        let i = add(out, map, &c.stable, zone, false);
        let arena = c.stable.arena_id;
        let f = out.of(i);
        public_facts(f, &c.characteristics);
        f[O_TOKEN] = f32::from(u8::from(c.is_token));
        f[O_TAPPED] = f32::from(u8::from(c.tapped));
        f[O_SICK] = f32::from(u8::from(c.summoning_sick));
        f[O_DAMAGE] = f32::from(c.damage) / 10.0;
        f[O_PLUS1] = f32::from(c.counters.plus1_plus1) / 10.0;
        f[O_MINUS1] = f32::from(c.counters.minus1_minus1) / 10.0;
        f[O_OTHER_COUNTERS] =
            (f32::from(c.counters.minus0_minus1) + f32::from(c.counters.stun) + f32::from(c.counters.lore)) / 10.0;
        f[O_N_ATTACHMENTS] = c.attachments.len() as f32 / 3.0;
        f[O_ENTERED_THIS_TURN] = f32::from(u8::from(c.entered_battlefield_turn == Some(turn)));
        f[O_ABILITY_USED] = f32::from(u8::from(!c.ability_uses_this_turn.is_empty()));
        f[O_SKIP_UNTAP] = f32::from(u8::from(c.skip_next_untap));
        f[O_ATTACKING] = f32::from(u8::from(contains(&combat.ordered_attackers, arena)));
        for (att, blockers) in &combat.attacker_to_ordered_blockers {
            if att.arena_id == arena && !blockers.is_empty() {
                f[O_BLOCKED] = 1.0;
            }
            if contains(blockers, arena) {
                f[O_BLOCKING] = 1.0;
            }
        }
        if let Some(s) = scan {
            f[O_SCAN_SELECTED] = f32::from(u8::from(contains(&s.selected, arena)));
            f[O_SCAN_CURRENT] = f32::from(u8::from(s.current_candidate.arena_id == arena));
            f[O_SCAN_REMAINING] = f32::from(u8::from(contains(&s.remaining_after_current, arena)));
            f[O_SCAN_ATTACKER] = f32::from(u8::from(s.attacker.as_ref().is_some_and(|a| a.arena_id == arena)));
        }
        if let Some(pb) = &surf.surface_context.private_blockers {
            if pb.accumulated.iter().any(|(_, b)| b.arena_id == arena) {
                f[O_SCAN_SELECTED] = 1.0;
            }
            if pb.current_attacker.as_ref().is_some_and(|a| a.arena_id == arena) {
                f[O_SCAN_ATTACKER] = 1.0;
            }
        }
    };
    // battlefield lists are per player; the zone category follows the controller
    for p in [mi, oi] {
        for c in &surf.battlefield[p] {
            add_public(out, &mut map, c, if is_me(c.stable.controller) { Z_MY_BF } else { Z_OPP_BF });
        }
    }
    // the actor's known cards in the opponent's hand and in each library
    for c in &obs.known_hand_cards[oi] {
        add(out, &mut map, &c.stable, Z_OPP_HAND_KNOWN, false);
    }
    for (p, zone) in [(mi, Z_MY_LIB_KNOWN), (oi, Z_OPP_LIB_KNOWN)] {
        for k in &obs.known_library_cards[p] {
            let i = add(out, &mut map, &k.card.stable, zone, false);
            out.of(i)[O_LIBRARY_POSITION] = k.position as f32 / 40.0;
        }
    }
    // graveyards and exile: the zone category follows the owner
    for p in [mi, oi] {
        for c in &surf.graveyards[p] {
            add_public(out, &mut map, c, if is_me(c.stable.owner) { Z_MY_GY } else { Z_OPP_GY });
        }
    }
    for c in &surf.exile {
        add_public(out, &mut map, c, if is_me(c.stable.owner) { Z_MY_EXILE } else { Z_OPP_EXILE });
    }
    // targets of stack items (by controller) and of the pending cast / activation / effect
    let mark = |out: &mut Encoded, r: &CardStableRefV1, k: usize| {
        if let Some(i) = map.find(r) {
            out.of(i)[k] = 1.0;
        }
    };
    for item in &surf.stack {
        let k = if is_me(item.controller) { O_TARGETED_BY_MINE } else { O_TARGETED_BY_OPP };
        for t in &item.targets {
            if let TargetRefV1::Object { object } = t {
                mark(out, object, k);
            }
        }
    }
    let ec = &surf.engine_context;
    let mut pending_targets: Vec<&TargetRefV1> = Vec::new();
    if let Some(pc) = &ec.pending_cast {
        pending_targets.extend(pc.chosen_targets.iter());
    }
    if let Some(pa) = &ec.pending_activation {
        pending_targets.extend(pa.chosen_targets.iter());
    }
    let mut pending_minmax = None;
    if let Some(pe) = &ec.pending_effect {
        if let Some(PendingEffectChoiceSemanticV4::Targets { selected_targets, min_targets, max_targets, .. }) =
            &pe.choice
        {
            pending_targets.extend(selected_targets.iter());
            pending_minmax = Some((*min_targets, *max_targets, selected_targets.len()));
        }
    }
    let mut pending_player = [0.0f32; 2];
    for t in &pending_targets {
        match t {
            TargetRefV1::Object { object } => mark(out, object, O_PENDING_TARGET),
            TargetRefV1::Player { player } => pending_player[usize::from(!is_me(*player))] = 1.0,
        }
    }
    for rel in &surf.object_relations {
        if let ObjectRelationPublicV4::AttachedTo { object, attached_to } = rel {
            if let Some(i) = map.find(object) {
                let mine = map.find(attached_to).is_some_and(|h| out.obj_zone[h] == Z_MY_BF);
                let f = out.of(i);
                f[O_ATTACHED] = 1.0;
                f[O_ATTACHED_TO_MINE] = f32::from(u8::from(mine));
            }
        }
    }

    // ---------------------------------------------------------------- global
    out.global.resize(G, 0.0);
    let g = &mut out.global;
    for (k, p) in [mi, oi].into_iter().enumerate() {
        g[G_LIFE + k] = surf.life_totals[p] as f32 / 20.0;
        g[G_HAND + k] = surf.hand_counts[p] as f32 / 10.0;
        g[G_LIBRARY + k] = surf.library_counts[p] as f32 / 40.0;
        g[G_GRAVE + k] = surf.graveyards[p].len() as f32 / 20.0;
        g[G_SPELLS_CAST + k] = f32::from(surf.player_status[p].spells_cast_this_turn) / 5.0;
        g[G_PASSED + k] = f32::from(u8::from(ec.priority_passes[p]));
        g[G_HAS_LOST + k] = f32::from(u8::from(surf.player_status[p].has_lost));
        g[G_PENDING_TARGETS_PLAYER + k] = pending_player[k];
    }
    let pools = [&surf.mana_pools[mi], &surf.mana_pools[oi]];
    for c in 0..6 {
        g[G_POOL_ME + c] = f32::from(pools[0][c]) / 5.0;
        g[G_POOL_OPP + c] = f32::from(pools[1][c]) / 5.0;
    }
    g[G_EXILE] = surf.exile.iter().filter(|c| is_me(c.stable.owner)).count() as f32 / 10.0;
    g[G_EXILE + 1] = surf.exile.iter().filter(|c| !is_me(c.stable.owner)).count() as f32 / 10.0;
    g[G_TURN] = turn as f32 / 30.0;
    g[G_ACTIVE] = f32::from(u8::from(is_me(surf.active_player)));
    g[G_PRIORITY] = f32::from(u8::from(is_me(surf.priority_player)));
    g[G_PHASE + step_index(surf.phase)] = 1.0;
    let mut power = [0.0f32; 2];
    for p in 0..2 {
        for c in &surf.battlefield[p] {
            let k = usize::from(!is_me(c.stable.controller));
            let tf = &c.characteristics.type_flags;
            if tf.land {
                g[G_LANDS + k] += 0.1;
                if !c.tapped {
                    g[G_UNTAPPED_LANDS + k] += 0.1;
                }
            }
            if tf.creature {
                g[G_CREATURES + k] += 0.1;
                if !c.tapped {
                    g[G_UNTAPPED_CREATURES + k] += 0.1;
                }
                power[k] += c.characteristics.effective_power.or(c.characteristics.base_power).unwrap_or(0) as f32;
            }
        }
    }
    g[G_POWER] = power[0] / 20.0;
    g[G_POWER + 1] = power[1] / 20.0;
    g[G_LANDS_PLAYED] = f32::from(surf.player_status[mi].lands_played_this_turn);
    g[G_DRAWS] = surf.player_status[mi].draws_this_turn as f32 / 5.0;
    if let Some(m) = &proj.london_mulligans {
        g[G_MULL_ACTIVE] = f32::from(u8::from(!(m.kept[0] && m.kept[1])));
        g[G_MULL_COUNT] = f32::from(m.counts[mi]) / 7.0;
        g[G_MULL_COUNT + 1] = f32::from(m.counts[oi]) / 7.0;
        g[G_MULL_KEPT] = f32::from(u8::from(m.kept[mi]));
        g[G_MULL_KEPT + 1] = f32::from(u8::from(m.kept[oi]));
    }
    g[G_FAMILY + decision_family(actions.first().map(|a| &a.semantic))] = 1.0;
    g[G_STACK] = n_stack as f32 / 5.0;
    if let Some(top) = surf.stack.last() {
        g[if is_me(top.controller) { G_STACK_TOP_MINE } else { G_STACK_TOP_OPP }] = 1.0;
    }
    g[G_ATTACKERS_DECLARED] = f32::from(u8::from(combat.attackers_declared));
    g[G_BLOCKERS_DECLARED] = f32::from(u8::from(combat.blockers_declared));
    g[G_N_ATTACKERS] = combat.ordered_attackers.len() as f32 / 5.0;
    match proj.policy_surface_context.current_stage {
        PolicySurfaceStageV5::AttackerInclusion => g[G_STAGE_ATTACK] = 1.0,
        PolicySurfaceStageV5::BlockerInclusion => g[G_STAGE_BLOCK] = 1.0,
        PolicySurfaceStageV5::Surface => {}
    }
    if let Some(s) = scan {
        g[G_SCAN_PROGRESS] = s.candidate_index as f32 / s.candidate_count.max(1) as f32;
        g[G_SCAN_COUNT] = s.candidate_count as f32 / 10.0;
        g[G_SCAN_SELECTED] = s.selected.len() as f32 / 5.0;
    }
    g[G_N_ACTIONS] = (1.0 + actions.len() as f32).ln() / 5.0;
    g[G_ON_THE_PLAY] = f32::from(u8::from(me == PlayerSeatV1::P0));
    g[G_PENDING_CAST] = f32::from(u8::from(ec.pending_cast.is_some() || ec.pending_activation.is_some()));
    g[G_PENDING_EFFECT] = f32::from(u8::from(ec.pending_effect.is_some()));
    if let Some((lo, hi, sel)) = pending_minmax {
        g[G_PENDING_MIN] = f32::from(lo) / 5.0;
        g[G_PENDING_MAX] = f32::from(hi) / 5.0;
        g[G_PENDING_SELECTED] = sel as f32 / 5.0;
    }
    g[G_DAMAGE_ASSIGNMENT] = f32::from(u8::from(proj.foundations_combat.is_some()));
    g[G_OPP_KNOWN_HAND] = obs.known_hand_cards[oi].len() as f32 / 5.0;

    // ---------------------------------------------------------------- the cap, then action tokens
    cap_objects(out, &mut map);
    for a in actions {
        encode_action(&a.semantic, me, &map, out);
    }
    // outliers (a hundred tokens, a thousand life) stay bounded
    for x in out.global.iter_mut().chain(out.obj_feat.iter_mut()).chain(out.act_feat.iter_mut()) {
        *x = x.clamp(-FEATURE_CLAMP, FEATURE_CLAMP);
    }
}

fn encode_action(s: &ActionSemanticV1, me: PlayerSeatV1, map: &ArenaMap, out: &mut Encoded) {
    use ActionSemanticV1::*;
    let mut x = [0.0f32; A];
    let mut src: Option<&CardStableRefV1> = None;
    let mut tgt: Option<&CardStableRefV1> = None;
    let mut tgt_player = 0u8;
    let player = |p: &PlayerSeatV1| if *p == me { 1 } else { 2 };
    match s {
        Pass { .. } | Ambiguous { .. } => {}
        PlayLand { source, .. } | CastSpell { source, .. } | PlotSpell { source, .. } => src = Some(source),
        ActivateManaAbility { source, mana_choice, cost_target, .. } => {
            src = Some(source);
            tgt = cost_target.as_ref();
            if let Some(c) = mana_choice {
                x[X_COLOR + color_index(*c)] = 1.0;
            }
        }
        ActivateAbility { source, ability_index, .. } => {
            src = Some(source);
            x[X_ABILITY] = f32::from(*ability_index) / 3.0;
        }
        ChooseTarget { source, remaining, target, .. } => {
            src = Some(source);
            x[X_REMAINING] = f32::from(*remaining) / 5.0;
            match target {
                TargetRefV1::Object { object } => tgt = Some(object),
                TargetRefV1::Player { player: p } => tgt_player = player(p),
            }
        }
        ChooseCostTarget { source, cost_kind, remaining, candidate, .. } => {
            src = Some(source);
            tgt = Some(candidate);
            x[X_REMAINING] = f32::from(*remaining) / 5.0;
            x[X_COST_KIND] = (cost_kind_index(*cost_kind) + 1) as f32 / 12.0;
        }
        ChooseCastMode { source, mode, .. } => {
            src = Some(source);
            x[X_ALT_COST] = f32::from(u8::from(*mode == CastMode::Alternative));
        }
        ChooseKicker { source, pay, .. } | ChooseSpellCopyPayment { source, pay, .. } => {
            src = Some(source);
            x[X_BOOL] = f32::from(u8::from(*pay));
        }
        ChooseSpellMode { source, mode_index, mode_count, .. } => {
            src = Some(source);
            x[X_MODE] = f32::from(*mode_index) / 3.0;
            x[X_MODE_COUNT] = f32::from(*mode_count) / 3.0;
        }
        ChooseEffectOption { source, option_index, option_count, .. } => {
            src = Some(source);
            x[X_OPTION] = f32::from(*option_index) / 5.0;
            x[X_OPTION_COUNT] = f32::from(*option_count) / 5.0;
        }
        ChooseEffectTarget { source, target, selected_count, min_targets, max_targets, .. } => {
            src = Some(source);
            match target {
                TargetRefV1::Object { object } => tgt = Some(object),
                TargetRefV1::Player { player: p } => tgt_player = player(p),
            }
            x[X_SELECTED] = f32::from(*selected_count) / 5.0;
            x[X_MIN_TARGETS] = f32::from(*min_targets) / 5.0;
            x[X_MAX_TARGETS] = f32::from(*max_targets) / 5.0;
        }
        FinishEffectSelection { source, selected_count, .. } | FinishTargetSelection { source, selected_count, .. } => {
            src = Some(source);
            x[X_SELECTED] = f32::from(*selected_count) / 5.0;
        }
        ChooseEffectColor { source, color, .. } => {
            src = Some(source);
            x[X_COLOR + color_index(*color)] = 1.0;
        }
        ChooseEffectNumber { source, number, minimum, maximum, .. } => {
            src = Some(source);
            x[X_NUMBER] = *number as f32 / 10.0;
            x[X_NUMBER_MIN] = *minimum as f32 / 10.0;
            x[X_NUMBER_MAX] = *maximum as f32 / 10.0;
        }
        ChooseEffectBoolean { source, value, .. } => {
            src = Some(source);
            x[X_BOOL] = f32::from(u8::from(*value));
        }
        ChooseOptionalCostUse { use_cost, .. } => x[X_BOOL] = f32::from(u8::from(*use_cost)),
        ChooseOptionalCostWhich { choice, .. } => {
            x[X_COST_KIND] = (match choice {
                OptionalCostChoice::Decline => 0,
                OptionalCostChoice::Discard => 1,
                OptionalCostChoice::SacrificeLand => 2,
                OptionalCostChoice::ReturnPermanent => 3,
            }) as f32
                / 4.0
        }
        ChooseSpellCopyRetarget { source, change_target, .. } => {
            src = Some(source);
            x[X_BOOL] = f32::from(u8::from(*change_target));
        }
        ChooseMadnessCast { card, cast_it, .. } => {
            src = Some(card);
            x[X_BOOL] = f32::from(u8::from(*cast_it));
        }
        Discard { cards, .. } => {
            src = cards.first();
            tgt = if cards.len() > 1 { cards.last() } else { None };
            x[X_N_OBJECTS] = cards.len() as f32 / 5.0;
        }
        DeclareAttackers { attackers, .. } => {
            src = attackers.first();
            tgt = if attackers.len() > 1 { attackers.last() } else { None };
            x[X_N_OBJECTS] = attackers.len() as f32 / 5.0;
        }
        DeclareBlockersForAttacker { attacker, blockers, .. } => {
            src = Some(attacker);
            tgt = blockers.first();
            x[X_N_OBJECTS] = blockers.len() as f32 / 5.0;
        }
        ChooseAttackerInclusion { attacker, include, .. } => {
            src = Some(attacker);
            x[X_INCLUDE] = f32::from(u8::from(*include));
        }
        ChooseBlockerInclusion { attacker, blocker, include, .. } => {
            src = Some(blocker);
            tgt = Some(attacker);
            x[X_INCLUDE] = f32::from(u8::from(*include));
        }
        OrderTriggers { pending_sources, order, .. } => {
            // first and last trigger of the order (the full permutation is not encoded)
            src = order.first().and_then(|&i| pending_sources.get(i));
            tgt = order.last().and_then(|&i| pending_sources.get(i));
            x[X_PENDING_COUNT] = log_count(pending_sources.len());
            x[X_PREFIX] = log_count(order.len());
        }
        ChooseCombatDamageRange { source, recipient, minimum, maximum, split_at, upper_half, .. } => {
            src = Some(source);
            match recipient {
                TargetRefV1::Object { object } => tgt = Some(object),
                TargetRefV1::Player { player: p } => tgt_player = player(p),
            }
            x[X_DMG_MIN] = *minimum as f32 / 10.0;
            x[X_DMG_MAX] = *maximum as f32 / 10.0;
            x[X_DMG_SPLIT] = *split_at as f32 / 10.0;
            x[X_DMG_UPPER] = f32::from(u8::from(*upper_half));
        }
        ChooseLegendPermanent { keep, candidates, .. } => {
            src = Some(keep);
            x[X_N_OBJECTS] = candidates.len() as f32 / 5.0;
        }
        ChooseTriggerOrderNext { source, trigger_index, ordered_prefix, pending_count, .. } => {
            src = Some(source);
            x[X_TRIGGER_INDEX] = log_count(*trigger_index);
            x[X_PREFIX] = log_count(ordered_prefix.len());
            x[X_PENDING_COUNT] = log_count(*pending_count);
        }
        ChooseLondonMulligan { mulligan_count, mulligan, .. } => {
            x[X_MULLIGAN] = f32::from(u8::from(*mulligan));
            x[X_MULLIGAN_COUNT] = f32::from(*mulligan_count) / 7.0;
        }
        ChooseLondonBottom { remaining, card, .. } => {
            src = Some(card);
            x[X_REMAINING] = f32::from(*remaining) / 5.0;
        }
    }
    let mut link = |r: Option<&CardStableRefV1>| -> (i16, u16) {
        match r {
            None => (-1, 0),
            Some(r) => {
                let i = map.find(r);
                if i.is_none() {
                    out.unlinked_refs += 1;
                }
                (i.map_or(-1, |i| i as i16), vocab_id(r.card_db_id))
            }
        }
    };
    let (si, sc) = link(src);
    let (ti, tc) = link(tgt);
    out.act_kind.push(action_kind(s));
    out.act_src.push(si);
    out.act_src_card.push(sc);
    out.act_tgt.push(ti);
    out.act_tgt_card.push(tc);
    out.act_tgt_player.push(tgt_player);
    out.act_feat.extend_from_slice(&x);
}

fn cost_kind_index(k: CostKind) -> usize {
    match k {
        CostKind::SacrificeLands => 0,
        CostKind::SacrificePermanents => 1,
        CostKind::SacrificeCreatures => 2,
        CostKind::SacrificeArtifacts => 3,
        CostKind::DiscardCards => 4,
        CostKind::ExileFromGraveyard => 5,
        CostKind::TapPermanents => 6,
        CostKind::ReturnPermanentsToHand => 7,
        CostKind::PayLife => 8,
        CostKind::RemoveCounters => 9,
        CostKind::PutCounters => 10,
        CostKind::ChooseCreatureOrRevealCreature => 11,
    }
}

/// Why a decision could not be encoded (the caller falls back and counts it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    /// The game is at a terminal (or the session holds no observation for the decision).
    NoObservation,
}

/// Encode `game`'s current decision from its actor's seat: `game.observation()` and the masked choices only.
pub fn encode_game(game: &crate::game::Game, out: &mut Encoded) -> Result<(), EncodeError> {
    let obs = game.observation().ok_or(EncodeError::NoObservation)?;
    let actions: Vec<&LegalActionV5> = (0..game.num_choices()).map(|k| game.choice(k)).collect();
    encode(obs, &actions, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_quotas_keep_every_group() {
        assert_eq!(cap_quotas([7, 0, 300, 300, 50], MAX_OBJECTS), [7, 0, 99, 99, 50]);
        assert_eq!(cap_quotas([7, 2, 3, 4, 5], MAX_OBJECTS), [7, 2, 3, 4, 5]);
        assert_eq!(cap_quotas([0, 0, 1000, 1, 0], MAX_OBJECTS), [0, 0, 254, 1, 0]);
        for counts in [[1, 1, 1, 1, 1000], [0, 3, 500, 400, 300], [255, 255, 255, 255, 255]] {
            let q = cap_quotas(counts, MAX_OBJECTS);
            assert!(q.iter().sum::<usize>() <= MAX_OBJECTS);
            assert!((0..CAP_GROUPS).all(|g| (counts[g] > 0) == (q[g] > 0) && q[g] <= counts[g]), "{q:?}");
        }
    }

    /// 300 identical tokens of mine, 40 distinct of the opponent's, 3 in hand: every group keeps tokens, distinct
    /// tokens are kept before duplicates, a dropped duplicate links to its kept twin.
    #[test]
    fn cap_objects_keeps_groups_and_links_twins() {
        let mut out = Encoded::default();
        let mut map = ArenaMap { entries: Vec::new() };
        let mut push = |out: &mut Encoded, arena: u32, card: u16, zone: u8, power: f32| {
            let i = out.push_object(card, zone);
            out.of(i)[O_POWER] = power;
            map.entries.push((arena, Zone::Battlefield, i as u32, false));
        };
        for a in 0..3 {
            push(&mut out, a, 10 + a as u16, Z_MY_HAND, 0.0);
        }
        for a in 100..400 {
            push(&mut out, a, 7, Z_MY_BF, 0.4);
        }
        push(&mut out, 400, 8, Z_MY_BF, 0.1); // a distinct permanent of mine after the 300 copies
        for a in 500..540 {
            push(&mut out, a, 20, Z_OPP_BF, (a - 500) as f32 / 10.0);
        }
        cap_objects(&mut out, &mut map);
        assert_eq!(out.n_obj(), MAX_OBJECTS);
        assert_eq!(out.dropped_objects as usize, 3 + 301 + 40 - MAX_OBJECTS);
        let count = |z: u8| out.obj_zone.iter().filter(|&&x| x == z).count();
        assert_eq!((count(Z_MY_HAND), count(Z_OPP_BF), count(Z_MY_BF)), (3, 40, MAX_OBJECTS - 43));
        // the distinct permanent of mine is kept although it comes after the copies
        assert!(out.obj_card.contains(&vocab_id(8)));
        // every reference links: kept tokens to themselves, dropped copies to a kept copy
        let r = |arena: u32| CardStableRefV1 {
            arena_id: arena,
            card_db_id: 7,
            zone: Zone::Battlefield,
            owner: PlayerSeatV1::P0,
            controller: PlayerSeatV1::P0,
            zone_change_count: 0,
        };
        for a in (0..3).chain(100..401).chain(500..540) {
            let i = map.find(&r(a)).unwrap_or_else(|| panic!("arena {a} unlinked"));
            assert!(i < MAX_OBJECTS);
        }
        let twin = map.find(&r(399)).unwrap();
        assert_eq!((out.obj_card[twin], out.obj_zone[twin]), (vocab_id(7), Z_MY_BF));
    }

    #[test]
    fn log_counts_stay_distinct() {
        assert_eq!(log_count(0), 0.0);
        assert!((log_count(7) - 1.0).abs() < 1e-6);
        assert!(log_count(57) > log_count(56) && log_count(5040) < FEATURE_CLAMP);
    }
}
