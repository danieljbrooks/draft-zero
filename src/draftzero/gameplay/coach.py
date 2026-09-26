"""
coach.py — grade a human's decisions with MageZero's search: expected win-probability loss over
the hidden information, never with hindsight.

For one decision (a StateSpec plus the human's answer) `coach_decision` asks the bridge
(java/mzbridge, `coach` op) to rate every legal option with MageZero's MCTS over K determinizations
of the hidden cards, then ranks the options, measures the human's regret against the noise of the
search and grades it. `coach_17lands` builds the decision from a 17lands replay row and turn,
`coach_arena` coaches every non-trivial local decision (priority, attack, block) of an MTG Arena
log, and `render_markdown` writes the report.

The recipe (docs/008 research: web_literature.md §7.7, critique.md E5/E6):

  hidden information  The opponent's hand and library are sampled, never read from the log.
                      A partial spec (B.handUnknown > 0) or a placeholder B decklist is
                      determinized by the opponent belief model (`belief.OpponentModel`: 17lands
                      decks of the colours seen so far, hand-retention weights on; an exact B
                      decklist keeps its cards and only the hand is drawn), one spec per
                      determinization (bridge `options.specs`). When the model does not apply (a
                      non-FDN game such as a Cube log: most known cards are outside its pool), or
                      the spec is already complete, the bridge re-draws B's hand from B's library
                      as built (`resample`). Which one ran is recorded in `Verdict.determinization`.
                      hindsight=True (17lands mirrored pairs only) coaches the TRUE position (B's
                      real deck and hand): for evaluating the coach, never as a verdict for a
                      player (it is flagged `hindsight`). So is any verdict that kept part of the
                      truth: an exact B list with an unknown hand card, or an exact B list of a
                      logged game coached without hindsight=True (determinization.hindsight_partial).
  unit                Q is MageZero's backed-up value of an option from the decision player's view,
                      in [-1, 1] (a terminal win is +1, a loss -1, discounted 0.99 per ply). Read
                      as an expected outcome, P(win) = (1 + Q) / 2, so a regret of r (in Q) is a
                      win-probability loss of r / 2. That is an ASSUMPTION: the offline evaluator
                      scores non-terminal leaves with tanh(material difference / 15), which is not
                      calibrated, and the discount shrinks values; the win-probability numbers are
                      nominal until a calibrated value network replaces it.
  noise               Every option's mean Q comes with its sd over the K determinizations. A
                      regret is SIGNIFICANT when it exceeds Z_SIG x the standard error of the
                      paired per-determinization differences (both options searched in the same
                      K samples), and ROBUST when it also exceeds ROBUST_SD x the pooled sd of the
                      two options' Q (it holds whatever the hidden cards are).
  search bias         Neither test sees the search's own error. A child's Q is the MEAN value of
                      the lines searched below it, so an option the search rarely visits is
                      valued by its exploratory lines, not its best continuation, and the regret
                      against the well-searched best is inflated. The K searches share that bias
                      (with fixed hidden cards they can even be identical: paired SE 0). Measured
                      on 9 17lands decisions graded mistake or blunder at 300 simulations (K=4;
                      the human's option 5-43 visits per determinization; 6 were a land drop
                      graded against casting a spell first): at 3,000 simulations 7 became fine
                      or best, 1 an inaccuracy and 1 stayed a blunder. So an inaccuracy or worse
                      whose human option got fewer than MIN_VISITS visits per determinization is
                      searched again at VERIFY_BUDGET (offline, same determinizations; flag
                      verified, the first pass kept in noise.first_pass), and one still under
                      MIN_VISITS is flagged thin_search (a caution flag). MIN_VISITS is
                      provisional: at 1,000 simulations two faults whose human option had 21 and
                      60 visits per determinization were still artefacts (fine or best at 3,000).
  classes             best         the human's option ranks first
                      fine         not significant, a near tie with the best, or a loss < FINE_WP
                      inaccuracy   significant and a loss >= FINE_WP (and not a robust mistake)
                      mistake      significant, robust and a loss >= MISTAKE_WP
                      blunder      significant, robust and a loss >= BLUNDER_WP
                      (thresholds after chess.com's expected-points classes, scaled for MTG noise;
                      provisional until E5 checks that regret falls with player skill)
                      Not graded: trivial (< 2 options), ungraded (the human's answer is not one of
                      the options, the bridge reached another decision, ...), error (build failed).
  set-valued humans   17lands records a turn's actions without order, so the answer to the turn's
                      first main-phase decision is one of them (plus Pass when a post-combat play is
                      possible: the user attacked, or did nothing). The regret is reported for the
                      best member (best-of-set, the charitable reading, used for the grade) and for
                      their mean (mean-of-set).
  flags               trivial, unstable (the best option differs across determinizations), near_tie,
                      low_fidelity (tier T2/T3 or a risky spec flag, substituted cards),
                      combat_low_budget (offline combat verdicts below COMBAT_BUDGET simulations
                      are noise: the default raises the budget instead and flags
                      combat_budget_raised), verified / thin_search (search bias, above),
                      hindsight (the opponent's real deck or hand was used), inconsistent
                      (determinizations reached different questions, or another one than the
                      build the human's answer was matched on), best_not_offered (the engine's
                      best was not on the human's list, e.g. an Arena-less stand-in option: the
                      regret is measured against the best option that was), ...
  luck                Kept apart from skill: between two coached decisions of one game, the change
                      of the best option's value not explained by the human's regret
                      (V(next) - Q(human's option)) is the draws plus the opponent's play.
  validation          `validate_17lands` groups the loss by the 17lands users' win-rate bucket and
                      rank: a coach worth trusting gives lower losses to stronger players.

Usage:
  dz gameplay coach 17lands --row 4 --turn 5 [--hindsight] [--blocks] [-K 8] [--budget 300] [--out r.md]
                            [--verify-budget 3000]    # 0: no re-search of thinly searched faults
  dz gameplay coach arena [LOG ...] [--substitute-missing Plains --substitute-token GoblinToken]
  dz gameplay coach spec path/to/spec.json [--human "Play Plains"] [--human-set "Cast Stab" ...]
  dz gameplay coach validate [--limit 200 --every 2003]     # E5 pilot: regret by player skill

  from draftzero.gameplay import bridge, coach
  with bridge.Bridge("coach0") as b:
      v = coach.coach_decision(spec, b, human="Play Plains")      # a label, a set of labels, or a resolver
      v.classification, v.human["regret"], v.flags
      v = coach.coach_17lands(4, 5, bridge=b)
      r = coach.coach_arena(bridge=b, substitute={"missing": "Plains", "missingToken": "GoblinToken"})
      coach.write_arena_report(r)                                  # data/gameplay/coach/, privacy-checked

Privacy: Arena reports are written only under data/gameplay/ (gitignored), after
`arena.check_privacy` passed on exactly the text written; stdout gets aggregate counts. Reports never
name the partner row of a mirrored pair or any draft id.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from draftzero.gameplay.statespec import StateSpec

REPO = Path(__file__).resolve().parents[3]
GAMEPLAY_DATA = REPO / "data" / "gameplay"
DEFAULT_OUT = GAMEPLAY_DATA / "coach"

K_DEFAULT = 8
BUDGET_DEFAULT = 300
COMBAT_BUDGET = 1000        # offline attack/block verdicts below this are noise (bridge review, README)
VERIFY_BUDGET = 3000        # offline re-search of an inaccuracy or worse on a thinly searched human option
MIN_VISITS = 100            # visits per determinization below which an option's Q is its exploration's mean
Z_SIG = 2.0                 # significance: regret > Z_SIG x paired standard error
ROBUST_SD = 2.0             # mistake / blunder: regret > ROBUST_SD x pooled sd of the two options
FINE_WP = 0.02              # win-probability loss below this is fine
MISTAKE_WP = 0.05
BLUNDER_WP = 0.10
NEAR_TIE_WP = 0.01          # options this close to the best (or not separable from it) tie with it
UNSTABLE_AGREEMENT = 0.75   # unstable: the best option is the argmax in fewer determinizations than this
BELIEF_MIN_COVERAGE = 0.8   # the belief model applies when this share of the known cards is in its pool
LOG_SOURCES = ("17lands", "arena")   # provenance.source of specs from logged human games

GRADED = ("best", "fine", "inaccuracy", "mistake", "blunder")
FAULTS = ("inaccuracy", "mistake", "blunder")
CLASSES = GRADED + ("ungraded", "trivial", "error")
COMBAT_STEPS = ("DECLARE_ATTACKERS", "DECLARE_BLOCKERS")
# spec flags that make a state an approximation of the logged one (arena.py / reconstruct.py)
RISKY_FLAGS = ("control_changed", "mid_action", "stack_ability_omitted", "back_face", "stale_after_gap",
               "stack_multi_target_slot", "stack_target_unrepresentable", "combat_by_control_changed_omitted",
               "annotations_unverified_after_gap", "unknown_card", "A.cards_unaccounted", "pair_view_mismatch",
               "history_unexplained", "attacker_unmatched", "prev_snapshot_missing",
               # 17lands block specs: the attackers' "whenever ... attacks" triggers (evasion, "can't
               # block") are not played, so the legal blocks are not the real game's
               "attack_triggers_skipped")
# flags that make a grade untrustworthy (shown next to it and kept out of the trusted mistake list)
CAUTION_FLAGS = ("unstable", "low_fidelity", "combat_low_budget", "hindsight", "inconsistent",
                 "legal_set_differs", "thin_search", "pair_status_unknown")


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------

@dataclass
class OptionStat:
    """One legal option, aggregated over the determinizations (the bridge's `aggregate`)."""
    label: str
    idx: int | None = None
    mean_q: float | None = None      # None: never visited
    sd_q: float = 0.0
    visit_share: float = 0.0
    n: int = 0                       # visits summed over determinizations
    n_det: int = 0
    rank: int = 0
    argmax_share: float = 0.0        # share of determinizations where this option has the highest Q

    @property
    def win_prob(self) -> float | None:
        return win_prob(self.mean_q)


@dataclass
class HumanAction:
    """What the human did, as candidate labels for the decision the bridge reached.
    labels empty + reason: the answer cannot be graded here."""
    labels: list[str] = field(default_factory=list)
    set_valued: bool = False
    source: str = ""
    reason: str | None = None
    note: str | None = None


@dataclass
class Verdict:
    ref: str = ""
    source: str = ""
    classification: str = "ungraded"
    reason: str | None = None
    decision: dict = field(default_factory=dict)       # player, type, text, turn, phase, step, passedBefore, legal
    options: list[OptionStat] = field(default_factory=list)
    best: str | None = None
    human: dict | None = None
    flags: list[str] = field(default_factory=list)
    determinization: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    noise: dict = field(default_factory=dict)
    value: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    fidelity: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)        # game, decision, arena labels summary, ...
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def graded(self) -> bool:
        return self.classification in GRADED

    @property
    def caution(self) -> list[str]:
        return [f for f in self.flags if f.split(":")[0] in CAUTION_FLAGS]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["caution"] = self.caution
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Verdict":
        """Back from to_dict() (e.g. a verdicts.jsonl line), to re-render a report without searching."""
        names = {f for f in cls.__dataclass_fields__}
        v = cls(**{k: x for k, x in d.items() if k in names and k != "options"})
        v.options = [OptionStat(**o) for o in d.get("options") or []]
        return v


def win_prob(q: float | None) -> float | None:
    """P(win) read from a value in [-1, 1] as an expected outcome (+1 win, -1 loss)."""
    return None if q is None else (1.0 + q) / 2.0


def wp_loss(regret: float | None) -> float | None:
    """Win-probability loss of a regret in Q units (see the module docstring for the assumption)."""
    return None if regret is None else regret / 2.0


def remote(port: int = 50052, host: str = "127.0.0.1") -> dict:
    """The evaluator option for a MageZero inference server (see java/mzbridge/README.md)."""
    return {"type": "remote", "host": host, "port": int(port)}


def _evaluator(e) -> dict:
    if e is None or e == "offline":
        return {"type": "offline"}
    if isinstance(e, dict):
        return e
    if isinstance(e, str) and e.startswith("remote"):
        _, _, port = e.partition(":")
        return remote(int(port)) if port else remote()
    raise ValueError(f"evaluator must be 'offline', 'remote[:PORT]' or a dict, not {e!r}")


# ------------------------------------------------------------------------------------------------
# matching human answers to the bridge's labels
# ------------------------------------------------------------------------------------------------

_REMINDER = re.compile(r"\s*<i>\(.*?\)</i>")


def match_label(action: str, legal: Iterable[str], decision_type: str | None = None) -> str | None:
    """The legal label a human action names: exact; modulo XMage reminder text (Arena keys); a
    17lands ability text without its cost ("Create a 2/1 ..." for "-2: Create a 2/1 ..."); any case.
    For yes/no questions, true/false/attack map to yes/no."""
    legal = list(legal)
    want = (action or "").strip()
    if decision_type == "CHOOSE_USE":
        w = want.lower()
        want = "yes" if w in ("true", "1", "yes", "attack") else "no" if w in ("false", "0", "no") else want
    if want in legal:
        return want
    for lab in legal:
        if _REMINDER.sub("", lab) == want:
            return lab
    for lab in legal:
        i = lab.find(": ")
        if i > 0 and lab[i + 2:] == want:
            return lab
    for lab in legal:
        if lab.casefold() == want.casefold():
            return lab
    return None


def _as_human(h, decision: dict, names: dict) -> HumanAction | None:
    if h is None:
        return None
    if callable(h):
        h = h(decision, names)
        if h is None:
            return HumanAction(reason="the human's answer to this question is not known")
    if isinstance(h, HumanAction):
        return h
    if isinstance(h, str):
        return HumanAction([h], False, "given")
    h = list(dict.fromkeys(h))
    return HumanAction(h, len(h) > 1, "given")


# ------------------------------------------------------------------------------------------------
# grading (pure: unit-tested with made-up numbers)
# ------------------------------------------------------------------------------------------------

def per_determinization_q(response: dict) -> list[dict[str, float]]:
    """{label: Q} per determinization, duplicate children merged as the bridge does (N summed, Q
    visit-weighted); unvisited children left out."""
    out = []
    for d in response.get("determinizations") or []:
        acc: dict[str, list[float]] = {}
        for c in d.get("children") or []:
            if c.get("Q") is None or not c.get("N"):
                continue
            a = acc.setdefault(c["label"], [0.0, 0.0])
            a[0] += c["N"] * c["Q"]
            a[1] += c["N"]
        out.append({k: s / n for k, (s, n) in acc.items() if n})
    return out


def options_from(response: dict, per_det: list[dict[str, float]]) -> list[OptionStat]:
    k = max(len(per_det), 1)
    argmax = Counter(max(d, key=d.get) for d in per_det if d)
    out = []
    for a in response.get("aggregate") or []:
        out.append(OptionStat(a["label"], a.get("idx"), a.get("meanQ"), a.get("sdQ") or 0.0,
                              a.get("visitShare") or 0.0, a.get("N") or 0, a.get("nDet") or 0,
                              a.get("rank") or 0, argmax.get(a["label"], 0) / k))
    return out


def _paired(per_det: list[dict[str, float]], a: str, b: str) -> tuple[float | None, int]:
    """(standard error of mean(Q_a - Q_b) over the determinizations where both were searched, n)."""
    diffs = [d[a] - d[b] for d in per_det if a in d and b in d]
    if len(diffs) < 2:
        return None, len(diffs)
    return statistics.stdev(diffs) / math.sqrt(len(diffs)), len(diffs)


def separable(per_det: list[dict[str, float]], best: OptionStat, other: OptionStat) -> bool:
    """The best option is clearly better than `other`: a significant paired gap of at least
    NEAR_TIE_WP in win probability."""
    if other.mean_q is None:
        return True
    gap = best.mean_q - other.mean_q
    se, _ = _paired(per_det, best.label, other.label)
    return se is not None and gap > Z_SIG * se and wp_loss(gap) >= NEAR_TIE_WP


def grade(options: list[OptionStat], per_det: list[dict[str, float]], members: list[str],
          offered: Iterable[str] | None = None) -> dict:
    """Grade a human answer given as the legal labels it may be (one label, or a set), against the
    best option among `offered` (the options the human really had, e.g. what Arena offered; None:
    every searched option). An engine-only option (a stand-in card's "Play Plains") is no yardstick.

    Returns {classification, reason, best (the yardstick), label (the best member), rank, regret,
    regret_best_of_set, regret_mean_of_set, wp_loss, se_paired, n_paired, pooled_sd, significant, robust, near_tie
    (labels tied with the best), member_ranks, visits_per_det (the human option's visits per
    determinization), thin (fewer than MIN_VISITS: its Q is biased low, see the module docstring)}.
    Regrets are in Q units."""
    rated = [o for o in options if o.mean_q is not None]
    by = {o.label: o for o in rated}
    out: dict[str, Any] = {"near_tie": []}
    if not rated:
        return dict(out, classification="ungraded", reason="no option was searched")
    offered = None if offered is None else set(offered)
    rated = [o for o in rated if offered is None or o.label in offered] or rated
    best = rated[0]
    out["best"] = best.label
    out["near_tie"] = [o.label for o in rated if o is best or not separable(per_det, best, o)]
    hits = [by[m] for m in dict.fromkeys(members) if m in by]
    if not hits:
        return dict(out, classification="ungraded", reason="the human's answer is not among the searched options")
    h = max(hits, key=lambda o: (o.mean_q, o.visit_share))
    regret = max(best.mean_q - h.mean_q, 0.0)
    mean_member = sum(o.mean_q for o in hits) / len(hits)
    se, n = _paired(per_det, best.label, h.label)
    pooled = math.sqrt((best.sd_q ** 2 + h.sd_q ** 2) / 2)
    vpd = h.n / max(len(per_det), 1)
    out.update(label=h.label, rank=h.rank, member_ranks={o.label: o.rank for o in hits},
               visits_per_det=vpd, thin=bool(h is not best and vpd < MIN_VISITS),
               q_human=h.mean_q, q_best=best.mean_q, regret=regret, regret_best_of_set=regret,
               regret_mean_of_set=max(best.mean_q - mean_member, 0.0), wp_loss=wp_loss(regret),
               wp_loss_mean_of_set=wp_loss(max(best.mean_q - mean_member, 0.0)),
               se_paired=se, n_paired=n, pooled_sd=pooled,
               significant=bool(h is not best and se is not None and regret > Z_SIG * se),
               robust=bool(h is not best and regret > ROBUST_SD * pooled))
    wp = out["wp_loss"]
    if h is best:
        c, why = "best", None
    elif se is None:
        c, why = "fine", "noise unknown (fewer than 2 determinizations searched both options)"
    elif h.label in out["near_tie"]:
        c, why = "fine", "near tie with the best option"
    elif not out["significant"]:
        c, why = "fine", "within search noise"
    elif wp < FINE_WP:
        c, why = "fine", f"loss below {FINE_WP:.0%}"
    elif wp >= BLUNDER_WP and out["robust"]:
        c, why = "blunder", None
    elif wp >= MISTAKE_WP and out["robust"]:
        c, why = "mistake", None
    else:
        c = "inaccuracy"
        why = None if wp < MISTAKE_WP else f"loss {wp:.1%} but not robust across the hidden cards"
    out.update(classification=c, reason=why)
    return out


# ------------------------------------------------------------------------------------------------
# determinization
# ------------------------------------------------------------------------------------------------

_MODELS: dict = {}


def opponent_model(exclude_drafts: Iterable[str] = ()):
    """`belief.OpponentModel.load(retention=True)`, cached, optionally without some drafts' decks
    (`reconstruct.holdout_drafts`: a mirrored pair's pool holds the opponent's real deck). The
    retention table is the cached full one: it is aggregate in-hand rates per (card kind, mana
    value, turn) over ~6,000 games, so two held-out drafts cannot move it, and re-learning it per row
    would cost ~40 s."""
    from draftzero.gameplay import belief as bl
    base = _MODELS.get(())
    if base is None:
        base = _MODELS[()] = bl.OpponentModel.load(retention=True)
    key = tuple(sorted(set(exclude_drafts or ())))
    if not key:
        return base
    m = _MODELS.get(key)
    if m is None:
        held = [k for k in _MODELS if isinstance(k, tuple) and k]
        if len(held) > 16:                                      # keep a few recent holdouts only
            for k in held[:8]:
                del _MODELS[k]
        m = _MODELS[key] = bl.OpponentModel(base.pool.without_drafts(key), base.meta, alpha=base.alpha,
                                            retention=base.retention)
    return m


def _pool_cards() -> set[str]:
    """The card names of the belief model's deck pool (cached; builds the pool cache once)."""
    if "cards" not in _MODELS:
        from draftzero.gameplay.belief import DeckPool
        _MODELS["cards"] = set(DeckPool.load().cards)
    return _MODELS["cards"]


def held_out(model, exclude_drafts: Iterable[str]):
    """A caller's deck model without these drafts' decks (same card axis, alpha and retention)."""
    from draftzero.gameplay import belief as bl
    ex = sorted(set(exclude_drafts))
    if not ex or not isinstance(model, bl.OpponentModel):
        return model
    return bl.OpponentModel(model.pool.without_drafts(ex), model.meta, alpha=model.alpha, retention=model.retention)


def belief_coverage(spec: StateSpec, cards: set[str]) -> tuple[float, int]:
    """(share of the non-basic cards known in this game that the belief model's pool knows, how
    many were checked): A's decklist and B's known zones. Low for a non-FDN game (a Cube log)."""
    from draftzero.gameplay.belief import zone_cards
    basics = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}
    known = {c for c in spec.players["A"].decklist} | set(zone_cards(spec, "B"))
    known = {c for c in known if c not in basics and not c.startswith("Snow-Covered ")}
    if not known:
        return 1.0, 0
    return sum(c in cards for c in known) / len(known), len(known)


def plan_determinization(spec: StateSpec, *, model=None, k: int = K_DEFAULT, seed: int = 0, hindsight: bool = False,
                         exclude_drafts: Iterable[str] | None = None, opp_colors: str | None = None,
                         seen=None, belief: bool = True) -> tuple[list[StateSpec] | None, dict]:
    """(K determinized specs to send as options.specs, or None for the bridge's own resampling;
    a record of what was done). See the module docstring for the rules."""
    B = spec.players["B"]
    info: dict[str, Any] = {"k": k, "seed": seed}
    complete = not spec.is_partial() and B.decklistSource != "placeholder"
    if hindsight:
        if complete and B.decklistSource == "exact":
            info.update(method="hindsight", resample=[],
                        note="B's real deck and hand (mirrored pair): evaluation only, not a verdict")
            return None, info
        info["hindsight_unavailable"] = "no exact opponent deck and hand in the spec"
    if B.decklistSource == "exact" and (hindsight or spec.provenance.source in LOG_SOURCES):
        # an exact B list (reconstruct.exact_spec of a mirrored pair, e.g. with an unknown card in
        # B's hand) stays B's list below, and its known hand stays: the verdict uses hindsight. In
        # a logged game the player never knows the opponent's exact list.
        info["hindsight_partial"] = "B's real decklist (and any known hand cards) kept; the rest sampled"
    if complete:
        info.update(method="bridge_resample", resample=["B"],
                    note="complete spec: B's hand re-drawn from B's library as built per determinization")
        return None, info
    why = None
    if not belief:
        why = "belief model disabled"
    elif model is None:
        try:
            cards = _pool_cards()
        except (OSError, ValueError, KeyError) as e:
            cards, why = None, f"no belief model here ({type(e).__name__})"
        if cards is not None:
            cov, n = belief_coverage(spec, cards)
            info["belief_coverage"] = round(cov, 3)
            if cov < BELIEF_MIN_COVERAGE:
                why = (f"belief model not applicable: {cov:.0%} of the {n} known non-basic cards are in its "
                       f"FDN deck pool")
            else:
                try:
                    model = opponent_model(exclude_drafts or ())
                except (OSError, ValueError, KeyError) as e:
                    why = f"no belief model here ({type(e).__name__})"
    else:
        cov, n = belief_coverage(spec, set(model.cards))
        info["belief_coverage"] = round(cov, 3)
        if cov < BELIEF_MIN_COVERAGE:
            why = f"belief model not applicable: {cov:.0%} of the {n} known non-basic cards are in its pool"
        elif exclude_drafts:
            model = held_out(model, exclude_drafts)
    if why is not None:
        info.update(method="bridge_resample", resample=["B"], fallback=why,
                    note=("B's hand re-drawn from B's library as built, which is a "
                          f"{B.decklistSource} list" + (": mostly basics and the cards B has shown"
                                                        if B.decklistSource == "placeholder" else "")))
        return None, info
    from draftzero.gameplay.belief import determinize
    specs = determinize(spec, model, k, seed=seed, opp_colors=opp_colors, seen=seen)
    info.update(method="belief", resample=[], model=getattr(model, "name", "deck"),
                retention=getattr(model, "retention", None) is not None,
                opp_colors=opp_colors or "seen-card colours", held_out_drafts=len(set(exclude_drafts or ())),
                note=("exact B decklist kept; only B's hand sampled" if B.decklistSource == "exact"
                      else "B's decklist and hand sampled"))
    return specs, info


# ------------------------------------------------------------------------------------------------
# state summaries
# ------------------------------------------------------------------------------------------------

def _perm_text(p: dict, logged: dict | None = None) -> str:
    x = p.get("x") or {}
    lid = p.get("id") or ""
    logged = logged or {}
    name = (logged.get(lid) or logged.get(lid.split("#")[0]) or x.get("name") or p.get("name") or p.get("token")
            or (p.get("tokenClass") or "?").rsplit(".", 1)[-1])
    bits = []
    if x.get("power") is not None:
        bits.append(f"{x['power']}/{x.get('toughness')}")
    for c, v in (p.get("counters") or {}).items():
        if v:
            bits.append(f"{v} {c.lower()}")
    if p.get("damage"):
        bits.append(f"{p['damage']} dmg")
    if p.get("sick"):
        bits.append("sick")
    if p.get("attachTo"):
        bits.append("attached")
    return name + (f" ({', '.join(bits)})" if bits else "")


def _battlefield_text(perms: list[dict], logged: dict | None = None) -> str:
    """Creatures first (the dump gives them P/T), then every other permanent; [T] = tapped."""
    creatures, other = Counter(), Counter()
    for p in perms:
        t = _perm_text(p, logged) + (" [T]" if p.get("tapped") else "")
        ((creatures if (p.get("x") or {}).get("power") is not None else other))[t] += p.get("count", 1)
    fmt = lambda c: ", ".join(f"{k}{f' x{v}' if v > 1 else ''}" for k, v in sorted(c.items()))  # noqa: E731
    return "; ".join(x for x in (fmt(creatures), fmt(other)) if x) or "-"


def state_summary(spec: StateSpec, decision_state: dict | None = None, substitutions: dict | None = None) -> dict:
    """A compact view of the position the decision is taken in: the bridge's dump at the decision
    when there is one (A's hand after the draw, the engine's P/T), else the spec. B's hand is a count
    plus the cards known from the log: the dump's B hand is a determinization, not the truth.
    `substitutions` (the bridge's original -> stand-in): the dump names stand-ins ("Plains"), so
    substituted cards are shown by their logged name with a '*' (A's hand then comes from the spec,
    which for a logged decision is the hand at that decision)."""
    s = decision_state or spec.to_dict()
    P = s.get("players") or {}
    known_b = list(spec.players["B"].hand)
    b_hand = P.get("B", {}).get("hand") or []
    b_count = len(b_hand) + (P.get("B", {}).get("handUnknown") or 0)
    subs = substitutions or {}
    star = lambda c: c + "*" if c in subs else c                                        # noqa: E731
    logged = {seat: {p.id: star(p.name or p.token or "") for p in spec.players[seat].battlefield
                     if p.id and (p.name or p.token) in subs} for seat in ("A", "B")}
    hand_a = P.get("A", {}).get("hand") or []
    if set(spec.players["A"].hand) & set(subs):
        hand_a = [star(c) for c in spec.players["A"].hand]
    stack = [f"{x.get('controller')}: {x.get('card')}" for x in s.get("stack") or []]
    if {x.card for x in spec.stack} & set(subs):    # a stand-in land is dropped from the stack
        stack = [f"{x.controller}: {star(x.card)}" for x in spec.stack]
    return {
        "turn": s.get("turn"), "active": s.get("activePlayer"), "step": s.get("step"),
        "life": {k: P.get(k, {}).get("life", 20) for k in ("A", "B")},
        "hand_A": sorted(hand_a),
        "hand_B": f"{b_count} card(s)" + (f", known: {', '.join(sorted(map(star, known_b)))}" if known_b
                                          else ", all hidden"),
        "battlefield_A": _battlefield_text(P.get("A", {}).get("battlefield") or [], logged["A"]),
        "battlefield_B": _battlefield_text(P.get("B", {}).get("battlefield") or [], logged["B"]),
        "graveyards": {k: len(P.get(k, {}).get("graveyard") or []) for k in ("A", "B")},
        "library": {k: P.get(k, {}).get("librarySize") for k in ("A", "B")},
        "stack": stack,
        "attackers": [a.get("attacker") for a in s.get("attackers") or []],
    }


def alias_names(dump: dict | None) -> dict[str, str]:
    """'<seat>:<id>' -> the engine's name of that permanent (tokens: 'Goblin Token'), from a build dump."""
    out = {}
    for seat, p in ((dump or {}).get("players") or {}).items():
        for perm in p.get("battlefield") or []:
            if perm.get("id"):
                out[f"{seat}:{perm['id']}"] = (perm.get("x") or {}).get("name") or perm.get("name") or ""
    return out


def fidelity(spec: StateSpec) -> tuple[dict, bool]:
    """({tier, flags}, low_fidelity)."""
    pv = spec.provenance
    risky = [f for f in pv.flags if f.split(":")[0] in RISKY_FLAGS]
    low = pv.tier in ("T2", "T3") or bool(risky)
    return {"tier": pv.tier, "risky_flags": risky, "flags": list(pv.flags)}, low


_SUB_ZONES = re.compile(r"^substitute: '.*' -> '.*'.*: \{(.*)\}$")


def visible_substitutions(warnings: list[str]) -> list[str]:
    """The bridge's substitution warnings that touch the visible position (a hand, graveyard,
    exile, known library top, the battlefield or the stack, or anything dropped), as opposed to
    stand-ins only among the decklist's unseen library cards."""
    out = []
    for w in warnings:
        if not w.startswith("substitute"):
            continue
        m = _SUB_ZONES.match(w)
        if m is None or {z.split("=")[0].strip() for z in m.group(1).split(",")} - {"decklist"}:
            out.append(w)
    return out


def display(label: str) -> str:
    """A label without XMage's reminder text, for reports."""
    return _REMINDER.sub("", label or "")


def is_combat(decision: dict) -> bool:
    t, text = decision.get("type"), decision.get("text") or ""
    return (t == "CHOOSE_USE" and text.startswith("attack with: ")) or \
        (t == "CHOOSE_TARGET" and text.startswith("choose which creature to block for "))


def question_subject(decision: dict) -> str | None:
    """The creature an attack or block question is about ('attack with: X?', 'choose which creature
    to block for X:Choose a target:...')."""
    text = decision.get("text") or ""
    if text.startswith("attack with: "):
        return text[len("attack with: "):].rstrip("?")
    if text.startswith("choose which creature to block for "):
        return text[len("choose which creature to block for "):].split(":Choose a target:")[0]
    return None


# ------------------------------------------------------------------------------------------------
# one decision
# ------------------------------------------------------------------------------------------------

def _bridge_call(bridge, op: str, spec, **options) -> dict:
    return bridge.request(op, spec, **{k: v for k, v in options.items() if v is not None})


def coach_decision(spec, bridge, *, model=None, k: int = K_DEFAULT, budget: int = BUDGET_DEFAULT, seed: int = 0,
                   evaluator="offline", priors=None, human=None, hindsight: bool = False,
                   exclude_drafts: Iterable[str] | None = None, opp_colors: str | None = None, seen=None,
                   options: dict | None = None, substitute: dict | None = None,
                   combat_budget: int | None = COMBAT_BUDGET, verify_budget: int | None = VERIFY_BUDGET,
                   expect: Callable[[dict], str | None] | None = None,
                   offered: Callable[[dict], Iterable[str] | None] | None = None, belief: bool = True, ref: str | None = None, source: str | None = None,
                   timeout_sec: float | None = None) -> Verdict:
    """Coach the decision a spec leads to (see the module docstring).

    spec          StateSpec, dict or path. Its labels["bridge"] (decisionPlayer, decideFrom) are
                  the request options unless `options` is given; without either, A decides from
                  its next main phase (`bridge.turn_start_options`).
    bridge        a `Bridge` or `BridgePool` (anything with .request(op, spec, **options)).
    human         a label ("Play Plains", "yes"), a set of labels (set-valued), a
                  resolver(decision, alias_names) -> HumanAction | label | labels | None, or None.
    evaluator     "offline", "remote[:PORT]" or `remote(port)`; priors {priority, target, binary,
                  opponent, temperature} or a list of names (remote only).
    hindsight     coach B's real deck and hand when the spec has them (mirrored pairs): evaluation only.
    exclude_drafts, opp_colors, seen, model, belief: the belief determinization (plan_determinization).
    combat_budget attack/block questions are searched with at least this budget (None: keep
                  `budget` and flag combat_low_budget when it is below COMBAT_BUDGET offline).
    verify_budget offline, an inaccuracy or worse whose human option got fewer than MIN_VISITS
                  visits per determinization is searched again with this budget (None or 0: not;
                  the grade is then flagged thin_search). Costs ~K x budget / 450 s per such decision.
    expect        expect(decision) -> None if the bridge reached the logged decision, else why not.
    offered       offered(decision) -> the legal labels the human really had (None: all of them);
                  the regret is measured against the best of those (grade).
    """
    from draftzero.gameplay import bridge as br
    t0 = time.monotonic()
    if not isinstance(spec, StateSpec):
        spec = StateSpec.from_dict(br.spec_dict(spec))
    ev = _evaluator(evaluator)
    offline = ev.get("type") == "offline"
    if isinstance(priors, (list, tuple, set)):
        priors = {p: True for p in priors}
    opts = dict(options if options is not None else (spec.labels.get("bridge") or br.turn_start_options(spec, "A")))
    v = Verdict(ref=ref or spec.provenance.ref, source=source or spec.provenance.source)
    v.fidelity, low = fidelity(spec)
    if low:
        v.flags.append("low_fidelity")
    v.settings = {"k": k, "budget": budget, "evaluator": ev, "priors": priors or None, "seed": seed,
                  "request_options": opts, "substitute": substitute}

    def done(cls: str, reason: str | None = None) -> Verdict:
        v.classification, v.reason = cls, reason
        v.seconds = round(time.monotonic() - t0, 2)
        return v

    # 1. build once (a few ms): the question, its legal options, the position, alias -> name
    try:
        b = _bridge_call(bridge, "build", spec.to_dict(), seed=seed, dumpDecisionState=True, substitute=substitute,
                         **opts)
    except Exception as e:                                            # noqa: BLE001 - one bad spec must not stop a run
        return done("error", _first_line(e))
    v.warnings += b.get("warnings") or []
    if b.get("substitutions"):
        v.context["substitutions"] = b["substitutions"]
        if visible_substitutions(b.get("warnings") or []):
            v.flags.append("substituted")
            if "low_fidelity" not in v.flags:
                v.flags.append("low_fidelity")
        else:
            v.flags.append("substituted_library")    # stand-ins only among the unseen library cards
    d = b.get("decision")
    v.state = state_summary(spec, b.get("decisionState"), b.get("substitutions"))
    if not d:
        return done("ungraded", f"no decision reached: {b.get('noDecision') or 'unknown'}")
    where = d.get("where") or {}
    v.decision = {"player": d.get("player"), "type": d.get("type"), "text": d.get("text"),
                  "turn": where.get("turn"), "phase": where.get("phase"), "step": where.get("step"),
                  "passedBefore": where.get("passedBefore"), "stack": where.get("stack"),
                  "legal": [o["label"] for o in d.get("legal") or []]}
    if len(v.decision["legal"]) < 2:
        v.flags.append("trivial")
        return done("trivial", "fewer than 2 legal options")
    mismatch = expect(v.decision) if expect else None
    if mismatch:
        v.flags.append("decision_mismatch")
        return done("ungraded", mismatch)
    names = alias_names(b.get("dump"))
    avail = offered(v.decision) if offered else None
    if avail is not None:
        keep = set(avail)
        avail = [lab for lab in v.decision["legal"] if lab in keep]
    if avail is not None and len(avail) < len(v.decision["legal"]):
        v.context["not_offered"] = [lab for lab in v.decision["legal"] if lab not in avail]
    ha = _as_human(human, v.decision, names)
    members, unmatched = [], []
    if ha is not None:
        for a in ha.labels:
            lab = match_label(a, v.decision["legal"], v.decision["type"])
            (members if lab else unmatched).append(lab or a)
        v.human = {"actions": list(ha.labels), "set_valued": ha.set_valued, "source": ha.source,
                   "matched": list(dict.fromkeys(members)), "unmatched": unmatched, "note": ha.note}
        if ha.set_valued:
            v.flags.append("set_valued_human")
        # an answer that cannot be graded here is not worth a search (a few seconds each)
        if not ha.labels:
            return done("ungraded", ha.reason or "no human answer")
        if not members:
            v.flags.append("human_unmatched")
            return done("ungraded", "the human's answer is not a legal option here: " + ", ".join(unmatched))

    if avail is not None:
        avail += [m for m in members if m not in avail]      # what the human chose was on offer
    # 2. the search budget: combat questions need more simulations offline
    run_budget = budget
    if is_combat(v.decision) and offline:
        if combat_budget and budget < combat_budget:
            run_budget = combat_budget
            v.flags.append("combat_budget_raised")
        elif budget < COMBAT_BUDGET:
            v.flags.append("combat_low_budget")
    v.settings["budget"] = run_budget

    # 3. determinizations and the search
    specs, info = plan_determinization(spec, model=model, k=k, seed=seed, hindsight=hindsight,
                                       exclude_drafts=exclude_drafts, opp_colors=opp_colors, seen=seen, belief=belief)
    v.determinization = info
    if info["method"] == "hindsight" or info.get("hindsight_partial"):
        v.flags.append("hindsight")
    if info.get("fallback"):
        v.flags.append("belief_fallback")
    common = dict(seed=seed, evaluator=ev, priors=priors or None, substitute=substitute, timeoutSec=timeout_sec,
                  **opts)
    spec_dicts = [s.to_dict() for s in specs] if specs is not None else None

    def search(n_sims: int) -> dict:
        if spec_dicts is not None:
            return _bridge_call(bridge, "coach", None, specs=spec_dicts, budget=n_sims, **common)
        return _bridge_call(bridge, "coach", spec.to_dict(), determinizations=k, resample=info.get("resample"),
                            budget=n_sims, **common)

    def digest(r: dict) -> str | None:
        """Fill the verdict from a coach response; returns why it cannot be graded, if so."""
        v.warnings = list(dict.fromkeys(v.warnings + (r.get("warnings") or [])))
        per_det[:] = per_determinization_q(r)
        v.options = options_from(r, per_det)
        v.best = r.get("best")
        v.determinization["hands_B"] = [(x.get("hands") or x.get("sampledHands") or {}).get("B")
                                        for x in r.get("determinizations") or []]
        v.flags[:] = [f for f in v.flags if f not in ("unstable", "inconsistent")]
        rated = [o for o in v.options if o.mean_q is not None]
        if rated:
            v.value = {"best_q": rated[0].mean_q, "best_win_prob": rated[0].win_prob,
                       "root_value_mean": _mean([x.get("value") for x in r.get("determinizations") or []]),
                       "sims_per_s": (r.get("timing_ms") or {}).get("simsPerSec")}
            if rated[0].argmax_share < UNSTABLE_AGREEMENT:
                v.flags.append("unstable")
        v.noise = {"pooled_sd_all": _rms([o.sd_q for o in rated]), "k": len(per_det)}
        if r.get("consistent") is False:
            v.flags.append("inconsistent")
            return "the determinizations reached different questions"
        got = r.get("decision") or {}
        if got and (got.get("type"), got.get("text")) != (v.decision["type"], v.decision["text"]):
            # the human's answer was matched on the build's question: grading another one is wrong
            v.flags.append("inconsistent")
            return f"the search reached another question ({got.get('type')}) than the build"
        return None

    per_det: list[dict[str, float]] = []
    try:
        r = search(run_budget)
    except Exception as e:                                            # noqa: BLE001
        return done("error", _first_line(e))
    why = digest(r)
    if why:
        return done("ungraded", why)
    g = grade(v.options, per_det, members, avail)
    # 4. a fault on an option the search hardly visited is mostly the search's bias (module
    # docstring): search again, deeper, with the same determinizations
    if (ha is not None and offline and verify_budget and verify_budget > run_budget
            and g["classification"] in FAULTS and g.get("thin")):
        first = {"budget": run_budget, "classification": g["classification"], "regret": g.get("regret"),
                 "wp_loss": g.get("wp_loss"), "visits_per_det": g.get("visits_per_det"), "best": g.get("best")}
        try:
            r = search(verify_budget)
        except Exception as e:                                        # noqa: BLE001 - keep the first grade
            v.warnings.append(f"verification at {verify_budget} simulations failed: {_first_line(e)}")
        else:
            v.settings["budget"] = verify_budget
            v.flags.append("verified")
            why = digest(r)
            if why:
                return done("ungraded", why)
            g = grade(v.options, per_det, members, avail)
        v.noise["first_pass"] = first
    if len(g.get("near_tie") or []) > 1:
        v.flags.append("near_tie")
    v.noise.update(near_tie=g.get("near_tie"))
    if g.get("best") and g["best"] != v.best:
        # the engine's favourite was not on offer to the human (Arena did not list it)
        v.value["engine_best"], v.best = v.best, g["best"]
        v.flags.append("best_not_offered")
    if ha is None:
        return done("ungraded", "no human answer given")
    v.human.update({k2: g.get(k2) for k2 in ("label", "rank", "member_ranks", "q_human", "q_best", "regret",
                                              "regret_best_of_set", "regret_mean_of_set", "wp_loss",
                                              "wp_loss_mean_of_set", "visits_per_det")})
    v.noise.update({k2: g.get(k2) for k2 in ("se_paired", "n_paired", "pooled_sd", "significant", "robust")})
    if g["classification"] in FAULTS and g.get("thin"):
        v.flags.append("thin_search")
    return done(g["classification"], g.get("reason"))


def _first_line(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}"[:300]


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _rms(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else None


# ------------------------------------------------------------------------------------------------
# 17lands
# ------------------------------------------------------------------------------------------------

def human_17lands(labels: dict) -> Callable[[dict, dict], HumanAction]:
    """Resolver for a 17lands turn: the turn's actions as a set at the first main-phase decision
    (critique WP4 label policy: S = the turn's lands, casts and activations, plus Pass when a
    post-combat play is possible: the user attacked, or did nothing); the recorded attack for an
    'attack with: X?' question; the recorded, exactly paired block for a block question."""
    def resolve(decision: dict, names: dict) -> HumanAction:
        t = decision.get("type")
        if t == "PRIORITY" and not ("lands" in labels or "casts" in labels):
            # the labels of the state after a user turn (after_turn_label): off-turn plays have no
            # timing in 17lands, so no priority window there has a known answer (critique WP4)
            return HumanAction(reason="17lands has no timing for plays in the opponent's turn")
        if t == "PRIORITY":
            acts = [x["key"] for x in labels.get("lands", []) if x.get("key")]
            acts += [x["key"] for x in labels.get("casts", []) if x.get("key")]
            acts += [x["key"] for x in labels.get("activations", []) if x.get("key")]
            attacked = any(labels.get("attacks", {}).values())
            note = None
            if not acts:
                acts, note = ["Pass"], "no land, cast or activation this turn"
            elif attacked:
                acts.append("Pass")
                note = "Pass included: the user attacked, so some of these may have come after combat"
            return HumanAction(list(dict.fromkeys(acts)), len(set(acts)) > 1, "17lands turn actions (unordered)",
                               note=note)
        subject = question_subject(decision)
        if t == "CHOOSE_USE" and subject is not None:
            # "new:<name>": a creature that entered this turn and attacked (not in the spec)
            ans = {bool(v) for a, v in labels.get("attacks", {}).items()
                   if (a[4:] if a.startswith("new:") else names.get(a)) == subject}
            if len(ans) == 1:
                return HumanAction(["yes" if ans.pop() else "no"], False, "17lands attacks")
            return HumanAction(reason=f"attack of '{subject}' not recorded unambiguously ({len(ans)} answers)")
        if t == "CHOOSE_TARGET" and subject is not None:
            rows = [b for b in labels.get("blocks", []) if names.get(b[0]) == subject]
            if not rows:
                return HumanAction(reason=f"no block label for '{subject}'")
            if not all(b[2] for b in rows):
                # exact is False for a guessed pairing (block_pairing ambiguous, ...) and for "did not
                # block" by a creature that left the battlefield outside combat that turn
                return HumanAction(reason=f"block of '{subject}' not exact (turn pairing "
                                          f"{labels.get('block_pairing')}; a guess, or it left the battlefield "
                                          f"outside combat that turn)")
            ans = {names.get(b[1], "?") if b[1] else "Stop Choosing" for b in rows}
            if len(ans) == 1:
                return HumanAction([ans.pop()], False, "17lands blocks")
            return HumanAction(reason=f"copies of '{subject}' blocked differently")
        return HumanAction(reason=f"no 17lands label for a {t} decision")
    return resolve


def expect_17lands(spec: StateSpec) -> Callable[[dict], str | None]:
    """The labels describe the turn's first main-phase decision (or the attack question when A has
    nothing to do before combat), or the block decision: anything else is another decision."""
    df = (spec.labels.get("bridge") or {}).get("decideFrom") or {}

    def check(decision: dict) -> str | None:
        if df.get("turn") is not None and decision.get("turn") != df["turn"]:
            return f"the bridge reached turn {decision.get('turn')}, the labels describe turn {df['turn']}"
        step, t = decision.get("step"), decision.get("type")
        ok = ((t == "PRIORITY" and step == df.get("step")) or
              (t == "CHOOSE_USE" and (decision.get("text") or "").startswith("attack with: ")) or
              (df.get("step") == "DECLARE_BLOCKERS" and step == "DECLARE_BLOCKERS" and t == "CHOOSE_TARGET"))
        return None if ok else f"the bridge reached a {t} decision at {step}, not the labelled one"
    return check


def coach_17lands(row: int, turn: int, *, bridge, game=None, partner=None, pairs: dict | None = None,
                  path=None, blocks: bool = False, entry: str | None = None, hindsight: bool = False,
                  model=None, ids=None, **kw) -> Verdict:
    """Coach user turn `turn` of 17lands replay row `row`: the turn's first main-phase decision
    (blocks=True: the block decision in the opponent's next turn, from its declared attackers).

    The spec comes from reconstruct (entry eot_rollover, or declare_attackers for blocks) with the
    turn's labels. The belief model leaves out the row's own draft and, when the row is half of a
    mirrored pair (`pairs`: row -> partner row, default data/gameplay/pairs_*.jsonl when present),
    the partner's draft too: the pool holds the opponent's real deck otherwise. hindsight=True
    coaches the true position of a mirrored pair (reconstruct.exact_spec): evaluation only.
    kw goes to coach_decision (k, budget, seed, evaluator, priors, ...)."""
    from draftzero.gameplay import reconstruct as rc, replay
    from draftzero.gameplay.ids import Ids
    notes = []
    if partner is None and pairs is None:
        pairs = _load_partner_map()
        if pairs is None:
            notes.append("pairs file not found: mirrored-pair status unknown")
    prow = (pairs or {}).get(row) if partner is None else partner.row_index
    if game is None or (prow is not None and partner is None):
        got = replay.read_games([row] + ([prow] if prow is not None else []), path)
        game = game or got[row]
        partner = partner or (got.get(prow) if prow is not None else None)
    ids = ids or Ids.load(game.meta.get("expansion") or "FDN")
    after = blocks
    ent = entry or ("declare_attackers" if blocks else "eot_rollover")
    if hindsight and partner is not None:
        spec = rc.exact_spec(game, partner, turn, ent, ids=ids, labels=True, after=after)
    else:
        if hindsight:
            notes.append("hindsight requested but the row has no mirrored pair")
        make = rc.state_after_user_turn if after else rc.state_at_user_turn
        spec = make(game, turn, ent, ids=ids, labels=True)
    slot = game.user_slot(turn) if after else game.prev_slot(turn)
    seen = Counter(rc.analyze(game, ids).states[slot.seq].revealed) if slot is not None else Counter()
    v = coach_decision(spec, bridge, model=model, human=human_17lands(spec.labels), hindsight=hindsight,
                       exclude_drafts=rc.holdout_drafts(game, partner), seen=seen, expect=expect_17lands(spec),
                       source="17lands", **kw)
    if pairs is None and partner is None and v.determinization.get("method") == "belief":
        # the row may be half of a mirrored pair whose partner draft (the opponent's real deck)
        # stayed in the belief model's pool
        v.flags.append("pair_status_unknown")
    v.context.update(row=row, user_turn=turn, decision="blocks" if blocks else "turn start", entry=ent,
                     mirrored_pair=partner is not None, notes=notes,
                     rank=game.meta.get("rank"), win_rate_bucket=game.meta.get("user_game_win_rate_bucket"),
                     won=game.meta.get("won"))
    return v


_PARTNERS: dict = {}


def _load_partner_map() -> dict | None:
    """row -> partner row from data/gameplay/pairs_FDN_PremierDraft.jsonl (memory only), or None."""
    from draftzero.gameplay import pairs as pm
    if "map" not in _PARTNERS:
        try:
            _PARTNERS["map"] = pm.partner_map(pm.load_pairs())
        except OSError:
            _PARTNERS["map"] = None
    return _PARTNERS["map"]


WR_BANDS = ((0.0, 0.50), (0.50, 0.54), (0.54, 0.58), (0.58, 1.01))


def validate_17lands(*, bridge, every: int = 2003, start: int = 0, limit: int = 200, turns=range(3, 8), seed: int = 0,
                     path=None, min_games: int = 100, progress: Callable | None = None, **kw) -> dict:
    """E5 pilot (web_literature.md §7.7 point 8): does the coach's regret fall with player skill?

    Coaches one random decision turn (in `turns`) of every `every`-th replay row until `limit`
    decisions, and groups the win-probability loss by the user's 17lands win-rate bucket (users with
    at least `min_games` games only: the bucket includes the game itself, which dominates it for
    users with few games; critique N3) and by rank. Rows that are half of a mirrored pair are skipped
    (their belief model would need the partner's draft held out, i.e. a second read of the file).
    Returns per-decision rows (row index, turn, skill fields, grade; no draft ids) and the groups,
    with Spearman's rho between the win-rate bucket and the loss."""
    import random
    from draftzero.gameplay import replay
    pairs = _load_partner_map()
    warnings = [] if pairs is not None else [
        "pairs file not found: mirrored-pair rows were not skipped, and their belief pool holds the opponent's "
        "real deck (hindsight)"]
    pairs = pairs or {}
    rng = random.Random(seed)
    rows, skipped = [], Counter()
    t0 = time.monotonic()
    for g in replay.iter_games(path, every=every, start=start):
        if len(rows) >= limit:
            break
        _revive(bridge)             # a request that timed out closed the worker: start a fresh one
        if g.row_index in pairs:
            skipped["mirrored_pair"] += 1
            continue
        if (g.meta.get("user_n_games_bucket") or 0) < min_games:
            skipped["few_games"] += 1
            continue
        cand = [n for n in g.decision_turns() if n in turns]
        if not cand:
            skipped["no_turn"] += 1
            continue
        n = rng.choice(cand)
        v = coach_17lands(g.row_index, n, bridge=bridge, game=g, pairs={}, path=path, **kw)
        h = v.human or {}
        rows.append({"row": g.row_index, "turn": n, "rank": g.meta.get("rank"),
                     "win_rate_bucket": g.meta.get("user_game_win_rate_bucket"),
                     "n_games_bucket": g.meta.get("user_n_games_bucket"), "won": g.meta.get("won"),
                     "classification": v.classification, "reason": v.reason, "wp_loss": h.get("wp_loss"),
                     "wp_loss_mean_of_set": h.get("wp_loss_mean_of_set"), "members": len(h.get("matched") or []),
                     "options": len(v.options), "flags": v.flags, "caution": v.caution, "tier": v.fidelity.get("tier"),
                     "first_pass": (v.noise.get("first_pass") or {}).get("classification"),
                     "budget": v.settings.get("budget"), "seconds": v.seconds})
        if progress:
            progress(rows[-1])
    graded = [r for r in rows if r["classification"] in GRADED]

    def group(key) -> dict:
        out: dict = {}
        for r in graded:
            out.setdefault(key(r), []).append(r)
        return {str(k): {"n": len(xs), "mean_wp_loss": _mean([x["wp_loss"] for x in xs]),
                         "mean_wp_loss_mean_of_set": _mean([x["wp_loss_mean_of_set"] for x in xs]),
                         "share_inaccuracy_or_worse": sum(x["classification"] in FAULTS for x in xs) / len(xs),
                         "share_faults_without_caution": sum(x["classification"] in FAULTS and not x["caution"]
                                                             for x in xs) / len(xs),
                         "mean_members": _mean([x["members"] for x in xs])}
                for k, xs in sorted(out.items(), key=lambda kv: str(kv[0]))}

    def band(r):
        w = r["win_rate_bucket"]
        return next((f"{a:.2f}-{b:.2f}" for a, b in WR_BANDS if w is not None and a <= w < b), "none")

    pts = [(r["win_rate_bucket"], r["wp_loss"]) for r in graded if r["win_rate_bucket"] is not None]
    rho = spearman([x for x, _ in pts], [y for _, y in pts]) if len(pts) > 3 else None
    ci = None
    if rho is not None and abs(rho) < 1:
        z, se = math.atanh(rho), 1 / math.sqrt(len(pts) - 3)
        ci = [math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)]
    return {"decisions": len(rows), "graded": len(graded), "skipped": dict(skipped), "warnings": warnings,
            "classes": dict(Counter(r["classification"] for r in rows)),
            "by_win_rate_band": group(band), "by_rank": group(lambda r: r["rank"]),
            "by_result": group(lambda r: "won" if r["won"] else "lost"),
            "spearman_win_rate_vs_loss": rho, "spearman_95ci": ci, "n_spearman": len(pts),
            "settings": {"every": every, "start": start, "limit": limit, "turns": list(turns), "seed": seed,
                         "min_games": min_games,
                         **{k: v for k, v in kw.items() if k in ("k", "budget", "evaluator", "combat_budget",
                                                                 "verify_budget")}},
            "seconds": round(time.monotonic() - t0, 1), "rows": rows}


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman's rank correlation (average ranks for ties)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for m in range(i, j + 1):
                r[order[m]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return sxy / (sx * sy) if sx and sy else None


# ------------------------------------------------------------------------------------------------
# Arena
# ------------------------------------------------------------------------------------------------

ARENA_KINDS = {"ActionsAvailableReq": "priority", "DeclareAttackersReq": "attack", "DeclareBlockersReq": "block"}
_MANA_TYPES = ("FloatMana", "Activate_Mana", "ActivateMana")


def arena_choices(labels: dict) -> list[dict]:
    """The Arena options a priority decision really offered: no mana abilities, and no spell or
    ability Arena could not pay for (a cost without an auto-tap solution), except the one chosen."""
    chosen = {json.dumps(c, sort_keys=True) for c in labels.get("chosen") or []}
    out = []
    for o in labels.get("options") or []:
        if o.get("arena_action_type") in _MANA_TYPES:
            continue
        if o.get("cost") and o.get("autotap") is False and json.dumps(o, sort_keys=True) not in chosen:
            continue
        out.append(o)
    return out


def arena_trivial(labels: dict) -> str | None:
    """Why an Arena decision is not worth coaching (None: coach it)."""
    kind = ARENA_KINDS.get(labels.get("decision_type"))
    if kind == "priority":
        n = len(arena_choices(labels))
        return None if n >= 2 else f"{n} real option(s) (mana abilities and unaffordable spells aside)"
    if kind == "attack":
        return None if labels.get("options") else "no creature could attack"
    if kind == "block":
        return None if any(o.get("can_block") for o in labels.get("options") or []) else "no creature could block"
    return "not a priority, attack or block decision"


def human_arena(labels: dict) -> Callable[[dict, dict], HumanAction]:
    """Resolver for an Arena decision: the chosen action's mz_key; for the bridge's first attack or
    block question, the human's answer for that creature (by engine name; copies must agree)."""
    kind = ARENA_KINDS.get(labels.get("decision_type"))

    def resolve(decision: dict, names: dict) -> HumanAction:
        chosen = labels.get("chosen")
        if chosen is None:
            return HumanAction(reason="the request was not answered")
        if kind == "priority":
            if not chosen:
                return HumanAction(reason="no action recorded")
            c = chosen[0]
            if c.get("arena_action_type") in _MANA_TYPES:
                return HumanAction(reason="the human activated a mana ability (not a MageZero action)")
            return HumanAction([c.get("mz_key") or c.get("name") or "?"], False, "arena chosen action")
        subject = question_subject(decision)
        if subject is None:
            return HumanAction(reason=f"the bridge asked a {decision.get('type')} question, not an {kind}")
        if kind == "attack":
            ans = {bool(v) for a, v in (labels.get("attack") or {}).items() if names.get(a) == subject}
            if len(ans) == 1:
                return HumanAction(["yes" if ans.pop() else "no"], False, "arena attack")
            return HumanAction(reason=f"'{subject}': {len(ans)} different answers among same-named creatures")
        if kind == "block":
            ans = set()
            for b, atk in (labels.get("block") or {}).items():
                if names.get(b) == subject:
                    ans.add(names.get(atk[0], "?") if atk else "Stop Choosing")
            if len(ans) == 1:
                return HumanAction([ans.pop()], False, "arena block")
            return HumanAction(reason=f"'{subject}': {len(ans)} different answers among same-named creatures")
        return HumanAction(reason="not coached")
    return resolve


def expect_arena(spec: StateSpec) -> Callable[[dict], str | None]:
    """The bridge must reach the logged decision: same type, same turn; a priority decision in its
    own window (the bridge moves on past windows where the decider has nothing to do)."""
    L = spec.labels
    kind = ARENA_KINDS.get(L.get("decision_type"))

    def check(d: dict) -> str | None:
        if L.get("mz_decision") and d.get("type") != L["mz_decision"]:
            return f"the bridge asked a {d.get('type')} question, the log a {L['mz_decision']} one"
        if d.get("turn") != spec.turn:
            return f"the bridge reached turn {d.get('turn')}, the log's decision is in turn {spec.turn}"
        if kind == "priority" and (d.get("step") != spec.step or d.get("passedBefore")):
            return (f"the bridge reached {d.get('step')} after {d.get('passedBefore')} passed window(s), the log's "
                    f"decision is at {spec.step}")
        if kind in ("attack", "block") and d.get("step") != ("DECLARE_ATTACKERS" if kind == "attack" else "DECLARE_BLOCKERS"):
            return f"the bridge reached {d.get('step')}"
        return None
    return check


def legal_set_agreement(labels: dict, legal: list[str]) -> dict:
    """Arena's affordable options vs XMage's legal labels for a priority decision (mana aside)."""
    from draftzero.gameplay.arena import same_action
    keys = [o.get("mz_key") for o in arena_choices(labels) if o.get("mz_key")]
    arena_only = [k for k in keys if not any(same_action(k, lab) for lab in legal)]
    xmage_only = [lab for lab in legal if not any(same_action(k, lab) for k in keys)]
    return {"arena": len(keys), "xmage": len(legal), "arena_only": arena_only, "xmage_only": xmage_only}


def arena_offered(labels: dict) -> Callable[[dict], list[str]]:
    """offered(decision) for coach_decision: the XMage labels of a priority decision that Arena offered
    too (legal_set_agreement). XMage-only options are mostly stand-ins of substituted cards (a
    'Play Plains' for a spell XMage lacks): the human never had them, so they are no yardstick."""
    def offered(decision: dict) -> list[str]:
        legal = decision.get("legal") or []
        extra = set(legal_set_agreement(labels, legal)["xmage_only"])
        return [lab for lab in legal if lab not in extra]
    return offered


@dataclass
class ArenaCoaching:
    verdicts: list[Verdict]
    skipped: Counter                         # decisions of other kinds, by kind
    logs: list[dict]                         # per log: file name, detailed-logs flag, client version
    games: dict                              # game number -> {turns, result, decisions}
    identifiers: Any = None                  # arena.Identifiers: memory only, for the privacy check
    allowed_words: set = field(default_factory=set)
    seconds: float = 0.0


def coach_arena(logs=None, *, bridge, k: int = K_DEFAULT, budget: int = BUDGET_DEFAULT, seed: int = 0,
                evaluator="offline", priors=None, substitute: dict | None = None, card_db=None, model=None,
                games: Iterable[int] | None = None, limit: int | None = None, progress: Callable | None = None,
                **kw) -> ArenaCoaching:
    """Coach every priority, attack and block decision the local player answered in these Arena
    logs (default: the local Player-prev.log and Player.log), with the options Arena offered, the
    human's answer and the bridge request options arena.py stored (labels['bridge']).

    Trivial decisions (fewer than two real options) are listed but not searched. `substitute`
    (bridge options.substitute, e.g. {"missing": "Plains", "missingToken": "GoblinToken"}) builds
    cards this XMage lacks as stand-ins; such decisions are flagged. Nothing is written here: see
    write_arena_report."""
    from draftzero.gameplay import arena as ar
    t0 = time.monotonic()
    paths = [Path(p).expanduser() for p in (logs or ar.default_logs())]
    if not paths:
        raise FileNotFoundError("no Arena log given and none found in the usual places")
    infos = [dict(ar.log_info(p), file=p.name) for p in paths]
    if card_db is None:
        card_db = ar.pick_card_db(next((i["grp_build"] for i in infos if i.get("grp_build")), None))
    names = ar.CardNames.load(card_db=card_db)
    idents = ar.Identifiers.from_logs(paths)
    parser = ar.parse_logs(paths, names)
    verdicts, skipped = [], Counter()
    want = set(games) if games is not None else None
    for gm in parser.games:
        for dec in gm.decisions:
            spec = dec.spec
            L = spec.labels
            kind = ARENA_KINDS.get(L.get("decision_type"))
            if kind is None:
                skipped[L.get("decision_type") or "?"] += 1
                continue
            if want is not None and L.get("game") not in want:
                continue
            if limit is not None and len(verdicts) >= limit:
                break
            ref = f"game{L.get('game')}:decision{L.get('decision')}"
            why = arena_trivial(L)
            if why is not None:
                v = Verdict(ref=ref, source="arena", classification="trivial", reason=why, flags=["trivial"])
                v.fidelity, _ = fidelity(spec)
            else:
                _revive(bridge)
                v = coach_decision(spec, bridge, model=model, k=k, budget=budget, seed=seed, evaluator=evaluator,
                                   priors=priors, human=human_arena(L), substitute=substitute,
                                   expect=expect_arena(spec), offered=arena_offered(L) if kind == "priority" else None,
                                   ref=ref, source="arena", **kw)
                if kind == "priority" and v.decision.get("legal"):
                    agree = legal_set_agreement(L, v.decision["legal"])
                    v.context["legal_set"] = agree
                    if agree["arena_only"] or agree["xmage_only"]:
                        v.flags.append("legal_set_differs")
                if kind == "attack" and len(L.get("options") or []) > 1:
                    v.flags.append("first_attack_question_only")
                if kind == "block" and len(L.get("options") or []) > 1:
                    v.flags.append("first_block_question_only")
            v.context.update(game=L.get("game"), decision=L.get("decision"), kind=kind,
                             arena_turn=(L.get("arena") or {}).get("turn"), latency_s=L.get("latency_s"))
            verdicts.append(v)
            if progress:
                progress(v)
    games_meta = {gm.index: {"turns": gm.state.turn.get("turnNumber"), "result": gm.result(),
                             "decisions": len(gm.decisions)} for gm in parser.games}
    allowed = {w for s in names.emitted for w in re.findall(r"\w+", s)}
    return ArenaCoaching(verdicts, skipped, [{k2: v2 for k2, v2 in i.items() if k2 != "grp_build"} for i in infos],
                         games_meta, idents, allowed, round(time.monotonic() - t0, 1))


def _revive(bridge) -> None:
    """A single Bridge whose worker died (a request timed out, the JVM crashed) is restarted, so one
    bad decision does not turn the rest of a run into errors (a BridgePool restarts its own)."""
    proc = getattr(bridge, "proc", False)
    if proc is not False and (proc is None or proc.poll() is not None) and hasattr(bridge, "start"):
        bridge.start()


def luck_track(verdicts: list[Verdict]) -> dict:
    """Per game: skill loss (sum of the human's regrets) and luck (the value swings between two
    consecutive graded decisions that the human's choice does not explain: V(next) - Q(human),
    which includes the draws, the opponent's play and every uncoached decision in between). Values
    are the evaluator's, from A's view."""
    out: dict = {}
    by_game: dict = {}
    for v in verdicts:
        if v.graded and v.human and v.human.get("q_human") is not None and v.context.get("game") is not None:
            by_game.setdefault(v.context["game"], []).append(v)
    for g, vs in by_game.items():
        if len(vs) < 2:
            continue
        swings = [b.human["q_best"] - a.human["q_human"] for a, b in zip(vs, vs[1:])]
        out[g] = {"graded": len(vs), "regret_sum": sum(v.human["regret"] for v in vs),
                  "luck_sum": sum(swings), "start_value": vs[0].human["q_best"], "end_value": vs[-1].human["q_best"],
                  "swings": swings}
    return out


# ------------------------------------------------------------------------------------------------
# reports
# ------------------------------------------------------------------------------------------------

CAVEATS = [
    "Offline evaluator: values come from MageZero's heuristic (tanh of life, hand size and board "
    "mana value difference / 15, uniform priors) after a short search. It has no notion of card "
    "quality, tokens or loyalty. A trained value network (remote evaluator on a GPU) should replace it "
    "before these grades mean much.",
    "Win probability = (1 + Q) / 2 assumes Q is an expected outcome on a win = +1 / loss = -1 scale. The "
    "heuristic is not calibrated and search values are discounted 0.99 per ply, so the percentages are nominal.",
    "Perfect-information search per determinization (PIMC): inside one sample the search sees every card, "
    "so it never values playing around a trick or bluffing. Grades are averages over K samples of the "
    "hidden cards, never the real opponent hand.",
    f"Thresholds (win-probability loss: fine < {FINE_WP:.0%}, inaccuracy, mistake >= {MISTAKE_WP:.0%}, blunder >= "
    f"{BLUNDER_WP:.0%}; a mistake or blunder must also exceed {ROBUST_SD:g} x the pooled sd of the two options and "
    f"{Z_SIG:g} x the paired standard error) are provisional: not yet validated against player skill.",
    "One decision per state: the bridge answers the first question of a position, so an attack or block "
    "grade covers the first creature the engine asks about, not the whole declaration.",
    "A few hundred simulations look only a few actions ahead: 'pass now and act later in the turn' (an "
    "instant-speed cycle or trick at end of turn) can be undervalued against acting at once.",
    f"An option's value is the mean over the lines searched below it, so an option the search hardly visited "
    f"(a land drop before the spell, a play it dismissed early) is undervalued and its regret overstated. "
    f"Faults on options with fewer than {MIN_VISITS} visits per sample are re-searched at {VERIFY_BUDGET:,} "
    f"simulations (flag verified); those still under it are flagged thin_search.",
]


def _pct(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}%"


def _q(x: float | None) -> str:
    return "-" if x is None else f"{x:+.3f}"


def render_verdict(v: Verdict, heading: str = "###") -> str:
    d = v.decision
    kind = v.context.get("kind") or (d.get("type") or "").lower()
    # the bridge's turn is the game's (both players' turns counted); 17lands counts the user's own
    ut = f" (user turn {v.context['user_turn']})" if v.context.get("user_turn") is not None else ""
    title = f"{heading} {v.ref}" + (f" · game turn {d.get('turn')}{ut} {d.get('step') or ''}" if d else "") + f" · {kind}"
    lines = [title, ""]
    s = v.state
    if s:
        lines.append(f"- Position: life A {s['life']['A']} / B {s['life']['B']}; A's hand: "
                     f"{', '.join(s['hand_A']) or '-'}; B's hand: {s['hand_B']}")
        lines.append(f"- A's battlefield: {s['battlefield_A']}")
        lines.append(f"- B's battlefield: {s['battlefield_B']}")
        if s.get("stack"):
            lines.append(f"- Stack (bottom to top): {'; '.join(s['stack'])}")
    if d.get("text") and d.get("type") != "PRIORITY":
        lines.append(f"- Question: {d['text']}")
    if v.options:
        hum = set((v.human or {}).get("matched") or [])
        k = max(v.noise.get("k") or 1, 1)
        lines += ["", "| rank | option | mean Q ± sd | win % | visits | n / sample | best in | |",
                  "|---|---|---|---|---|---|---|---|"]
        for o in v.options:
            tag = " ".join(t for t, on in (("best", o.label == v.best), ("human", o.label in hum),
                                           ("not offered", o.label in (v.context.get("not_offered") or ())))
                           if on)
            lines.append(f"| {o.rank} | {display(o.label)} | {_q(o.mean_q)} ± {o.sd_q:.3f} | {_pct(o.win_prob)} | "
                         f"{_pct(o.visit_share, 0)} | {o.n / k:.0f} | {_pct(o.argmax_share, 0)} | {tag} |")
        lines.append("")
    h = v.human or {}
    if h.get("actions"):
        what = ", ".join(display(a) for a in h["actions"])
        lines.append(f"- Human: {what}" + (" (one of these; unordered turn actions)" if h.get("set_valued") else "")
                     + (f". {h['note']}" if h.get("note") else ""))
    verdict = f"**{v.classification}**"
    if v.graded and h.get("regret") is not None and v.classification != "best":
        se = v.noise.get("se_paired")
        verdict += (f": regret {h['regret']:.3f} Q = {_pct(h.get('wp_loss'))} win probability"
                    + (f" (mean of set {_pct(h.get('wp_loss_mean_of_set'))})" if h.get("set_valued") else "")
                    + (f"; paired SE {se:.3f}" if se is not None else "; paired SE unknown")
                    + f", pooled sd {v.noise.get('pooled_sd') or 0:.3f}")
    elif v.classification == "best" and h.get("set_valued") and h.get("regret_mean_of_set"):
        verdict += f" (mean of set {_pct(h.get('wp_loss_mean_of_set'))})"
    if v.reason:
        verdict += f" ({v.reason})"
    lines.append(f"- Verdict: {verdict}")
    fp = v.noise.get("first_pass")
    if fp:
        lines.append(f"- First pass at {fp['budget']} simulations: {fp['classification']}, regret {fp['regret'] or 0:.3f} Q "
                     f"with {fp['visits_per_det'] or 0:.0f} visits per sample on the human's option; searched again at "
                     f"{v.settings.get('budget')}")
    flags = [f for f in v.flags if f != "trivial"]
    if flags:
        lines.append(f"- Flags: {', '.join(flags)}" + (f" (tier {v.fidelity.get('tier')}"
                     + (f"; {', '.join(v.fidelity.get('risky_flags') or [])}" if v.fidelity.get("risky_flags") else "")
                     + ")" if "low_fidelity" in flags else ""))
    det = v.determinization
    if det.get("method"):
        lines.append(f"- Hidden cards: {det['method']} x{det.get('k')}" + (f"; {det['fallback']}" if det.get("fallback") else "")
                     + f"; budget {v.settings.get('budget')}, {v.settings.get('evaluator', {}).get('type')}")
    return "\n".join(lines) + "\n"


def summarize(verdicts: list[Verdict]) -> dict:
    cls = Counter(v.classification for v in verdicts)
    flags = Counter(f.split(":")[0] for v in verdicts for f in v.flags)
    graded = [v for v in verdicts if v.graded]
    reasons = Counter((v.reason or "").split(":")[0][:80] for v in verdicts if v.classification == "ungraded")
    return {"decisions": len(verdicts), "classes": dict(cls), "flags": dict(flags),
            "graded": len(graded), "ungraded_reasons": dict(reasons),
            "mean_wp_loss_graded": _mean([v.human.get("wp_loss") for v in graded if v.human]),
            "methods": dict(Counter(v.determinization.get("method") for v in verdicts if v.determinization)),
            "seconds_per_searched_decision": _mean([v.seconds for v in verdicts if v.options])}


def render_markdown(verdicts: list[Verdict], title: str = "Coaching report", intro: str = "",
                    caveats: list[str] | None = None, games: dict | None = None) -> str:
    """The report: per game, per decision the position, the option table, the human's choice and the
    verdict; then a summary of mistakes (trusted ones apart from flagged ones), luck, and caveats."""
    out = [f"# {title}", ""]
    if intro:
        out += [intro, ""]
    s = summarize(verdicts)
    out.append(f"{s['decisions']} decisions: " + ", ".join(f"{c} {s['classes'][c]}" for c in CLASSES if c in s["classes"])
               + (f". Mean win-probability loss over graded decisions: {_pct(s['mean_wp_loss_graded'])}."
                  if s["mean_wp_loss_graded"] is not None else "."))
    out.append("")
    bad = sorted((v for v in verdicts if v.classification in ("inaccuracy", "mistake", "blunder")),
                 key=lambda v: -(v.human or {}).get("wp_loss", 0))
    if bad:
        out += ["## Mistakes", ""]
        for head, vs in (("Without caution flags:", [v for v in bad if not v.caution]),
                         ("With caution flags (the grade itself is doubtful):", [v for v in bad if v.caution])):
            if not vs:
                continue
            out += [head, ""]
            for v in vs:
                h = v.human or {}
                trust = "" if not v.caution else f" (caution: {', '.join(v.caution)})"
                out.append(f"- {v.ref} · game turn {v.decision.get('turn')}: **{v.classification}**, "
                           f"{', '.join(display(x) for x in h.get('matched') or h.get('actions') or [])} instead of "
                           f"{display(v.best)} ({_pct(h.get('wp_loss'))}){trust}")
            out.append("")
    luck = luck_track(verdicts)
    if luck:
        out += ["## Skill and luck", "",
                "Per game: the sum of the human's regrets (skill) and of the value swings between graded decisions "
                "that the human's choices do not explain (draws, the opponent's play and every uncoached decision "
                "in between). Q units, from A's view.", "",
                "| game | graded | regret sum | luck sum | first value | last value |", "|---|---|---|---|---|---|"]
        for g, x in sorted(luck.items(), key=lambda kv: str(kv[0])):
            out.append(f"| {g} | {x['graded']} | {x['regret_sum']:.3f} | {x['luck_sum']:+.3f} | {_q(x['start_value'])} | "
                       f"{_q(x['end_value'])} |")
        out.append("")
    by_game: dict = {}
    for v in verdicts:
        by_game.setdefault(v.context.get("game"), []).append(v)
    for g, vs in by_game.items():
        if g is not None:
            meta = (games or {}).get(g) or {}
            out += [f"## Game {g}" + (f" ({meta['result']})" if meta.get("result") else ""), ""]
        for v in vs:
            if v.classification == "trivial":
                continue
            out.append(render_verdict(v, "###" if g is not None else "##"))
        trivial = [v.ref for v in vs if v.classification == "trivial"]
        if trivial:
            out += [f"Not coached (trivial): {len(trivial)} decision(s).", ""]
    out += ["## Caveats", ""] + [f"- {c}" for c in (caveats if caveats is not None else CAVEATS)]
    return "\n".join(out).rstrip() + "\n"


def write_arena_report(result: ArenaCoaching, out: Path | None = None, *, allow_outside: bool = False) -> dict:
    """Write <out>/arena_report.md and arena_verdicts.jsonl (default data/gameplay/coach/, which is
    gitignored; other places are refused unless allow_outside, for tests on synthetic logs). The
    privacy check runs on exactly the text written, before anything is written."""
    from draftzero.gameplay import arena as ar
    out = Path(out or DEFAULT_OUT)
    if not allow_outside and not _inside(out, GAMEPLAY_DATA):
        raise ValueError("Arena coaching reports go under data/gameplay/ only (gitignored): "
                         "they describe the user's own games")
    methods = Counter(v.determinization.get("method") for v in result.verdicts if v.determinization)
    fallback = next((v.determinization["fallback"] for v in result.verdicts if v.determinization.get("fallback")), None)
    intro = ("MTG Arena log coached with MageZero's search (bridge coach op). Seat A is the local player. "
             f"Hidden cards: {dict(methods) or 'none searched'}" + (f" ({fallback})." if fallback else "."))
    caveats = list(CAVEATS)
    if fallback:
        caveats.append("The opponent belief model is fitted on 17lands FDN decks; it does not apply to this "
                       "log, so B's hand was re-drawn from a placeholder library (the cards B showed plus basics).")
    if any("substituted" in v.flags for v in result.verdicts):
        caveats.append("Cards this XMage build lacks were built as stand-ins; legal sets and values around them "
                       "are artefacts (flag substituted).")
    text = render_markdown(result.verdicts, "Arena coaching report", intro, caveats, result.games)
    lines = [json.dumps(v.to_dict(), ensure_ascii=False, default=str) for v in result.verdicts]
    summary = dict(summarize(result.verdicts), skipped_other_kinds=dict(result.skipped), logs=result.logs,
                   seconds=result.seconds)
    summary_text = json.dumps(summary, indent=1, ensure_ascii=False, default=str)
    engine_words = {w for v in result.verdicts for x in ([v.decision.get("text") or ""] + v.decision.get("legal", [])
                                                         + [o.label for o in v.options])
                    for w in re.findall(r"\w+", x)}
    ar.check_privacy([text, "\n".join(lines), summary_text], result.identifiers,
                     result.allowed_words | engine_words)
    out.mkdir(parents=True, exist_ok=True)
    (out / "arena_report.md").write_text(text, encoding="utf-8")
    (out / "arena_verdicts.jsonl").write_text("".join(x + "\n" for x in lines), encoding="utf-8")
    (out / "arena_summary.json").write_text(summary_text + "\n", encoding="utf-8")
    return summary


def _inside(p: Path, root: Path) -> bool:
    try:
        Path(p).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------

def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-K", "--determinizations", type=int, default=K_DEFAULT)
    p.add_argument("--budget", type=int, default=BUDGET_DEFAULT)
    p.add_argument("--combat-budget", type=int, default=COMBAT_BUDGET,
                   help="minimum budget for attack/block questions offline (0: keep --budget and flag)")
    p.add_argument("--verify-budget", type=int, default=VERIFY_BUDGET,
                   help="offline re-search budget for a fault on a thinly searched human option (0: flag only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evaluator", choices=["offline", "remote"], default="offline")
    p.add_argument("--port", type=int, default=50052)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--priors", default="", help="comma list of priority,target,binary,opponent (remote only)")
    p.add_argument("--name", default="coach0", help="bridge worker name")
    p.add_argument("--heap", default="2500m")
    p.add_argument("--runtime-root", type=Path, default=None, help="bridge runtime dir root")
    p.add_argument("--no-build", action="store_true", help="use the existing bridge jar even if it is stale")


def _kw(a) -> dict:
    return dict(k=a.determinizations, budget=a.budget, seed=a.seed, combat_budget=a.combat_budget or None,
                verify_budget=a.verify_budget or None,
                evaluator=remote(a.port, a.host) if a.evaluator == "remote" else "offline",
                priors=[x for x in a.priors.split(",") if x] or None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dz gameplay coach", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("17lands", help="coach one user turn of a 17lands replay row")
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--turn", type=int, required=True, help="the user's turn number (17lands, per player)")
    p.add_argument("--blocks", action="store_true", help="the block decision in the opponent's next turn")
    p.add_argument("--hindsight", action="store_true", help="the true position of a mirrored pair (evaluation only)")
    p.add_argument("--replay", type=Path, default=None, help="replay file (default data/17lands)")
    p.add_argument("--out", type=Path, help="write the markdown report here (default stdout)")
    p.add_argument("--json", action="store_true", help="print the verdict as JSON")
    _common(p)
    p = sub.add_parser("arena", help="coach every non-trivial decision of an Arena log")
    p.add_argument("logs", nargs="*", type=Path, help="Player.log files (default: the local ones)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory, under data/gameplay/")
    p.add_argument("--substitute-missing", metavar="CARD", help="build cards XMage lacks as CARD (e.g. Plains)")
    p.add_argument("--substitute-token", metavar="CLASS", help="build tokens XMage lacks as CLASS (e.g. GoblinToken)")
    p.add_argument("--games", type=int, nargs="*", help="only these game numbers")
    p.add_argument("--limit", type=int, help="at most this many decisions")
    _common(p)
    p = sub.add_parser("validate", help="E5 pilot: the coach's regret by player skill on 17lands decisions")
    p.add_argument("--every", type=int, default=2003)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=200, help="decisions to coach")
    p.add_argument("--min-games", type=int, default=100)
    p.add_argument("--replay", type=Path, default=None)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT / "validate_FDN_PremierDraft.json")
    _common(p)
    p = sub.add_parser("spec", help="coach the decision of one StateSpec JSON file")
    p.add_argument("spec", type=Path)
    p.add_argument("--human", help="the human's action label")
    p.add_argument("--human-set", action="append", help="one member of a set-valued human action (repeat)")
    p.add_argument("--hindsight", action="store_true")
    p.add_argument("--json", action="store_true")
    _common(p)
    a = ap.parse_args(argv)

    from draftzero.gameplay import bridge as br
    problems = br.environment_problems()
    if problems:
        print("mzbridge unavailable:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2
    with br.Bridge(a.name, heap=a.heap, runtime_root=a.runtime_root, auto_build=not a.no_build) as b:
        if a.cmd == "17lands":
            try:
                v = coach_17lands(a.row, a.turn, bridge=b, blocks=a.blocks, hindsight=a.hindsight, path=a.replay,
                                  **_kw(a))
            except (ValueError, KeyError, FileNotFoundError) as e:     # no such turn, no attack to block, ...
                print(f"cannot coach row {a.row} turn {a.turn}: {e}", file=sys.stderr)
                return 2
            text = (json.dumps(v.to_dict(), indent=1, default=str) if a.json else
                    render_markdown([v], f"17lands row {a.row}, user turn {a.turn}",
                                    "17lands FDN Premier Draft replay data (CC BY 4.0, 17lands.com/public_datasets)."))
            if a.out:
                a.out.write_text(text)
                print(f"{v.classification} -> {a.out}")
            else:
                print(text)
            return 0
        if a.cmd == "validate":
            def show(r):
                print(f"  row {r['row']} turn {r['turn']}: {r['classification']} ({r['seconds']:.1f} s)",
                      file=sys.stderr, flush=True)
            res = validate_17lands(bridge=b, every=a.every, start=a.start, limit=a.limit, min_games=a.min_games,
                                   path=a.replay, progress=show, **_kw(a))
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
            print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1, default=str))
            return 0
        if a.cmd == "spec":
            human = a.human_set or a.human
            v = coach_decision(a.spec, b, human=human, hindsight=a.hindsight, **_kw(a))
            print(json.dumps(v.to_dict(), indent=1, default=str) if a.json else render_markdown([v], a.spec.name))
            return 0
        sub_opt = {k: v for k, v in (("missing", a.substitute_missing), ("missingToken", a.substitute_token)) if v}

        def tick(v):
            print(f"  {v.ref}: {v.classification} ({v.seconds:.1f} s)", file=sys.stderr, flush=True)
        try:
            res = coach_arena(a.logs or None, bridge=b, substitute=sub_opt or None, games=a.games, limit=a.limit,
                              progress=tick, **_kw(a))
            summary = write_arena_report(res, a.out)
        except (ValueError, FileNotFoundError) as e:
            print(str(e), file=sys.stderr)
            return 2
    rel = os.path.relpath(Path(a.out).resolve(), Path.cwd())
    shown = rel if not rel.startswith("..") else Path(a.out).name       # never print a home directory
    print(json.dumps({k: summary[k] for k in ("decisions", "classes", "flags", "methods", "graded")}, indent=1))
    print(f"-> {shown}/arena_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
