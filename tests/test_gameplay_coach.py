"""Coaching driver (draftzero.gameplay.coach): ranking, regret, thresholds, determinization choice,
human-answer resolution, reports and privacy, on a fake bridge; plus a real-bridge smoke test that
skips without Java and the XMage build.

The fake bridge answers `build` with a decision derived from the spec (or given) and `coach` with
per-determinization children from a table of Q values, aggregated the way java/mzbridge Coach.java
does (mean and sd of Q over the determinizations, mean visit share, rank by mean Q).
"""
import copy
import json
import statistics
from collections import Counter
from pathlib import Path

import pytest

from draftzero.gameplay import arena as ar
from draftzero.gameplay import belief as bl
from draftzero.gameplay import bridge as br
from draftzero.gameplay import coach as co
from draftzero.gameplay import replay
from draftzero.gameplay.statespec import Perm, PlayerState, Provenance, StateSpec

FIX = Path(__file__).parent / "fixtures" / "gameplay"
ARENA_LOG = FIX / "arena_synthetic.log"
ROWS = FIX / "fdn_premier_rows.csv.gz"


# ------------------------------------------------------------------------------ a fake bridge

def aggregate(per_det: list[dict]) -> list[dict]:
    """Coach.java's aggregation over {label: (N, Q)} per determinization."""
    by: dict = {}
    for d in per_det:
        total = sum(n for n, _ in d.values())
        for lab, (n, q) in d.items():
            a = by.setdefault(lab, {"label": lab, "idx": 0, "qs": [], "share": 0.0, "N": 0, "nDet": 0})
            a["nDet"] += 1
            a["N"] += n
            a["share"] += n / total if total else 0
            if q is not None:
                a["qs"].append(q)
    out = []
    for a in by.values():
        qs = a.pop("qs")
        a["meanQ"] = sum(qs) / len(qs) if qs else None
        a["sdQ"] = statistics.stdev(qs) if len(qs) > 1 else 0.0
        a["visitShare"] = a.pop("share") / len(per_det)
        out.append(a)
    out.sort(key=lambda a: (-(a["meanQ"] if a["meanQ"] is not None else -9), -a["visitShare"], a["label"]))
    for i, a in enumerate(out):
        a["rank"] = i + 1
    return out


def table(qs: dict[str, list[float]], n: int | dict = 50) -> list[dict]:
    """{label: [Q per determinization]} -> per-determinization {label: (N, Q)}; n: visits per
    determinization, for every label or per label (default 50)."""
    k = len(next(iter(qs.values())))
    visits = n if isinstance(n, dict) else {}
    return [{lab: (visits.get(lab, 50) if visits else n, q[i]) for lab, q in qs.items()} for i in range(k)]


def _dump(spec: dict) -> dict:
    d = copy.deepcopy(spec)
    for seat, p in d.get("players", {}).items():
        for perm in p.get("battlefield", []):
            perm["x"] = {"name": perm.get("name") or (perm.get("token") or "Token") + " Token", "power": 1, "toughness": 1}
    return d


class FakeBridge:
    """build: `decision` (or one derived from an Arena spec's labels); coach: the Q table (by budget
    when `by_budget` holds one for the request's budget), with `visits` per determinization, and
    `coach_decision` as the question the search reached (default: the build's)."""

    def __init__(self, qs: dict | None = None, decision: dict | None = None, consistent: bool = True,
                 substitutions: dict | None = None, warnings: list | None = None, visits: int | dict = 50,
                 by_budget: dict | None = None, coach_decision: dict | None = None):
        self.qs, self.decision, self.consistent = qs, decision, consistent
        self.substitutions, self.warnings = substitutions, warnings or []
        self.visits, self.by_budget, self.searched = visits, by_budget or {}, coach_decision
        self.calls = []

    def decision_for(self, spec: dict, options: dict) -> dict:
        if self.decision is not None:
            return self.decision
        L = spec.get("labels", {})
        kind = L.get("decision_type")
        where = {"turn": spec["turn"], "phase": spec["phase"], "step": spec["step"], "passedBefore": 0, "stack": 0}
        names = co.alias_names(_dump(spec))
        if kind == "DeclareAttackersReq":
            first = names[L["options"][0]["attacker"]]
            return {"player": "A", "type": "CHOOSE_USE", "text": f"attack with: {first}?",
                    "where": dict(where, step="DECLARE_ATTACKERS", phase="COMBAT"),
                    "legal": [{"label": "no", "idx": 0}, {"label": "yes", "idx": 1}]}
        if kind == "DeclareBlockersReq":
            o = L["options"][0]
            legal = sorted({names[a] for a in o["can_block"]}) + ["Stop Choosing"]
            return {"player": "A", "type": "CHOOSE_TARGET",
                    "text": f"choose which creature to block for {names[o['blocker']]}:Choose a target:attacking creature",
                    "where": dict(where, step="DECLARE_BLOCKERS", phase="COMBAT"),
                    "legal": [{"label": x, "idx": 0} for x in legal]}
        df = (L.get("bridge") or {}).get("decideFrom")
        if kind is None and df:          # a 17lands turn start: the turn's lands and casts, and Pass
            keys = [x["key"] for x in L.get("lands", []) + L.get("casts", [])] + ["Pass"]
            where = dict(where, turn=df["turn"], step=df["step"], phase=df["step"])
        else:
            keys = [o["mz_key"] for o in co.arena_choices(L) if o.get("mz_key")]
        return {"player": "A", "type": "PRIORITY", "text": "priority", "where": where,
                "legal": [{"label": x, "idx": 0} for x in dict.fromkeys(keys)]}

    def request(self, op, spec=None, **options):
        self.calls.append((op, spec, options))
        first = spec if spec is not None else (options.get("specs") or [None])[0]
        dec = self.decision_for(first, options)
        if op == "build":
            out = {"decision": dec, "dump": _dump(first), "decisionState": _dump(first), "warnings": list(self.warnings)}
            if self.substitutions:
                out["substitutions"] = self.substitutions
            return out
        qs, visits = self.by_budget.get(options.get("budget"), (self.qs, self.visits))
        qs = qs or {x["label"]: [0.1 * (i + 1) - 0.05 * j for j in range(4)] for i, x in enumerate(dec["legal"])}
        per_det = table(qs, visits)
        k = len(options["specs"]) if options.get("specs") else options.get("determinizations", len(per_det))
        per_det = (per_det * k)[:k] if len(per_det) < k else per_det
        dets = [{"k": i, "children": [{"label": lab, "idx": 0, "N": n, "Q": q} for lab, (n, q) in d.items()],
                 "value": 0.0} for i, d in enumerate(per_det)]
        agg = aggregate(per_det)
        return {"decision": self.searched or dec, "consistent": self.consistent, "determinizations": dets,
                "aggregate": agg, "best": agg[0]["label"], "warnings": [], "timing_ms": {"simsPerSec": 500}}

    def coach_calls(self):
        return [c for c in self.calls if c[0] == "coach"]


def priority(legal, turn=5, step="PRECOMBAT_MAIN"):
    return {"player": "A", "type": "PRIORITY", "text": "priority",
            "where": {"turn": turn, "phase": step, "step": step, "passedBefore": 0, "stack": 0},
            "legal": [{"label": x, "idx": i} for i, x in enumerate(legal)]}


def golden(name: str) -> dict:
    return json.loads(br.golden_specs()[name].read_text())


# ------------------------------------------------------------------------------ units

def test_win_probability_conversion():
    assert co.win_prob(0.0) == 0.5 and co.win_prob(1.0) == 1.0 and co.win_prob(-1.0) == 0.0
    assert co.wp_loss(0.1) == pytest.approx(0.05) and co.win_prob(None) is None


def test_match_label():
    legal = ["Play Plains", "Swampcycling {1} <i>({1}, Discard this card: ...)</i>",
             "-2: Create a 2/1 blue Ninja creature token.", "Pass"]
    assert co.match_label("Play Plains", legal) == "Play Plains"
    assert co.match_label("Swampcycling {1}", legal).startswith("Swampcycling")        # Arena key, no reminder
    assert co.match_label("Create a 2/1 blue Ninja creature token.", legal).startswith("-2:")   # 17lands, no cost
    assert co.match_label("pass", legal) == "Pass"
    assert co.match_label("Cast Stab", legal) is None
    assert co.match_label("true", ["no", "yes"], "CHOOSE_USE") == "yes"
    assert co.match_label("attack", ["no", "yes"], "CHOOSE_USE") == "yes"


def _options(qs):
    per_det = [{k: v for k, (n, v) in d.items()} for d in table(qs)]
    resp = {"aggregate": aggregate(table(qs)), "determinizations": []}
    return co.options_from(resp, per_det), per_det


def test_grade_classes():
    # a clear, robust gap of 0.3 Q (15% win probability): blunder
    opts, pd = _options({"A": [0.30, 0.31, 0.29, 0.30], "B": [0.00, 0.01, -0.01, 0.00]})
    assert [o.label for o in opts] == ["A", "B"] and opts[0].rank == 1
    g = co.grade(opts, pd, ["B"])
    assert g["classification"] == "blunder" and g["regret"] == pytest.approx(0.3) and g["wp_loss"] == pytest.approx(0.15)
    assert g["significant"] and g["robust"] and g["near_tie"] == ["A"]
    assert co.grade(opts, pd, ["A"])["classification"] == "best"
    # 0.14 Q = 7% win probability, robust: mistake
    opts, pd = _options({"A": [0.20, 0.21, 0.19, 0.20], "B": [0.06, 0.07, 0.05, 0.06]})
    assert co.grade(opts, pd, ["B"])["classification"] == "mistake"
    # 0.06 Q = 3%: inaccuracy
    opts, pd = _options({"A": [0.20, 0.21, 0.19, 0.20], "B": [0.14, 0.15, 0.13, 0.14]})
    assert co.grade(opts, pd, ["B"])["classification"] == "inaccuracy"
    # 0.75% win probability: a near tie, fine
    opts, pd = _options({"A": [0.20, 0.21, 0.19, 0.20], "B": [0.185, 0.195, 0.175, 0.185]})
    g = co.grade(opts, pd, ["B"])
    assert g["classification"] == "fine" and "B" in g["near_tie"]


def test_grade_needs_the_gap_to_beat_the_noise():
    # a 0.2 Q gap on average, but the determinizations disagree: not significant -> fine
    opts, pd = _options({"A": [0.6, -0.2, 0.5, -0.1], "B": [-0.1, 0.4, -0.2, 0.3]})
    g = co.grade(opts, pd, ["B"])
    assert not g["significant"] and g["classification"] == "fine"
    # significant (paired: A beats B in every sample) but the spread across hidden cards is larger
    # than the gap: a large loss stays an inaccuracy, not a mistake
    opts, pd = _options({"A": [0.70, 0.10, 0.50, -0.10], "B": [0.58, -0.02, 0.38, -0.22]})
    g = co.grade(opts, pd, ["B"])
    assert g["significant"] and not g["robust"] and g["wp_loss"] >= co.MISTAKE_WP
    assert g["classification"] == "inaccuracy" and "not robust" in g["reason"]
    # one determinization: the noise is unknown, nothing is called a mistake
    opts, pd = _options({"A": [0.5], "B": [0.0]})
    g = co.grade(opts, pd, ["B"])
    assert g["classification"] == "fine" and g["se_paired"] is None and "noise unknown" in g["reason"]


def test_grade_reports_thinly_searched_human_options():
    qs = {"A": [0.30, 0.31, 0.29, 0.30], "B": [0.00, 0.01, -0.01, 0.00]}
    per_det = [{k: v for k, (n, v) in d.items()} for d in table(qs)]
    for n, thin in ((co.MIN_VISITS - 1, True), (co.MIN_VISITS, False)):
        opts = co.options_from({"aggregate": aggregate(table(qs, {"A": 400, "B": n}))}, per_det)
        g = co.grade(opts, per_det, ["B"])
        assert g["visits_per_det"] == n and g["thin"] is thin and g["classification"] == "blunder"
    # the best option itself is never thin, however few its visits
    opts = co.options_from({"aggregate": aggregate(table(qs, {"A": 5, "B": 5}))}, per_det)
    assert co.grade(opts, per_det, ["A"])["thin"] is False


def test_a_fault_on_a_thin_option_is_searched_again():
    """Regression: at 300 simulations MageZero's mean-Q search undervalues options it hardly
    visits (a land drop before the spell); on 9 real 17lands decisions graded mistake or blunder,
    7 were fine or best at 3,000. Such faults are re-searched deeper, and flagged when still thin."""
    s = golden("main_phase")
    legal = ["Cast Stab", "Play Plains", "Pass"]
    shallow = ({"Cast Stab": [0.30, 0.31, 0.29, 0.30], "Play Plains": [0.00, 0.01, -0.01, 0.00], "Pass": [-0.3] * 4},
               {"Cast Stab": 250, "Play Plains": 20, "Pass": 10})
    deep = ({"Cast Stab": [0.30, 0.31, 0.29, 0.30], "Play Plains": [0.305, 0.30, 0.30, 0.295], "Pass": [-0.3] * 4},
            {"Cast Stab": 1500, "Play Plains": 1400, "Pass": 100})
    fb = FakeBridge(None, priority(legal, s["turn"]), by_budget={300: shallow, co.VERIFY_BUDGET: deep})
    v = co.coach_decision(s, fb, human="Play Plains", k=4)
    assert [c[2]["budget"] for c in fb.coach_calls()] == [300, co.VERIFY_BUDGET]
    assert v.classification in ("best", "fine") and "verified" in v.flags and "thin_search" not in v.flags
    assert v.noise["first_pass"]["classification"] == "blunder" and v.noise["first_pass"]["visits_per_det"] == 20
    assert v.settings["budget"] == co.VERIFY_BUDGET and "First pass at 300 simulations: blunder" in co.render_verdict(v)
    # the same searches at both budgets: the determinizations are not re-drawn
    first, second = (dict(c[2], budget=None) for c in fb.coach_calls())
    assert first == second and first["resample"] == ["B"] and first["determinizations"] == 4
    # still thin after the deeper search, or no deeper search: the fault is kept but cautioned
    fb = FakeBridge(None, priority(legal, s["turn"]), by_budget={300: shallow, co.VERIFY_BUDGET: shallow})
    v = co.coach_decision(s, fb, human="Play Plains", k=4)
    assert v.classification == "blunder" and "verified" in v.flags and "thin_search" in v.caution
    fb = FakeBridge(None, priority(legal, s["turn"]), by_budget={300: shallow})
    v = co.coach_decision(s, fb, human="Play Plains", k=4, verify_budget=None)
    assert len(fb.coach_calls()) == 1 and v.classification == "blunder" and v.caution == ["thin_search"]
    assert "With caution flags" in co.render_markdown([v]) and "Without caution flags" not in co.render_markdown([v])
    # a good answer is not re-searched, nor is a network evaluator's fault (too slow on a CPU)
    fb = FakeBridge(None, priority(legal, s["turn"]), by_budget={300: shallow})
    assert co.coach_decision(s, fb, human="Cast Stab", k=4).classification == "best" and len(fb.coach_calls()) == 1
    fb = FakeBridge(None, priority(legal, s["turn"]), by_budget={300: shallow})
    v = co.coach_decision(s, fb, human="Play Plains", k=4, evaluator="remote:50091")
    assert len(fb.coach_calls()) == 1 and "thin_search" in v.flags


def test_the_yardstick_is_the_best_option_the_human_was_offered():
    """Regression: on the Arena log (cards substituted), 2 of 50 searched priority decisions had an
    XMage-only best option, e.g. 'Play Plains' for a stand-in of a spell, and one was graded an
    inaccuracy against it. Grades are measured against the best option Arena offered."""
    opts, pd = _options({"Play Plains": [0.40, 0.41, 0.39, 0.40], "Cast Sear": [0.30, 0.31, 0.29, 0.30],
                         "Pass": [0.00, 0.01, -0.01, 0.00]})
    assert co.grade(opts, pd, ["Cast Sear"])["classification"] == "mistake"
    g = co.grade(opts, pd, ["Cast Sear"], offered=["Cast Sear", "Pass"])
    assert g["best"] == "Cast Sear" and g["classification"] == "best"
    g = co.grade(opts, pd, ["Pass"], offered=["Cast Sear", "Pass"])
    assert g["best"] == "Cast Sear" and g["regret"] == pytest.approx(0.3)
    # through coach_decision, with Arena's option list (arena_offered)
    s = golden("main_phase")
    L = {"options": [{"mz_key": "Cast Sear"}, {"mz_key": "Pass"}]}
    qs = {"Play Plains": [0.40, 0.41, 0.39, 0.40], "Cast Sear": [0.30, 0.31, 0.29, 0.30], "Pass": [0.0] * 4}
    fb = FakeBridge(qs, priority(list(qs), s["turn"]), visits=co.MIN_VISITS)
    v = co.coach_decision(s, fb, human="Cast Sear", offered=co.arena_offered(L))
    assert v.classification == "best" and v.best == "Cast Sear" and "best_not_offered" in v.flags
    assert v.value["engine_best"] == "Play Plains" and v.context["not_offered"] == ["Play Plains"]
    assert "| 1 | Play Plains |" in co.render_verdict(v) and "| not offered |" in co.render_verdict(v)
    # the human's own choice always counts as offered, even when Arena's key did not match it
    v = co.coach_decision(s, FakeBridge(qs, priority(list(qs), s["turn"]), visits=co.MIN_VISITS), human="Play Plains",
                          offered=lambda d: ["Pass"])
    assert v.classification == "best" and v.best == "Play Plains"


def test_the_search_must_reach_the_question_the_human_was_matched_on():
    s = golden("main_phase")
    legal = ["Play Plains", "Pass"]
    other = dict(priority(legal, s["turn"]), type="CHOOSE_TARGET", text="choose a target")
    v = co.coach_decision(s, FakeBridge(None, priority(legal, s["turn"]), coach_decision=other), human="Pass")
    assert v.classification == "ungraded" and "inconsistent" in v.flags and "another question" in v.reason


def test_grade_set_valued_and_unmatched():
    opts, pd = _options({"A": [0.30, 0.31, 0.29, 0.30], "B": [0.20, 0.21, 0.19, 0.20],
                         "C": [0.00, 0.01, -0.01, 0.00]})
    g = co.grade(opts, pd, ["C", "B", "Zzz"])
    assert g["label"] == "B" and g["regret_best_of_set"] == pytest.approx(0.1)
    assert g["regret_mean_of_set"] == pytest.approx(0.3 - 0.1) and g["member_ranks"] == {"C": 3, "B": 2}
    assert co.grade(opts, pd, ["Zzz"])["classification"] == "ungraded"


def test_coach_decision_ranks_regret_and_request_options():
    s = golden("main_phase")
    s["labels"] = {"bridge": {"decisionPlayer": "A"}}
    qs = {"Cast Stab": [0.05, 0.06, 0.04, 0.05], "Play Plains": [0.20, 0.21, 0.19, 0.20], "Pass": [0.0, 0.01, -0.01, 0.0]}
    fb = FakeBridge(qs, priority(list(qs), turn=s["turn"]), visits=co.MIN_VISITS)     # well searched: no re-search
    v = co.coach_decision(s, fb, human="Cast Stab", k=4, seed=3)
    assert [o.label for o in v.options] == ["Play Plains", "Cast Stab", "Pass"] and v.best == "Play Plains"
    assert v.human["label"] == "Cast Stab" and v.human["rank"] == 2 and v.human["visits_per_det"] == co.MIN_VISITS
    assert v.human["regret"] == pytest.approx(0.15) and v.human["wp_loss"] == pytest.approx(0.075)
    assert v.classification == "mistake" and v.graded and not v.caution and len(fb.coach_calls()) == 1
    build, coach = fb.calls[0], fb.coach_calls()[0]
    assert build[0] == "build" and build[2]["decisionPlayer"] == "A" and build[2]["seed"] == 3
    assert coach[2]["decisionPlayer"] == "A" and coach[2]["budget"] == 300 and coach[2]["evaluator"] == {"type": "offline"}
    # a complete spec (golden: both hands known, exact lists) -> the bridge re-draws B's hand
    assert v.determinization["method"] == "bridge_resample" and coach[2]["resample"] == ["B"]
    assert coach[2]["determinizations"] == 4 and "specs" not in coach[2]
    assert v.state["hand_A"] and v.state["life"] == {"A": s["players"]["A"].get("life", 20),
                                                     "B": s["players"]["B"].get("life", 20)}
    d = json.loads(json.dumps(v.to_dict(), default=str))
    assert d["classification"] == "mistake"
    back = co.Verdict.from_dict(d)
    assert back.options == v.options and back.human == v.human and co.render_verdict(back) == co.render_verdict(v)


def test_coach_decision_flags():
    s = golden("main_phase")
    legal = ["Play Plains", "Pass"]
    # the best option differs across determinizations -> unstable
    fb = FakeBridge({"Play Plains": [0.3, 0.3, -0.1, -0.1], "Pass": [0.0, 0.0, 0.1, 0.1]}, priority(legal, s["turn"]))
    v = co.coach_decision(s, fb, human="Pass", k=4)
    assert "unstable" in v.flags and v.options[0].argmax_share == 0.5
    # determinizations reached different questions -> not graded
    v = co.coach_decision(s, FakeBridge(None, priority(legal, s["turn"]), consistent=False), human="Pass", k=4)
    assert v.classification == "ungraded" and "inconsistent" in v.flags
    # one legal option -> trivial, no search
    fb = FakeBridge(None, priority(["Pass"], s["turn"]))
    v = co.coach_decision(s, fb, human="Pass")
    assert v.classification == "trivial" and not fb.coach_calls()
    # the bridge reached another decision than the logged one -> not graded, no search
    fb = FakeBridge(None, priority(legal, s["turn"] + 2))
    v = co.coach_decision(s, fb, human="Pass", expect=lambda d: "later turn" if d["turn"] != s["turn"] else None)
    assert v.classification == "ungraded" and "decision_mismatch" in v.flags and not fb.coach_calls()
    # the human's answer is not legal here: not graded, and no search spent on it
    fb = FakeBridge(None, priority(legal, s["turn"]))
    v = co.coach_decision(s, fb, human="Cast Nothing")
    assert v.classification == "ungraded" and "human_unmatched" in v.flags and not fb.coach_calls()
    # no human answer at all: the options are still rated
    v = co.coach_decision(s, FakeBridge(None, priority(legal, s["turn"])))
    assert v.classification == "ungraded" and v.reason == "no human answer given" and len(v.options) == 2
    # low-fidelity specs and substitutions: visible ones lower the fidelity, library-only ones do not
    t = copy.deepcopy(s)
    t["provenance"] = {"source": "arena", "ref": "g", "tier": "T2", "flags": ["control_changed"]}
    v = co.coach_decision(t, FakeBridge(None, priority(legal, s["turn"])), human="Pass")
    assert "low_fidelity" in v.flags and v.fidelity["risky_flags"] == ["control_changed"]
    # a 17lands block spec whose attack triggers were skipped (evasion, "can't block"): its legal
    # blocks are not the real game's, though reconstruct grades it only T1
    t["provenance"] = {"source": "17lands", "ref": "g", "tier": "T1", "flags": ["attack_triggers_skipped"]}
    assert "low_fidelity" in co.coach_decision(t, FakeBridge(None, priority(legal, s["turn"])), human="Pass").flags
    lib = ["substitute: 'Sear' -> 'Plains' (not in the XMage card database): {decklist=1}"]
    hand = ["substitute: 'Hidden Lair' -> 'Plains' (not in the XMage card database): {decklist=1, hand=1}"]
    v = co.coach_decision(s, FakeBridge(None, priority(legal, s["turn"]), substitutions={"Sear": "Plains"},
                                        warnings=lib), human="Pass")
    assert "substituted_library" in v.flags and "low_fidelity" not in v.flags
    v = co.coach_decision(s, FakeBridge(None, priority(legal, s["turn"]), substitutions={"Hidden Lair": "Plains"},
                                        warnings=hand), human="Pass")
    assert "substituted" in v.flags and "low_fidelity" in v.flags and "low_fidelity" in v.caution


def test_the_position_shows_logged_names_not_stand_ins():
    """With options.substitute the dump names a stand-in ("Plains") where the log has a card this
    XMage lacks; the report's position showed the stand-ins (in 33 hands and 25 battlefields of the
    66 Arena verdicts). It shows the logged name, starred."""
    s = golden("main_phase")
    subs = {"Stab": "Plains", "Cackling Prowler": "Plains"}
    fb = FakeBridge(None, priority(["Play Plains", "Pass"], s["turn"]), substitutions=subs,
                    warnings=["substitute: 'Stab' -> 'Plains' (not in the XMage card database): {decklist=1, hand=1}"])
    real = fb.request

    def request(op, spec=None, **o):                     # the engine's dump names the stand-ins
        out = real(op, spec, **o)
        if op == "build":
            for d in (out["dump"], out["decisionState"]):
                for p in d["players"].values():
                    p["hand"] = [subs.get(c, c) for c in p.get("hand", [])]
                    for perm in p.get("battlefield", []):
                        if perm.get("name") in subs:
                            perm["name"] = perm["x"]["name"] = subs[perm["name"]]
        return out
    fb.request = request
    v = co.coach_decision(s, fb, human="Pass")
    assert "Stab*" in v.state["hand_A"] and v.state["hand_A"].count("Plains") == 1
    assert "Cackling Prowler*" in v.state["battlefield_B"] and "substituted" in v.flags
    # without substitutions the engine's view stands
    v = co.coach_decision(s, FakeBridge(None, priority(["Play Plains", "Pass"], s["turn"])), human="Pass")
    assert "Stab" in v.state["hand_A"] and "Cackling Prowler" in v.state["battlefield_B"]


def test_combat_questions_get_the_combat_budget():
    s = golden("attack")
    dec = {"player": "B", "type": "CHOOSE_USE", "text": "attack with: Beast Token?",
           "where": {"turn": 8, "step": "DECLARE_ATTACKERS", "passedBefore": 1},
           "legal": [{"label": "no", "idx": 0}, {"label": "yes", "idx": 1}]}
    fb = FakeBridge(None, dec)
    v = co.coach_decision(s, fb, human="yes", options={"decisionPlayer": "B"}, budget=300)
    assert fb.coach_calls()[0][2]["budget"] == co.COMBAT_BUDGET and "combat_budget_raised" in v.flags
    fb = FakeBridge(None, dec)
    v = co.coach_decision(s, fb, human="yes", options={"decisionPlayer": "B"}, budget=300, combat_budget=None)
    assert fb.coach_calls()[0][2]["budget"] == 300 and "combat_low_budget" in v.flags and "combat_low_budget" in v.caution
    # a network evaluator is not second-guessed
    fb = FakeBridge(None, dec)
    v = co.coach_decision(s, fb, human="yes", options={"decisionPlayer": "B"}, evaluator="remote:50091",
                          priors=["binary"])
    assert fb.coach_calls()[0][2]["budget"] == 300 and not {"combat_low_budget", "combat_budget_raised"} & set(v.flags)
    assert fb.coach_calls()[0][2]["evaluator"] == {"type": "remote", "host": "127.0.0.1", "port": 50091}
    assert fb.coach_calls()[0][2]["priors"] == {"binary": True}


# ------------------------------------------------------------------------------ determinization choice

BASICS = {"Plains": "W", "Island": "U", "Swamp": "B"}
CARDS = sorted(BASICS) + [f"{c}crea{i}" for c in "WUB" for i in range(12)] + [f"{c}spell{i}" for c in "WUB" for i in range(8)]


def _synthetic_model():
    kinds = {**{b: bl.KIND_LAND for b in BASICS}, **{c: bl.KIND_CREATURE for c in CARDS if "crea" in c},
             **{c: bl.KIND_OTHER for c in CARDS if "spell" in c}}
    mvs = {c: 1 + int(c[-1]) % 5 for c in CARDS if c[-1].isdigit()}
    cols = {**BASICS, **{c: c[0] for c in CARDS if c not in BASICS}}
    land = {"W": "Plains", "U": "Island", "B": "Swamp"}
    decks = []
    for a, b in (("W", "U"), ("W", "B"), ("U", "B")):
        for k in range(4):
            d = Counter({land[a]: 9, land[b]: 8})
            d.update([f"{a}crea{i}" for i in range(k, k + 6)] + [f"{a}spell{i}" for i in range(k % 3, k % 3 + 5)]
                     + [f"{b}crea{i}" for i in range(k, k + 6)] + [f"{b}spell{i}" for i in range(6)])
            decks.append((d, a + b, ""))
    pool = bl.DeckPool.from_decks(decks, CARDS)
    return bl.OpponentModel(pool, bl.CardMeta.simple(CARDS, kinds, mvs, cols, BASICS), alpha=0.3)


def _partial_spec(names_a=None, decklist_source="placeholder") -> StateSpec:
    names_a = names_a or (["Plains"] * 17 + [f"Wcrea{i}" for i in range(12)] + [f"Wspell{i}" for i in range(8)]
                          + ["Uspell0", "Uspell1", "Uspell2"])
    b_deck = ["Island"] * 20 + ["Swamp"] * 18 + ["Ucrea3", "Bcrea2"]
    A = PlayerState(decklist=names_a, hand=["Plains", names_a[20]], librarySize=30,
                    battlefield=[Perm(name="Plains", id="p1"), Perm(name=names_a[18], id="w1")])
    B = PlayerState(decklist=b_deck, decklistSource=decklist_source, handUnknown=5, librarySize=30,
                    battlefield=[Perm(name="Island", id="i1"), Perm(name="Ucrea3", id="u3")], graveyard=["Bcrea2"])
    s = StateSpec(turn=5, activePlayer="A", players={"A": A, "B": B},
                  provenance=Provenance(source="synthetic", ref="partial", tier="T0"),
                  labels={"bridge": {"decisionPlayer": "A"}})
    assert s.validate() == [] and s.is_partial()
    return s


def test_partial_spec_is_determinized_by_the_belief_model():
    model = _synthetic_model()
    s = _partial_spec()
    fb = FakeBridge(None, priority(["Play Plains", "Pass"], 5))
    v = co.coach_decision(s, fb, model=model, k=3, seed=7, human="Pass")
    call = fb.coach_calls()[0]
    assert v.determinization["method"] == "belief" and v.determinization["retention"] is False
    assert call[1] is None and len(call[2]["specs"]) == 3 and "resample" not in call[2]
    for x in call[2]["specs"]:
        t = StateSpec.from_dict(x)
        assert not t.is_partial() and t.players["B"].decklistSource == "belief" and t.validate() == []
        assert Counter({"Ucrea3": 1, "Bcrea2": 1}) <= Counter(t.players["B"].decklist)
    assert v.determinization["opp_colors"] == "seen-card colours"
    # the same seed gives the same samples (belief.determinize is seeded per determinization)
    fb2 = FakeBridge(None, priority(["Play Plains", "Pass"], 5))
    co.coach_decision(s, fb2, model=model, k=3, seed=7, human="Pass")
    assert fb2.coach_calls()[0][2]["specs"] == call[2]["specs"]
    # an exact (mirrored-pair) decklist keeps its cards: only the hand is sampled
    e = _partial_spec(decklist_source="exact")
    fb3 = FakeBridge(None, priority(["Play Plains", "Pass"], 5))
    v = co.coach_decision(e, fb3, model=model, k=2, human="Pass")
    assert v.determinization["method"] == "belief" and "only B's hand" in v.determinization["note"]
    assert all(Counter(x["players"]["B"]["decklist"]) == Counter(e.players["B"].decklist)
               for x in fb3.coach_calls()[0][2]["specs"])


def test_belief_model_falls_back_outside_its_card_pool():
    model = _synthetic_model()
    cube = ["Plains"] * 17 + [f"Cube card {i}" for i in range(23)]
    s = _partial_spec(names_a=cube)
    fb = FakeBridge(None, priority(["Play Plains", "Pass"], 5))
    v = co.coach_decision(s, fb, model=model, k=3, human="Pass")
    assert v.determinization["method"] == "bridge_resample" and "not applicable" in v.determinization["fallback"]
    assert fb.coach_calls()[0][2]["resample"] == ["B"] and "belief_fallback" in v.flags
    # and when told not to use it
    v = co.coach_decision(_partial_spec(), FakeBridge(None, priority(["Pass", "Play Plains"], 5)), model=model,
                          belief=False, human="Pass")
    assert v.determinization["method"] == "bridge_resample" and v.determinization["fallback"] == "belief model disabled"


def test_hindsight_uses_the_true_hand_only_when_the_spec_has_it():
    s = golden("main_phase")                                  # complete: exact lists, known hands
    fb = FakeBridge(None, priority(["Play Plains", "Pass"], s["turn"]))
    v = co.coach_decision(s, fb, hindsight=True, human="Pass")
    assert v.determinization["method"] == "hindsight" and fb.coach_calls()[0][2]["resample"] == []
    assert "hindsight" in v.flags and "hindsight" in v.caution
    fb = FakeBridge(None, priority(["Play Plains", "Pass"], 5))
    v = co.coach_decision(_partial_spec(), fb, model=_synthetic_model(), k=2, hindsight=True, human="Pass")
    assert v.determinization["method"] == "belief" and "hindsight_unavailable" in v.determinization
    assert "hindsight" not in v.flags                          # a placeholder list holds no truth


def test_a_verdict_that_keeps_part_of_the_truth_is_flagged_hindsight():
    """Regression: a mirrored-pair spec with an unknown card in B's hand (reconstruct.exact_spec,
    flag opp_hand_unknown_card) fell back to the belief draw, which keeps B's real list and known
    hand, and the verdict came out without the hindsight flag, i.e. as a trusted verdict."""
    model = _synthetic_model()
    e = _partial_spec(decklist_source="exact")
    e.players["B"].decklist = e.players["B"].decklist[1:] + ["Ucrea4"]
    e.players["B"].hand, e.players["B"].handUnknown = ["Ucrea4"], 4   # the partner row's hand, one card unknown
    e.provenance.source = "17lands"
    assert e.validate() == [] and e.is_partial()
    fb = FakeBridge(None, priority(["Play Plains", "Pass"], 5))
    v = co.coach_decision(e, fb, model=model, k=2, hindsight=True, human="Pass")
    assert v.determinization["method"] == "belief" and v.determinization["hindsight_partial"]
    assert "hindsight" in v.flags and "hindsight" in v.caution
    assert all("Ucrea4" in x["players"]["B"]["hand"] for x in fb.coach_calls()[0][2]["specs"])
    # the same spec without hindsight=True: a logged game's exact list is still the truth
    v = co.coach_decision(e, FakeBridge(None, priority(["Play Plains", "Pass"], 5)), model=model, k=2, human="Pass")
    assert "hindsight" in v.flags
    # a complete logged-game spec with B's exact list: the bridge re-draws B's hand from the real deck
    s = golden("main_phase")
    s["provenance"] = dict(s["provenance"], source="17lands")
    v = co.coach_decision(s, FakeBridge(None, priority(["Play Plains", "Pass"], s["turn"])), human="Pass")
    assert v.determinization["method"] == "bridge_resample" and "hindsight" in v.flags
    # synthetic specs (the golden scenarios) own their lists: no flag
    v = co.coach_decision(golden("main_phase"), FakeBridge(None, priority(["Play Plains", "Pass"], s["turn"])),
                          human="Pass")
    assert "hindsight" not in v.flags


# ------------------------------------------------------------------------------ human answers

def test_human_17lands_resolver():
    L = {"lands": [{"key": "Play Island"}], "casts": [{"key": "Cast Aegis Turtle"}],
         "activations": [{"key": "+1: Draw a card."}], "attacks": {"A:Ninja_8": True, "A:Turtle_3": False},
         "blocks": [["A:Bear_1", "B:Wolf_2", True], ["A:Elf_4", None, True], ["A:Cat_5", "B:Wolf_2", False]],
         "block_pairing": "ambiguous"}
    r = co.human_17lands(L)
    h = r(priority(["Play Island", "Pass"]), {})
    assert h.set_valued and h.labels == ["Play Island", "Cast Aegis Turtle", "+1: Draw a card.", "Pass"]
    assert "attacked" in h.note
    quiet = co.human_17lands({"lands": [], "casts": [], "activations": [], "attacks": {}})(priority(["Pass"]), {})
    assert quiet.labels == ["Pass"] and not quiet.set_valued
    names = {"A:Ninja_8": "Ninja Token", "A:Turtle_3": "Aegis Turtle", "A:Bear_1": "Bear", "A:Elf_4": "Elf",
             "A:Cat_5": "Cat", "B:Wolf_2": "Wolf"}
    ask = lambda t: {"type": "CHOOSE_USE", "text": f"attack with: {t}?"}                     # noqa: E731
    assert r(ask("Ninja Token"), names).labels == ["yes"] and r(ask("Aegis Turtle"), names).labels == ["no"]
    blk = lambda b: {"type": "CHOOSE_TARGET",                                                # noqa: E731
                     "text": f"choose which creature to block for {b}:Choose a target:attacking creature"}
    assert r(blk("Bear"), names).labels == ["Wolf"] and r(blk("Elf"), names).labels == ["Stop Choosing"]
    assert "not exact" in r(blk("Cat"), names).reason and not r(blk("Cat"), names).labels   # inexact: not graded
    # "did not block" is inexact when the creature left the battlefield outside combat that turn,
    # even with a unique pairing (the reason used to say "block pairing ... is unique")
    gone = co.human_17lands({"blocks": [["A:Elf_4", None, False]], "block_pairing": "unique"})(blk("Elf"), names)
    assert not gone.labels and "not exact" in gone.reason and "left the battlefield" in gone.reason


def test_arena_triviality_and_resolver():
    L = {"decision_type": "ActionsAvailableReq", "mz_decision": "PRIORITY",
         "options": [{"arena_action_type": "Cast", "mz_key": "Cast Big", "cost": "{5}", "autotap": False},
                     {"arena_action_type": "Activate_Mana", "mz_key": "{T}: Add {W}."},
                     {"arena_action_type": "Pass", "mz_key": "Pass"}, {"arena_action_type": "FloatMana"}],
         "chosen": [{"arena_action_type": "Pass", "mz_key": "Pass"}]}
    assert co.arena_trivial(L)                                              # only Pass is really on offer
    L["options"].append({"arena_action_type": "Play", "mz_key": "Play Plains"})
    assert co.arena_trivial(L) is None
    assert co.human_arena(L)(priority(["Play Plains", "Pass"]), {}).labels == ["Pass"]
    L["chosen"] = [{"arena_action_type": "Activate_Mana", "mz_key": "{T}: Add {W}."}]
    assert "mana" in co.human_arena(L)(priority(["Pass"]), {}).reason
    A = {"decision_type": "DeclareAttackersReq", "options": [{"attacker": "A:1"}, {"attacker": "A:2"}],
         "attack": {"A:1": True, "A:2": False}, "chosen": []}
    same = {"A:1": "Skeleton Token", "A:2": "Skeleton Token"}
    ask = {"type": "CHOOSE_USE", "text": "attack with: Skeleton Token?"}
    assert "different answers" in co.human_arena(A)(ask, same).reason       # copies answered differently
    assert co.human_arena(A)(ask, {"A:1": "Skeleton Token", "A:2": "Goblin"}).labels == ["yes"]
    agree = co.legal_set_agreement({"options": [{"mz_key": "Play Plains"}, {"mz_key": "Swampcycling {1}"},
                                                {"mz_key": "Pass"}]},
                                   ["Play Plains", "Swampcycling {1} <i>(...)</i>", "Pass", "Cast Sear"])
    assert agree["arena_only"] == [] and agree["xmage_only"] == ["Cast Sear"]


def test_coach_arena_on_the_synthetic_log(tmp_path):
    fb = FakeBridge()
    res = co.coach_arena([ARENA_LOG], bridge=fb, k=2, card_db=False, belief=False)
    by = {v.ref: v for v in res.verdicts}
    kinds = Counter(v.context["kind"] for v in res.verdicts)
    assert kinds == {"priority": 6, "attack": 1, "block": 1}
    assert res.skipped == {"MulliganReq": 2, "ChooseStartingPlayerReq": 1, "SelectTargetsReq": 1}
    # a priority decision with Pass (and mana) only is trivial and never searched
    trivial = [v for v in res.verdicts if v.classification == "trivial"]
    assert trivial and all(v.context["kind"] == "priority" for v in trivial)
    graded = [v for v in res.verdicts if v.graded]
    assert {v.context["kind"] for v in graded} == {"priority", "attack", "block"}
    atk = next(v for v in graded if v.context["kind"] == "attack")
    assert atk.human["label"] == "yes" and "combat_budget_raised" in atk.flags
    blk = next(v for v in graded if v.context["kind"] == "block")
    assert blk.human["label"] != "Stop Choosing" and blk.decision["type"] == "CHOOSE_TARGET"
    # every coach request carried the spec's bridge options (decideFrom for attack and block specs)
    for op, spec, opts in fb.coach_calls():
        assert opts["decisionPlayer"] == "A" and opts["resample"] == ["B"]
        if spec["labels"]["decision_type"] != "ActionsAvailableReq":
            assert opts["decideFrom"]["step"] in co.COMBAT_STEPS
    # the report: written only after the privacy check, only under data/gameplay unless told otherwise
    with pytest.raises(ValueError, match="data/gameplay"):
        co.write_arena_report(res, tmp_path)
    summary = co.write_arena_report(res, tmp_path, allow_outside=True)
    text = (tmp_path / "arena_report.md").read_text()
    assert summary["decisions"] == len(res.verdicts) and "## Game 1" in text and "## Caveats" in text
    assert "| rank | option |" in text and "Not coached (trivial)" in text
    lines = (tmp_path / "arena_verdicts.jsonl").read_text().splitlines()
    assert len(lines) == len(res.verdicts) and json.loads(lines[0])["ref"] in by
    for fake in ("FAKEACCT0000042", "Testy McFakeface", "Opponent Fakename", "fakeuser",
                 "00000000-1111-2222-3333-444444444444"):
        assert fake not in text and not any(fake in x for x in lines)


def test_arena_report_refuses_to_leak_an_identifier(tmp_path):
    fb = FakeBridge()
    res = co.coach_arena([ARENA_LOG], bridge=fb, k=2, card_db=False, limit=3, belief=False)
    res.verdicts[0].warnings.append("seen at Testy McFakeface#12345's table")
    with pytest.raises(ar.PrivacyError):
        co.write_arena_report(res, tmp_path, allow_outside=True)
    assert not (tmp_path / "arena_report.md").exists()


def test_render_markdown_and_luck():
    s = golden("main_phase")
    qs = {"Play Plains": [0.20, 0.21, 0.19, 0.20], "Pass": [-0.05, -0.04, -0.06, -0.05]}
    vs = []
    for i, h in enumerate(("Pass", "Play Plains", "Pass")):
        v = co.coach_decision(s, FakeBridge(qs, priority(list(qs), s["turn"])), human=h, ref=f"game1:decision{i}")
        v.context.update(game=1, kind="priority")
        vs.append(v)
    text = co.render_markdown(vs, "Test report")
    assert text.startswith("# Test report") and "## Mistakes" in text and "## Skill and luck" in text
    assert "| 1 | Play Plains | +0.200" in text and "| 50% | 50 | 100% |" in text and "best human" in text and "**blunder**" in text
    luck = co.luck_track(vs)[1]
    assert luck["graded"] == 3 and luck["regret_sum"] == pytest.approx(0.5)
    # V(next) - Q(human): 0.2 - (-0.05) after a Pass, 0.2 - 0.2 after the best play
    assert luck["swings"] == pytest.approx([0.25, 0.0])
    one = co.render_markdown(vs[:1], "One")                     # a lone decision has no luck section
    assert "Skill and luck" not in one


# ------------------------------------------------------------------------------ 17lands driver (fake bridge)

@pytest.fixture(scope="module")
def fixture_games():
    return {g.row_index: g for g in replay.iter_games(ROWS)}


@pytest.fixture(scope="module")
def fixture_model(fixture_games):
    from draftzero.gameplay.ids import Ids
    decks = [(Counter(g.deck), g.meta.get("main_colors") or "", g.meta.get("splash_colors") or "")
             for g in fixture_games.values()]
    names = sorted({c for d, _, _ in decks for c in d})
    return bl.OpponentModel(bl.DeckPool.from_decks(decks, names), bl.CardMeta.from_ids(names, Ids.load()), alpha=0.3)


def test_coach_17lands_with_a_fake_bridge(fixture_games, fixture_model):
    g = fixture_games[4]
    fb = FakeBridge({"Cast Banishing Light": [0.3] * 4, "Cast Youthful Valkyrie": [0.1] * 4, "Play Plains": [0.12] * 4,
                     "Pass": [0.0] * 4}, priority(["Cast Banishing Light", "Cast Youthful Valkyrie", "Play Plains", "Pass"],
                                                  turn=8))
    v = co.coach_17lands(4, 4, bridge=fb, game=g, pairs={}, model=fixture_model, k=3)
    op, spec, opts = fb.coach_calls()[0]
    assert opts["decideFrom"] == {"turn": 8, "step": "PRECOMBAT_MAIN"} and opts["decisionPlayer"] == "A"
    assert v.determinization["method"] == "belief" and len(opts["specs"]) == 3
    assert v.human["set_valued"] and set(v.human["matched"]) == {"Play Plains", "Cast Youthful Valkyrie", "Pass"}
    assert v.human["label"] == "Play Plains" and v.human["regret_best_of_set"] == pytest.approx(0.18)
    assert v.context["row"] == 4 and v.context["mirrored_pair"] is False
    assert "pair_status_unknown" not in v.flags
    # the bridge moved on to another turn: not graded
    fb = FakeBridge(None, priority(["Play Plains", "Pass"], turn=10))
    v = co.coach_17lands(4, 4, bridge=fb, game=g, pairs={}, model=fixture_model, k=2)
    assert v.classification == "ungraded" and "decision_mismatch" in v.flags


def test_a_missing_pairs_file_is_not_silent(monkeypatch, fixture_games, fixture_model):
    """Without data/gameplay/pairs_*.jsonl a mirrored-pair row keeps its partner's draft (the
    opponent's real deck) in the belief pool: say so instead of grading as if it were clean."""
    monkeypatch.setitem(co._PARTNERS, "map", None)
    fb = FakeBridge(None, priority(["Cast Youthful Valkyrie", "Play Plains", "Pass"], turn=8))
    v = co.coach_17lands(4, 4, bridge=fb, game=fixture_games[4], model=fixture_model, k=2)
    assert "pair_status_unknown" in v.caution and v.context["notes"]
    res = co.validate_17lands(bridge=FakeBridge(), path=ROWS, every=1, limit=2, min_games=0, model=fixture_model,
                              k=2, turns=range(2, 8))
    assert res["warnings"] and "mirrored_pair" not in res["skipped"]


def test_spearman():
    assert co.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert co.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert co.spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(0.894, abs=1e-3)     # average ranks for ties
    assert co.spearman([1, 1], [1, 2]) is None


def test_validate_groups_regret_by_skill(monkeypatch, fixture_model):
    monkeypatch.setitem(co._PARTNERS, "map", {76: 1})              # pretend fixture row 76 is half of a pair
    res = co.validate_17lands(bridge=FakeBridge(), path=ROWS, every=1, limit=20, min_games=0, model=fixture_model,
                              k=2, turns=range(2, 8))
    assert res["skipped"]["mirrored_pair"] == 1 and res["decisions"] == 8
    assert all(r["row"] != 76 for r in res["rows"]) and res["graded"] >= 6
    for groups in (res["by_rank"], res["by_win_rate_band"], res["by_result"]):
        assert sum(g["n"] for g in groups.values()) == res["graded"]
        assert all(g["share_faults_without_caution"] <= g["share_inaccuracy_or_worse"] for g in groups.values())
    assert res["n_spearman"] == res["graded"] and "draft_id" not in json.dumps(res) and res["warnings"] == []


# ------------------------------------------------------------------------------ real bridge

PROBLEMS = br.environment_problems()
needs_worker = pytest.mark.skipif(bool(PROBLEMS), reason="mzbridge unavailable: " + "; ".join(PROBLEMS))


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    try:
        # the jar is shared with other sessions: build it only when it is missing (the bridge's own
        # tests check that it is current); a stale one still speaks the protocol this test uses
        b = br.Bridge("coachtest", heap="2g", runtime_root=tmp_path_factory.mktemp("mzbridge"),
                      auto_build=not br.JAR.exists())
    except br.BridgeUnavailable as e:        # e.g. java/mzbridge sources that do not compile right now
        pytest.skip(f"mzbridge worker unavailable: {str(e).splitlines()[0]}")
    yield b
    b.close()


@needs_worker
def test_real_bridge_smoke(worker, fixture_games, fixture_model):
    s = golden("main_phase")
    for human in ("Play Plains", "Pass"):
        v = co.coach_decision(s, worker, human=human, k=2, budget=60, seed=1, options={"decisionPlayer": "A"},
                              verify_budget=150)
        assert v.classification in co.GRADED and [o.rank for o in v.options] == list(range(1, len(v.options) + 1))
        assert v.decision["type"] == "PRIORITY" and human in v.decision["legal"]
        # the regret is the bridge's own (best mean Q minus the human option's), at the budget the
        # grade stands on (a re-searched fault: the same determinizations, searched deeper)
        assert v.settings["budget"] == (150 if "verified" in v.flags else 60)
        assert ("first_pass" in v.noise) == ("verified" in v.flags)
        r = worker.coach(s, seed=1, budget=v.settings["budget"], determinizations=2, decisionPlayer="A",
                         resample=["B"], humanAction=human)
        assert v.human["regret"] == pytest.approx(r["human"]["regret"], abs=1e-9)
    # a 17lands fixture turn, determinized by a (tiny) belief model, the turn's actions as a set
    v = co.coach_17lands(4, 4, bridge=worker, game=fixture_games[4], pairs={}, model=fixture_model, k=2, budget=40,
                         verify_budget=None)
    assert v.determinization["method"] == "belief" and v.decision["step"] == "PRECOMBAT_MAIN"
    assert v.classification in co.GRADED and v.human["set_valued"] and v.human["matched"]
    assert "verified" not in v.flags and v.settings["budget"] == 40
    # the same turn is a fault at 40 simulations on a thinly searched option (offline search is
    # deterministic): it is searched again, deeper, over the same belief samples, and the grade is
    # the bridge's own set regret at that budget
    class Spy:
        def __init__(self, b):
            self.b, self.calls = b, []

        def request(self, op, spec=None, **o):
            self.calls.append((op, spec, o))
            return self.b.request(op, spec, **o)

    spy = Spy(worker)
    v = co.coach_17lands(4, 4, bridge=spy, game=fixture_games[4], pairs={}, model=fixture_model, k=2, budget=40,
                         verify_budget=120)
    calls = [c for c in spy.calls if c[0] == "coach"]
    assert [c[2]["budget"] for c in calls] == [40, 120] and "verified" in v.flags, v.flags
    assert calls[0][2]["specs"] == calls[1][2]["specs"] and v.noise["first_pass"]["budget"] == 40
    r = worker.request("coach", calls[1][1], **dict(calls[1][2], humanActions=v.human["actions"]))
    assert v.human["regret_best_of_set"] == pytest.approx(r["humanSet"]["regret_best_of_set"], abs=1e-9)
