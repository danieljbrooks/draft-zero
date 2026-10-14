"""draftzero.engine.specs: PolicySpec renders the fork's spec strings in one order and parses them back."""
import pytest

from draftzero.engine.specs import DECIDED_DEFAULT, FORCED_DEFAULT, FULL, KIND_NAMES, PolicySpec


def test_render_contract_example():
    s = PolicySpec(sims=100, explore=True, closedlist=True, cpuct=0.25, topk=6, gonet="/x/v3.dzgw")
    assert s.render() == ("az:sims=100:explore:autopay:oppnodes:oppfull:morekinds:closedlist:cpuct=0.25:topk=6:"
                          "gonet=/x/v3.dzgw")
    assert str(s) == s.render()


def test_defaults():
    assert PolicySpec().render() == "az:sims=100:autopay:oppnodes:oppfull:morekinds"
    assert PolicySpec(keys=(), sims=None).render() == "az"


ROUND_TRIP = [
    PolicySpec(),
    PolicySpec(sims=None, keys=()),
    PolicySpec(sims=200, explore=True, cpuct=0.25, topk=6, gonet="/r/gen3/best.dzgw"),
    PolicySpec(sims=100, closedlist=True, cpuct=1.0, topk=6, gonet="v.dzgw"),
    PolicySpec(pcr=0.25),
    PolicySpec(pcr=0.25, cheap=32),
    PolicySpec(pcr=1.0, cheap=4, pcrk=(("priority", 0.65), ("attackers", 0.5), ("optional", 0.0))),
    PolicySpec(pcr=0.5, pcrk={"blockers": 1.0}),
    PolicySpec(keys=("autopay", "nonoise", "skippass")),
    PolicySpec(extra=("kinds=priority,attackers", "worlds=4", "fpu=0.1")),
    PolicySpec(extra=("skippass=2", "remote=unix:/tmp/dzg.sock")),
    PolicySpec(keys=(), extra=("kinds=priority", "morekinds")),
    PolicySpec(keys=("autopay",), extra=("worlds=4", "nonoise", "skippass=2", "skippass")),
    PolicySpec(kind="prior", sims=None, gonet="/n.dzgw"),
    PolicySpec(kind="prior", gonet="/n.dzgw"),
    PolicySpec(kind="bot"),
    PolicySpec(kind="bot", keys=("autopay",)),
    PolicySpec(kind="sbtactical", keys=("mull",)),
    PolicySpec(kind="random"),
    PolicySpec(cpuct=1e-5, topk=1, sims=1),
    PolicySpec(decided=0.9),
    PolicySpec(decided=0.6, decidedn=1),
    PolicySpec(decided=0.95, decidedn=2, decidedsims=10, decidedw=0.0),
    PolicySpec(decided=0.75, decidedw=1.0),
    PolicySpec(decided=0.5000001, decidedsims=1, decidedw=1e-7),
    PolicySpec(forced=2.0),
    PolicySpec(forced=1e-3),
    PolicySpec(sims=8, explore=True, closedlist=True, pcr=0.5, cheap=4, pcrk={"priority": 0.65}, decided=0.9,
               decidedn=3, decidedsims=40, decidedw=0.1, forced=2.0, topk=6, gonet="/n.dzgw",
               extra=("worlds=4", "nonoise")),
]


@pytest.mark.parametrize("spec", ROUND_TRIP, ids=lambda s: s.render())
def test_parse_render_round_trip(spec):
    assert PolicySpec.parse(spec.render()) == spec
    assert PolicySpec.parse(spec.render()).render() == spec.render()
    assert hash(PolicySpec.parse(spec.render())) == hash(spec)


@pytest.mark.parametrize("s", [
    "az:sims=200:explore:autopay:oppnodes:oppfull:morekinds:cpuct=0.25:topk=6:gonet=/a/b.dzgw",
    "az:sims=100:autopay:oppnodes:oppfull:morekinds:closedlist:topk=6:gonet=B7.dzgw",
    "az:sims=8:explore:pcr=0.25:cheap=4",
    "az:sims=8:morekinds:pcr=0.5:pcrk=priority.0.65,attackers.0.5,optional.0.0",
    "prior:gonet=/x.dzgw",
    "bot", "sbtactical", "random", "bot:autopay",
    "az:sims=8:explore:closedlist:decided=0.6:decidedn=1",
    "az:sims=8:explore:pcr=0.5:cheap=4:decided=0.9:decidedn=3:decidedsims=40:decidedw=0.1:forced=2.0:gonet=/a.dzgw",
    "az:sims=8:forced=0.5",
])
def test_canonical_strings_render_back(s):
    assert PolicySpec.parse(s).render() == s


def test_parse_fields():
    p = PolicySpec.parse("az:sims=200:explore:autopay:oppnodes:oppfull:morekinds:cpuct=0.25:topk=6:gonet=/a/b.dzgw")
    assert (p.kind, p.sims, p.explore, p.keys) == ("az", 200, True, FULL)
    assert (p.cpuct, p.topk, p.gonet) == (0.25, 6, "/a/b.dzgw")
    assert p.closedlist is False and p.pcr is None and p.extra == ()
    # the order of the fields in the string does not matter to the fork; parse puts them in their fields (keys
    # keep their own order)
    q = PolicySpec.parse("az:gonet=/a/b.dzgw:topk=6:cpuct=0.25:autopay:oppnodes:oppfull:morekinds:explore:sims=200")
    assert q == p and q.render() == p.render()
    assert PolicySpec.parse("az").sims is None          # the fork's default (100) is not written
    assert PolicySpec.parse("az:kinds=all:worlds=4").extra == ("kinds=all", "worlds=4")


@pytest.mark.parametrize("s", [
    "az:sims=10:kinds=priority:morekinds",              # the fork reads left to right: full kinds win
    "az:sims=10:skippass=3:skippass",                   # SkipPass 1, not 3
    "az:sims=10:autopay:worlds=4:oppnodes",
])
def test_parse_keeps_bare_keys_after_an_extra_in_place(s):
    """A bare key after an extra entry may override it in the fork (morekinds after kinds=, skippass after
    skippass=N): parse keeps it in extra, in its place, so a re-render (dz play --sims) means the same."""
    import dataclasses
    p = PolicySpec.parse(s)
    assert p.render() == s and PolicySpec.parse(p.render()) == p
    assert dataclasses.replace(p, sims=100).render() == s.replace("sims=10", "sims=100")
    q = PolicySpec.parse("az:sims=10:kinds=priority:morekinds")
    assert (q.keys, q.extra) == ((), ("kinds=priority", "morekinds"))
    assert PolicySpec.parse("az:morekinds:kinds=priority").keys == ("morekinds",)


def test_parse_topk_0_is_off():
    assert PolicySpec.parse("az:sims=10:topk=0").topk is None              # the fork: 0 = no top-k cut


def test_pcr_rendering():
    s = PolicySpec(pcr=0.25, cheap=32, pcrk=(("priority", 0.65), ("attackers", 0.5)), topk=6, gonet="/n.dzgw")
    assert ":pcr=0.25:cheap=32:pcrk=priority.0.65,attackers.0.5:" in s.render()
    assert s.render().endswith(":gonet=/n.dzgw")
    # off: neither pcr nor cheap nor pcrk is rendered (or kept)
    for pcr in (None, 0, 0.0):
        off = PolicySpec(pcr=pcr, cheap=40, pcrk={"priority": 0.5})
        assert "pcr" not in off.render() and "cheap" not in off.render()
        assert (off.pcr, off.cheap, off.pcrk) == (None, None, ())
        assert off == PolicySpec()
    # the fork's kind=P form parses too
    assert PolicySpec.parse("az:pcr=0.4:cheap=10:pcrk=priority=0.65,blockers=1").pcrk == (("priority", 0.65),
                                                                                          ("blockers", 1.0))
    assert PolicySpec(pcr=0.5, pcrk={"priority": 1}).pcrk == (("priority", 1.0),)


def test_decided_and_forced_rendering():
    s = PolicySpec(decided=0.95, decidedn=2, decidedsims=10, decidedw=0.25, forced=1.5, pcr=0.5, topk=6,
                   gonet="/n.dzgw", extra=("worlds=4",))
    assert s.render() == ("az:sims=100:autopay:oppnodes:oppfull:morekinds:topk=6:pcr=0.5:decided=0.95:decidedn=2:"
                          "decidedsims=10:decidedw=0.25:forced=1.5:gonet=/n.dzgw:worlds=4")
    # None leaves the fork's default (decidedn 3, decidedsims 40, decidedw 0.1) unwritten
    assert PolicySpec(decided=0.9).render() == "az:sims=100:autopay:oppnodes:oppfull:morekinds:decided=0.9"
    # off: decided None or 0 keeps none of its keys; forced 0 is None
    for d in (None, 0, 0.0):
        off = PolicySpec(decided=d, decidedn=5, decidedsims=7, decidedw=0.5, forced=0)
        assert (off.decided, off.decidedn, off.decidedsims, off.decidedw, off.forced) == (None,) * 5
        assert off == PolicySpec() and "decided" not in off.render() and "forced" not in off.render()
    # the fork's bare keys are its defaults, rendered explicitly; "key=" is bare too (floatOr)
    p = PolicySpec.parse("az:sims=8:decided:forced")
    assert (p.decided, p.forced) == (DECIDED_DEFAULT, FORCED_DEFAULT) == (0.9, 2.0)
    assert p.render() == "az:sims=8:decided=0.9:forced=2.0" and PolicySpec.parse("az:decided=:forced=") == \
        PolicySpec.parse("az:decided:forced")
    assert PolicySpec.parse("az:decided=0:forced=0") == PolicySpec.parse("az")
    assert PolicySpec.parse("az:decided=0.7:decided=0.8").decided == 0.8        # later wins, as in the fork
    # integers and numeric strings are taken as numbers
    q = PolicySpec(decided="0.8", decidedn=2.0, decidedsims="5", decidedw=1, forced=3)
    assert (q.decided, q.decidedn, q.decidedsims, q.decidedw, q.forced) == (0.8, 2, 5, 1.0, 3.0)
    # no-search kinds drop them (as they drop pcr); prior refuses them (az keys in the fork)
    assert PolicySpec(kind="bot", decided=0.9, decidedn=2, forced=2).render() == "bot"
    for kw in (dict(decided=0.9), dict(forced=2.0)):
        with pytest.raises(ValueError, match="az keys"):
            PolicySpec(kind="prior", gonet="/n.dzgw", **kw)
    assert PolicySpec(kind="prior", gonet="/n.dzgw", decided=0, forced=0).render() == "prior:sims=100:autopay:" \
        "oppnodes:oppfull:morekinds:gonet=/n.dzgw"


@pytest.mark.parametrize("kw", [
    dict(decided=0.5), dict(decided=1.0), dict(decided=1.5), dict(decided=-0.9), dict(decided=float("nan")),
    dict(decided=float("inf")), dict(decided=True), dict(decided="high"),
    dict(decided=0.9, decidedn=0), dict(decided=0.9, decidedn=1.5), dict(decided=0.9, decidedn=True),
    dict(decided=0.9, decidedsims=0), dict(decided=0.9, decidedw=-0.1), dict(decided=0.9, decidedw=1.01),
    dict(decided=0.9, decidedw=float("nan")),
    dict(forced=-1.0), dict(forced=float("inf")), dict(forced=float("nan")), dict(forced=True),
    dict(extra=("decided=0.9",)), dict(extra=("worlds=4", "forced")), dict(extra=("decidedw=0.1",)),
])
def test_invalid_decided_and_forced(kw):
    with pytest.raises(ValueError):
        PolicySpec(**kw)


@pytest.mark.parametrize("s", [
    "az:decidedn=3", "az:decidedsims=40", "az:decidedw=0.1",           # policy.go: they need decided=H
    "az:decided=0:decidedn=3", "az:decided:decided=0:decidedw=0.5",
    "az:decided=1", "az:decided=0.5", "az:decided=x", "az:decided:decidedn=x", "az:decided:decidedn",
    "az:forced=-2", "az:forced=inf", "az:forced=x", "prior:gonet=/n.dzgw:forced", "prior:gonet=/n.dzgw:decided",
])
def test_parse_refuses_what_the_fork_refuses_of_decided_and_forced(s):
    with pytest.raises(ValueError):
        PolicySpec.parse(s)


@pytest.mark.parametrize("kw", [
    dict(kind="mcts"),
    dict(gonet="/a:b.dzgw"), dict(gonet=""),
    dict(pcr=1.5), dict(pcr=-0.1), dict(pcr=float("nan")),
    dict(pcr=0.5, cheap=0),
    dict(pcr=0.5, pcrk={"casting": 0.5}), dict(pcr=0.5, pcrk={"priority": 1.5}),
    dict(pcr=0.5, pcrk=(("priority", 0.5), ("priority", 0.6))),
    dict(sims=0), dict(sims=2.5), dict(topk=0), dict(cpuct=-1.0), dict(cpuct=float("inf")),
    dict(keys=("autopay", "cpuct=1")), dict(keys=("explore",)), dict(keys=("frobnicate",)),
    dict(extra=("cpuct=1",)), dict(extra=("autopay",)), dict(extra=("closedlist",)), dict(extra=("",)),
    dict(extra=("remote=unix:/s", "worlds=4")), dict(extra=("net=a:b",)),
    dict(kind="prior", gonet="/n.dzgw", closedlist=True), dict(kind="prior", gonet="/n.dzgw", pcr=0.5),
])
def test_invalid(kw):
    with pytest.raises(ValueError):
        PolicySpec(**kw)


@pytest.mark.parametrize("s", ["az::sims=1", "az:sims=x", "az:pcr=half", "az:pcr=0.5:pcrk=priority",
                               "az:pcr=0.5:pcrk=priority.x", "mcts:sims=1"])
def test_parse_invalid(s):
    with pytest.raises(ValueError):
        PolicySpec.parse(s)


def test_no_search_kinds_drop_search_fields():
    for kind in ("bot", "sbtactical", "random"):
        s = PolicySpec(kind=kind, sims=100, explore=True, closedlist=True, cpuct=0.25, topk=6, gonet="/n.dzgw",
                       pcr=0.5)
        assert s.render() == kind
        assert s == PolicySpec.parse(kind)
    assert PolicySpec(kind="bot", keys=("autopay", "mull")).render() == "bot:autopay:mull"


def test_frozen_and_kind_names():
    s = PolicySpec()
    with pytest.raises(Exception):
        s.sims = 5
    assert KIND_NAMES == ("priority", "attackers", "blockers", "target", "modes", "optional", "choose", "multitarget")
    import dataclasses
    assert dataclasses.replace(s, sims=50).render().startswith("az:sims=50:")
