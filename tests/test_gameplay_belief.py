"""Mirrored-pair matcher and opponent belief models (draftzero.gameplay.pairs / belief, WP1 + WP3).

The sampler and determinization tests run on a synthetic deck pool and on specs rebuilt from the
committed 17lands fixture (tests/fixtures/gameplay/fdn_premier_rows.csv.gz, CC BY 4.0); the matcher
test on a synthetic replay excerpt written to tmp_path. The real-data tests read the first rows of
the 17lands FDN replay file and the deck pool, and skip without them. No pair list is committed:
mirrored pairs stay in data/gameplay/ (gitignored).
"""
import csv
import gzip
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from draftzero.gameplay import belief as bl
from draftzero.gameplay import pairs as pm
from draftzero.gameplay import replay
from draftzero.gameplay.ids import Ids

FIXTURE = Path(__file__).parent / "fixtures" / "gameplay" / "fdn_premier_rows.csv.gz"
REPLAY = replay.replay_path("FDN", "PremierDraft")
GAME_DATA = REPLAY.parent / "game_data_public.FDN.PremierDraft.csv.gz"


# --- a synthetic pool: three colour pairs, 17 lands + 23 spells per deck --------------------------

BASICS = {"Plains": "W", "Island": "U", "Swamp": "B"}


def _spells(color: str) -> list[str]:
    return [f"{color}crea{i}" for i in range(12)] + [f"{color}spell{i}" for i in range(8)]


CARDS = sorted(BASICS) + [c for col in "WUB" for c in _spells(col)] + ["Signet"]
KINDS = {**{b: bl.KIND_LAND for b in BASICS},
         **{c: bl.KIND_CREATURE for c in CARDS if "crea" in c},
         **{c: bl.KIND_OTHER for c in CARDS if "spell" in c or c == "Signet"}}
MVS = {**{c: 1 + int(c[-1]) % 5 for c in CARDS if c[-1].isdigit()}, "Signet": 2}
COLS = {**BASICS, **{c: c[0] for c in CARDS if c[0] in "WUB" and c not in BASICS}}


def _deck(c1: str, c2: str, k: int) -> Counter:
    """A 40-card two-colour deck; k shifts which spells it plays."""
    land = {"W": "Plains", "U": "Island", "B": "Swamp"}
    d = Counter({land[c1]: 9, land[c2]: 8})
    pool = _spells(c1)[k % 4:][:12] + _spells(c2)[k % 3:][:10] + ["Signet"]
    d.update(pool)
    assert sum(d.values()) == 40, sum(d.values())
    return d


def _pool():
    decks = [(_deck(a, b, k), a + b, "") for a, b in (("W", "U"), ("W", "B"), ("U", "B")) for k in range(6)]
    return bl.DeckPool.from_decks(decks, CARDS), bl.CardMeta.simple(CARDS, KINDS, MVS, COLS, BASICS)


@pytest.fixture(scope="module")
def synth():
    pool, meta = _pool()
    return {"deck": bl.OpponentModel(pool, meta, alpha=0.1), "freq": bl.FrequencyModel(pool, meta),
            "uniform": bl.UniformPoolModel(pool, meta)}


@pytest.mark.parametrize("name", ["deck", "freq", "uniform"])
@pytest.mark.parametrize("colors", ["WU", None])
def test_sampler_invariants(synth, name, colors):
    m = synth[name]
    seen = Counter({"Plains": 3, "Island": 2, "Wcrea3": 1, "Ucrea5": 2, "Uspell1": 1})
    zones = Counter({"Plains": 3, "Island": 2, "Wcrea3": 1, "Ucrea5": 1, "Bcrea0": 1})  # Bcrea0: a stolen card, say
    for seed in range(20):
        deck, hand = m.sample(colors, seen, 6, zones, np.random.default_rng(seed))
        D = Counter(deck)
        assert len(deck) >= 40 and len(hand) == 6
        forced = seen | zones
        assert not forced - D, "a seen or zone card is missing from the deck"
        # the hand comes out of the deck minus the known zones: a seen card in no zone (the second
        # Ucrea5, Uspell1: bounced, say) may be in it
        assert not (Counter(hand) + zones) - D, "a card is used more times than the deck has"
        assert sum(D.values()) - sum(zones.values()) - len(hand) >= 1, "no library left"
        if name == "deck":
            assert len(deck) == 40                 # real 40-card decks, swaps keep the size


def test_force_include_swaps_same_kind(synth):
    m = synth["deck"]
    # a W/U deck that never plays Ucrea11: forcing two copies must replace creatures, not lands
    seen = Counter({"Ucrea11": 2})
    for seed in range(10):
        deck, _ = m.sample("WU", seen, 5, seen, seed)
        D = Counter(deck)
        assert D["Ucrea11"] >= 2 and len(deck) == 40
        assert sum(k for c, k in D.items() if KINDS[c] == bl.KIND_LAND) == 17
    # one swap on a known deck: a creature goes out for the creature, lands untouched
    d0 = m.pool.counts[0].astype(np.int16)
    forced, _ = m.vec(Counter({"Bcrea7": 1}))
    d1, swaps = m.force_include(d0, forced, m.group_freq[m.gbits[0]])
    assert swaps == 1 and d1.sum() == d0.sum()
    kind = m.meta.kind
    assert (d1[kind == bl.KIND_CREATURE].sum(), d1[kind == bl.KIND_LAND].sum()) == (
        d0[kind == bl.KIND_CREATURE].sum(), d0[kind == bl.KIND_LAND].sum())


def test_deck_model_follows_colours_and_seen_cards(synth):
    m = synth["deck"]
    # colours: W/B revealed -> no Islands unless seen
    for seed in range(10):
        deck, hand = m.sample("WB", Counter({"Swamp": 1}), 7, None, seed)
        assert "Island" not in deck and not any(c.startswith("U") for c in deck)
    # seen cards pick the deck: Wcrea0..2 are only in the k % 4 == 0 W/x decks
    idx, w = m.posterior("WU", m.vec(Counter({"Wcrea0": 1, "Wcrea1": 1, "Wcrea2": 1}))[0])
    has = m.pool.counts[idx, m.index["Wcrea0"]] > 0
    assert w[has].sum() > 0.9
    # without colours the seen cards' colours decide
    idx, w = m.posterior(None, m.vec(Counter({"Bcrea4": 1, "Ucrea4": 1}))[0])
    assert all(m.gbits[i] == bl.color_bits("UB") for i in idx[w > 1e-3])


def test_marginals_are_distributions(synth):
    seen = Counter({"Plains": 2, "Wcrea3": 1})
    for m in synth.values():
        p = m.marginal("WU", seen, seen)
        assert p.shape == (len(CARDS),) and abs(p.sum() - 1) < 1e-6 and (p >= 0).all()
        assert p[m.index["Bcrea3"]] < 1e-6        # off-colour
    # the deck model puts the seen copies out of the hidden part: fewer Wcrea3 than the prior
    m = synth["deck"]
    assert m.marginal("WU", seen, seen)[m.index["Wcrea3"]] < m.marginal("WU", Counter({"Plains": 2}), None)[m.index["Wcrea3"]]


@pytest.mark.parametrize("name", ["deck", "freq", "uniform"])
def test_cards_in_no_known_zone_can_be_in_the_hand(synth, name):
    """A card the opponent showed that is in no known zone (it left the battlefield for an
    unrecorded zone) is somewhere hidden: the hand is drawn from the deck minus the known zones.
    One reconstruct marks as 'likely in hand' (it left as the hand count rose) is in the hand with
    probability P_LIKELY_IN_HAND; the marginal says the same."""
    m = synth[name]
    zones = Counter({"Plains": 2, "Island": 2, "Wcrea3": 1})
    seen = zones + Counter({"Ucrea5": 1})                    # Ucrea5 was bounced
    likely = (Counter({"Ucrea5": 1}), Counter())
    held = Counter()
    for seed in range(300):
        for how, lost in (("plain", None), ("likely", likely), ("rest", (Counter(), Counter({"Ucrea5": 1})))):
            deck, hand = m.sample("WU", seen, 5, zones, seed, lost=lost)
            D = Counter(deck)
            assert not (seen | zones) - D and not (Counter(hand) + zones) - D and len(hand) == 5
            held[how] += "Ucrea5" in hand
    assert held["plain"] > 0 and held["rest"] > 0                 # no longer ruled out
    assert 0.8 <= held["likely"] / 300 <= 0.97 and held["likely"] > 2 * held["plain"]
    i = m.index["Ucrea5"]
    p_likely = m.marginal("WU", seen, zones, lost=likely, hand_count=5)
    p_plain = m.marginal("WU", seen, zones)
    assert abs(p_likely.sum() - 1) < 1e-6 and abs(p_plain.sum() - 1) < 1e-6
    assert p_likely[i] >= bl.P_LIKELY_IN_HAND / 5 > p_plain[i] > 0
    # hand_count 1 and three likely copies: they cannot all be in it
    three = (Counter({"Ucrea5": 1, "Ucrea6": 1, "Ucrea7": 1}), Counter())
    s3 = seen + three[0]
    for seed in range(20):
        _, hand = m.sample("WU", s3, 1, zones, seed, lost=three)
        assert len(hand) == 1
    assert abs(m.marginal("WU", s3, zones, lost=three, hand_count=1).sum() - 1) < 1e-6


def test_seed_determinism(synth):
    for m in synth.values():
        a = m.sample("WU", Counter({"Plains": 1}), 5, None, np.random.default_rng(7))
        b = m.sample("WU", Counter({"Plains": 1}), 5, None, np.random.default_rng(7))
        assert a == b
    outs = {tuple(tuple(x) for x in synth["deck"].sample("WU", Counter(), 7, None, s)) for s in range(12)}
    assert len(outs) > 1


def test_retention_weighted_hand(synth):
    pool, meta = _pool()
    # lands rarely stay in hand, 5-drops usually do
    cells = {bl.HandRetention.key(k, mv, 4): [1 if k == bl.KIND_LAND else 15 * mv, 100]
             for k in (0, 1, 2) for mv in range(7)}
    ret = bl.HandRetention(cells)
    assert bl.HandRetention.from_json(ret.to_json()).cells == ret.cells
    plain, weighted = bl.OpponentModel(pool, meta, alpha=0.1), bl.OpponentModel(pool, meta, alpha=0.1, retention=ret)
    seen = Counter({"Plains": 2, "Island": 2})
    lands = Counter()
    for seed in range(40):
        for name, m in (("plain", plain), ("weighted", weighted)):
            deck, hand = m.sample("WU", seen, 5, seen, seed, turn=4)
            assert len(hand) == 5 and not (Counter(hand) + seen) - Counter(deck)
            lands[name] += sum(c in BASICS for c in hand)
    assert lands["weighted"] < lands["plain"] / 3
    p = weighted.marginal("WU", seen, seen, turn=4)
    assert abs(p.sum() - 1) < 1e-6 and p[weighted.index["Plains"]] < plain.marginal("WU", seen, seen)[weighted.index["Plains"]]
    # without a turn the draw stays uniform
    assert weighted.sample("WU", seen, 5, seen, 3) == plain.sample("WU", seen, 5, seen, 3)


def test_turns_done():
    from draftzero.gameplay.statespec import PlayerState, StateSpec
    s = StateSpec(turn=6, activePlayer="B", phase="END", step="END_TURN", startingPlayer="A",
                  players={"A": PlayerState(), "B": PlayerState()})
    assert (bl.turns_done(s, "A"), bl.turns_done(s, "B")) == (3, 3)
    s.turn, s.activePlayer, s.phase, s.step = 7, "A", "PRECOMBAT_MAIN", "PRECOMBAT_MAIN"
    assert (bl.turns_done(s, "A"), bl.turns_done(s, "B")) == (3, 3)
    s.startingPlayer = None                     # derived from parity: A active on an odd turn started
    assert bl.turns_done(s, "B") == 3
    # the user on the draw: B started and took the odd turns
    s = StateSpec(turn=5, activePlayer="B", phase="END", step="END_TURN", startingPlayer="B",
                  players={"A": PlayerState(), "B": PlayerState()})
    assert (bl.turns_done(s, "A"), bl.turns_done(s, "B")) == (2, 3)
    s.phase, s.step = "COMBAT", "DECLARE_ATTACKERS"      # B's 3rd turn is not complete yet
    assert bl.turns_done(s, "B") == 2


def test_retention_odds_follow_the_card_axis(monkeypatch):
    # regression: the odds cache was keyed by id(meta) alone; a new CardMeta at a freed one's
    # address got the old card axis's odds (land odds for creatures). Address reuse depends on the
    # allocator, so it is simulated here: every meta gets the same id().
    monkeypatch.setattr(bl, "id", lambda x: 0, raising=False)
    cells = {bl.HandRetention.key(k, mv, 4): [1 if k == bl.KIND_LAND else 50, 100] for k in (0, 1, 2) for mv in range(7)}
    ret = bl.HandRetention(cells)
    cards = ["a", "b", "c"]
    for kind in (bl.KIND_LAND, bl.KIND_CREATURE, bl.KIND_LAND):
        o = ret.odds(bl.CardMeta.simple(cards, {c: kind for c in cards}, {}, {}), 4)
        assert (o < 0.1).all() if kind == bl.KIND_LAND else (o > 0.5).all(), kind


def test_determinize_exact_list_without_room(synth):
    # regression: an exact list whose known zones leave fewer hidden cards than the unknown hand
    # silently shrank the hand; now it is padded with a basic and marked as no longer exact
    from draftzero.gameplay.statespec import PlayerState, StateSpec
    deck = sorted(_deck("W", "U", 0).elements())
    B = PlayerState(name="B", decklist=deck, decklistSource="exact", graveyard=deck[:36], handUnknown=7)
    spec = StateSpec(turn=12, activePlayer="B", phase="END", step="END_TURN", startingPlayer="A",
                     players={"A": PlayerState(name="A", decklist=list(deck)), "B": B})
    for s in bl.determinize(spec, synth["deck"], 3, seed=5):
        sb = s.players["B"]
        assert len(sb.hand) == 7 and s.validate() == [] and not s.is_partial()
        assert sum((Counter(sb.decklist) - bl.zone_cards(s)).values()) >= 1, "no library left"
        assert not Counter(deck) - Counter(sb.decklist), "a card of the real list was dropped"
        assert sb.decklistSource == "belief" and "B.decklist" in s.provenance.sampled
        assert "B_decklist_padded" in s.provenance.flags


def test_zone_cards_follow_the_owner(synth):
    """A permanent is listed under its controller; its card is its owner's (Perm.owner). A card of
    B's that A controls is not in B's library, so it must never be drawn into B's hand; a card of
    A's that B stole is not B's; B's spells on the stack are out of its library too."""
    from draftzero.gameplay.statespec import Perm, PlayerState, StackItem, StateSpec
    deck = sorted(_deck("W", "U", 0).elements())
    assert Counter(deck)["Ucrea5"] == 1                          # B's only copy
    A = PlayerState(name="A", decklist=sorted(_deck("W", "B", 1).elements()),
                    battlefield=[Perm(name="Ucrea5", id="stolen", owner="B"), Perm(name="Plains", id="p")])
    B = PlayerState(name="B", decklist=deck, decklistSource="exact", handUnknown=7,
                    battlefield=[Perm(name="Bcrea3", id="took", owner="A"), Perm(name="Island", id="i")])
    spec = StateSpec(turn=8, activePlayer="A", startingPlayer="A", players={"A": A, "B": B},
                     stack=[StackItem("B", "Ucrea6")], enterMode="PRIORITY_HELD", priorityPlayer="A")
    assert spec.validate() == []
    assert bl.zone_cards(spec, "B") == Counter({"Ucrea5": 1, "Island": 1, "Ucrea6": 1})
    assert bl.zone_cards(spec, "A") == Counter({"Bcrea3": 1, "Plains": 1})
    for s in bl.determinize(spec, synth["deck"], 40, seed=2):              # the exact-list branch
        assert "Ucrea5" not in s.players["B"].hand and s.validate() == []
    B.decklistSource = "placeholder"
    # the belief branch; reconstruct's seen cards count A's Bcrea3 too (it was seen on B's side)
    for s in bl.determinize(spec, synth["deck"], 10, seed=2, opp_colors="WU", seen=Counter({"Bcrea3": 1})):
        assert s.validate() == [] and "Bcrea3" not in s.players["B"].decklist     # A's card, not B's


def test_determinize_puts_likely_cards_in_the_hand(synth):
    from draftzero.gameplay.statespec import PlayerState, StateSpec
    deck = sorted(_deck("W", "U", 0).elements())
    B = PlayerState(name="B", decklist=deck, decklistSource="exact", graveyard=["Wcrea4"], handUnknown=4)
    spec = StateSpec(turn=8, activePlayer="A", startingPlayer="A",
                     players={"A": PlayerState(name="A", decklist=list(deck)), "B": B})
    lost = (Counter({"Ucrea5": 1}), Counter())
    for exact in (True, False):
        B.decklistSource = "exact" if exact else "placeholder"
        outs = bl.determinize(spec, synth["deck"], 300, seed=5, opp_colors="WU", seen=Counter({"Ucrea5": 1}),
                              lost=lost)
        share = sum("Ucrea5" in s.players["B"].hand for s in outs) / len(outs)
        assert 0.75 <= share <= 0.95, (exact, share)                   # P_LIKELY_IN_HAND = 0.84
        assert all(s.validate() == [] and len(s.players["B"].hand) == 4 for s in outs)


# --- determinization on real reconstructed specs ------------------------------------------------

@pytest.fixture(scope="module")
def ids():
    return Ids.load()


@pytest.fixture(scope="module")
def fixture_games():
    return {g.row_index: g for g in replay.iter_games(FIXTURE)}


@pytest.fixture(scope="module")
def fixture_model(fixture_games, ids):
    # the fixture users' own decks are a (tiny) opponent pool
    decks = [(Counter(g.deck), g.meta.get("main_colors") or "", g.meta.get("splash_colors") or "")
             for g in fixture_games.values()]
    names = sorted({c for d, _, _ in decks for c in d})
    pool = bl.DeckPool.from_decks(decks, names)
    return bl.OpponentModel(pool, bl.CardMeta.from_ids(names, ids), alpha=0.3)


def test_deckpool_build_and_cache(fixture_games, tmp_path):
    # the parser behind the real pool (game_data_public, or the replay file's deck_ columns as here)
    import shutil
    shutil.copy(FIXTURE, tmp_path / "replay_data_public.FDN.PremierDraft.csv.gz")
    pool = bl.DeckPool.load("FDN", "PremierDraft", data_dir=tmp_path, cache_dir=tmp_path)
    assert (tmp_path / "deckpool_FDN_PremierDraft.npz").exists()
    built = {tuple(sorted((pool.cards[i], int(k)) for i, k in enumerate(row) if k)): int(m)
             for row, m in zip(pool.counts, pool.main)}
    # every game's deck, under its own main colours (a column/name misalignment would break this)
    assert built == {tuple(sorted(g.deck.items())): bl.color_bits(g.meta.get("main_colors"))
                     for g in fixture_games.values()}
    assert int(pool.games.sum()) == len(fixture_games)
    again = bl.DeckPool.load("FDN", "PremierDraft", data_dir=tmp_path, cache_dir=tmp_path)
    assert again.cards == pool.cards and (again.counts == pool.counts).all() and (again.games == pool.games).all()


def test_determinize_fixture_specs(fixture_games, fixture_model, ids):
    n = 0
    for g in fixture_games.values():
        for turn in g.decision_turns()[:6]:
            spec, seen, colors = bl.game_evidence(g, turn, ids)
            assert spec.is_partial() or spec.players["B"].handUnknown == 0
            # determinize's retention turn = the opponent turn whose snapshot the spec is (pair_views)
            p = g.prev_slot(turn)
            assert bl.turns_done(spec, "B") == (p.n if p is not None else 0)
            before = spec.to_dict()
            outs = bl.determinize(spec, fixture_model, 3, seed=11, opp_colors=colors, seen=seen)
            assert spec.to_dict() == before, "the input spec was modified"
            for s in outs:
                assert s.validate() == [] and not s.is_partial()
                A0, B0, B = spec.players["A"], spec.players["B"], s.players["B"]
                assert s.to_dict()["players"]["A"] == before["players"]["A"]
                assert B.decklistSource == "belief" and len(B.decklist) >= 40
                assert B.handUnknown == 0 and len(B.hand) == len(B0.hand) + B0.handUnknown
                assert not Counter(B0.hand) - Counter(B.hand)
                assert {"B.decklist"} <= set(s.provenance.sampled)
                assert ("B.hand" in s.provenance.sampled) == (B0.handUnknown > 0)
                assert "opp_decklist_placeholder" not in s.provenance.flags
                assert not (seen | bl.zone_cards(spec)) - Counter(B.decklist)
                assert A0.decklist == s.players["A"].decklist
            n += 1
    assert n >= 30


def test_retention_learned_from_fixture(fixture_games, fixture_model, ids):
    ret = bl.HandRetention.learn(fixture_games.values(), ids, fixture_model.index, fixture_model.meta)
    assert ret.cells and all(0 <= a <= b for a, b in ret.cells.values())
    # users hold few of their hidden lands: well under a third on average after turn 3
    lands = [v for k, v in ret.cells.items() if k[0] == bl.KIND_LAND and k[2] >= 3]
    assert sum(a for a, _ in lands) < sum(b for _, b in lands) / 3


def test_determinize_seeds_and_prefix(fixture_games, fixture_model, ids):
    g = fixture_games[4]
    spec, seen, colors = bl.game_evidence(g, 5, ids)
    a = bl.determinize(spec, fixture_model, 4, seed=3, opp_colors=colors, seen=seen)
    b = bl.determinize(spec, fixture_model, 4, seed=3, opp_colors=colors, seen=seen)
    c = bl.determinize(spec, fixture_model, 2, seed=3, opp_colors=colors, seen=seen)
    d = bl.determinize(spec, fixture_model, 4, seed=4, opp_colors=colors, seen=seen)
    j = lambda xs: [x.to_json() for x in xs]
    assert j(a) == j(b) and j(a)[:2] == j(c) and j(a) != j(d)


def test_determinize_keeps_exact_decklist(fixture_games, fixture_model, ids):
    g = fixture_games[0]
    true_deck = sorted(Counter(fixture_games[4].deck).elements())        # any real 40, as if from a pair
    spec, seen, colors = bl.game_evidence(g, 4, ids)
    zones = bl.zone_cards(spec)
    B = spec.players["B"]
    B.decklist = sorted((Counter(true_deck) | zones).elements())
    B.decklistSource = "exact"
    for s in bl.determinize(spec, fixture_model, 3, seed=1):
        assert s.players["B"].decklist == B.decklist and s.players["B"].decklistSource == "exact"
        assert s.provenance.sampled == (["B.hand"] if B.handUnknown else [])
        assert s.validate() == [] and not s.is_partial()


# --- the pair matcher on a synthetic excerpt ----------------------------------------------------

def _write_rows(path: Path, rows: list[dict]):
    fams = ["eot_user_lands_in_play", "eot_user_creatures_in_play", "eot_oppo_lands_in_play",
            "eot_oppo_creatures_in_play", "eot_user_cards_in_hand", "eot_oppo_cards_in_hand", "creatures_cast"]
    cols = ["draft_id", "game_time", "on_play", "won", "num_mulligans", "opp_num_mulligans", "num_turns"]
    cols += [f"{s}_turn_{n}_{f}" for s in ("user", "oppo") for n in (3, 5) for f in fams]
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])


def _game_rows(t: str, play_lands: str, draw_lands: str, crea: str, won_play: bool, dt: int = 2):
    """The same game from both sides: (on-play row, on-draw row)."""
    a = {"game_time": f"2024-11-20 10:{t}:00", "on_play": "True", "won": str(won_play), "num_mulligans": 0,
         "opp_num_mulligans": 1, "num_turns": 8,
         "user_turn_3_eot_user_lands_in_play": play_lands, "user_turn_3_eot_oppo_lands_in_play": draw_lands,
         "user_turn_3_eot_user_creatures_in_play": crea,
         "user_turn_5_eot_user_cards_in_hand": "1|2|3", "user_turn_5_creatures_cast": "77"}
    b = {"game_time": f"2024-11-20 10:{t}:{dt:02d}", "on_play": "False", "won": str(not won_play), "num_mulligans": 1,
         "opp_num_mulligans": 0, "num_turns": 8,
         "oppo_turn_3_eot_oppo_lands_in_play": "|".join(reversed(play_lands.split("|"))),
         "oppo_turn_3_eot_user_lands_in_play": draw_lands, "oppo_turn_3_eot_oppo_creatures_in_play": crea,
         "oppo_turn_5_eot_oppo_cards_in_hand": "3.0", "oppo_turn_5_creatures_cast": "77"}
    return a, b


def test_find_pairs_synthetic(tmp_path):
    g1a, g1b = _game_rows("00", "10|11|12", "20|21", "30", True)
    g2a, g2b = _game_rows("30", "10|11|13", "20|22", "", False, dt=40)
    solo, _ = _game_rows("45", "10|11|12", "20|21", "30", True)          # same board as game 1, 45 min later
    decoy = dict(g1b, won="True")                                       # right board, wrong result
    rows = [g1a, g2b, solo, decoy, g2a, g1b]
    path = tmp_path / "replay.csv.gz"
    _write_rows(path, rows)
    P, st = pm.find_pairs(path)
    assert sorted((q.row_a, q.row_b) for q in P) == [(0, 5), (4, 1)]
    assert {q.dt_s for q in P} == {2, 40}
    assert st["ambiguous"] == 0 and st["both_agree"] == 1.0 and st["checked_pairs"] == 2
    out = tmp_path / "pairs.jsonl"
    pm.write_pairs(P, out)
    assert pm.load_pairs(out) == P
    assert pm.partner_map(P)[5] == 0 and pm.partner_map(P)[0] == 5
    # a second candidate for the same on-play row is counted as ambiguous
    rows.append(dict(g1b, game_time="2024-11-20 10:00:05"))
    _write_rows(path, rows)
    assert pm.find_pairs(path)[1]["ambiguous"] == 1


def test_find_pairs_counts_draw_side_ambiguity(tmp_path):
    # regression: two on-play rows claiming one on-draw row were not counted (the first took it and
    # the second was left unmatched with no candidates)
    a, b = _game_rows("00", "10|11|12", "20|21", "30", True)
    path = tmp_path / "replay.csv.gz"
    _write_rows(path, [a, dict(a, game_time="2024-11-20 10:00:03"), b])
    P, st = pm.find_pairs(path)
    assert [(q.row_a, q.row_b) for q in P] == [(0, 2)]
    assert (st["ambiguous_play"], st["ambiguous_draw"], st["ambiguous"]) == (0, 1, 1)


def test_find_pairs_ignores_local_dst(tmp_path, monkeypatch):
    # regression: game_time was parsed as naive LOCAL time, so two rows 2 s apart across the
    # machine's DST fall-back were 3,602 s apart (outside the window) and never paired
    import time
    a, b = _game_rows("00", "10|11|12", "20|21", "30", True)
    a["game_time"], b["game_time"] = "2024-11-03 01:59:59", "2024-11-03 02:00:01"
    path = tmp_path / "replay.csv.gz"
    _write_rows(path, [a, b])
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        P, _ = pm.find_pairs(path)
    finally:
        monkeypatch.undo()
        time.tzset()
    assert [(q.row_a, q.row_b, q.dt_s) for q in P] == [(0, 1, 2)]


# --- real data -----------------------------------------------------------------------------------

@pytest.mark.skipif(not REPLAY.exists(), reason="17lands FDN replay file not downloaded")
def test_real_pairs_and_belief(ids, tmp_path):
    P, st = pm.find_pairs(REPLAY, stop=5000)
    assert len(P) >= 50 and st["ambiguous"] == 0 and st["both_agree"] >= 0.999
    full = pm.pairs_path()
    if full.exists():                     # the full-file run found the same games among these rows
        early = {(q.row_a, q.row_b) for q in pm.load_pairs(full) if max(q.row_a, q.row_b) < 5000}
        assert {(q.row_a, q.row_b) for q in P} == early
    if not (GAME_DATA.exists() or (pm.OUT_DIR / "deckpool_FDN_PremierDraft.npz").exists()):
        pytest.skip("no deck pool source")
    P = P[:12]
    games = replay.read_games({r for q in P for r in (q.row_a, q.row_b)}, REPLAY)
    views = []
    for q in P:
        a, b = games[q.row_a], games[q.row_b]
        assert a.won != b.won and a.on_play and not b.on_play
        views += bl.pair_views(a, b, (3, 5, 7), ids) + bl.pair_views(b, a, (3, 5, 7), ids)
    assert len(views) >= 30
    # the opponent's hand count in one row is the other row's own hand
    assert sum(v.hand_count == sum(v.true_hand.values()) for v in views) >= 0.95 * len(views)
    # the evaluation's opponent turn and determinize's (from the spec) are the same half-turn
    assert all(v.opp_turns == bl.turns_done(v.spec, "B") for v in views)
    pool = bl.DeckPool.load().without_drafts({g.meta["draft_id"] for g in games.values()})
    meta = bl.CardMeta.from_ids(pool.cards, ids)
    deck, unif = bl.OpponentModel(pool, meta, alpha=1.0), bl.UniformPoolModel(pool, meta)
    rd = bl.run_model(deck, views, 4, 0)["all"]
    ru = bl.run_model(unif, views, 4, 0)["all"]
    assert rd["ll_per_card"] > ru["ll_per_card"] and rd["recall@K"] > ru["recall@K"]
    assert rd["deck_overlap"] > ru["deck_overlap"]
    for v in views[:10]:
        s = bl.determinize(v.spec, deck, 1, seed=0, opp_colors=v.colors, seen=v.seen)[0]
        assert s.validate() == [] and not s.is_partial()
