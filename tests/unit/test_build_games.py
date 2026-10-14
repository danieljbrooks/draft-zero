"""draftzero.build.games on a fake dzgorge (tests/unit/conftest.py): explicit deck pairs played through Gorge.play
in chunks, a hung game replayed alone or skipped, and a resumed OUT file that plays only what is missing."""
import io
import json

from draftzero.build import games as G


def _inputs(tmp_path, n):
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text("".join(f"d{i}\te{i}\t{100 + i}\n" for i in range(n)))
    decks = tmp_path / "decks.jsonl"
    decks.write_text('{"name": "d0", "cards": []}\n')
    return str(decks), str(pairs), str(tmp_path / "games.jsonl")


def _rows(out):
    return [json.loads(l) for l in open(out) if l.strip()]


def test_plays_every_line_once_in_chunks(fake_dzgorge, tmp_path):
    decks, pairs, out = _inputs(tmp_path, 7)
    assert G.play(decks, pairs, out, sims=0, workers=2, chunk=3, gorge_root=fake_dzgorge.root, log=io.StringIO()) == 7
    rows = _rows(out)
    assert sorted(r["line"] for r in rows) == list(range(7)) and all(r["result"] in "AB" for r in rows)
    calls = [c for c in fake_dzgorge.calls() if c[0] == "play"]
    assert len(calls) == 3                                           # chunks of 3, 3, 1
    for c in calls:
        assert c[c.index("-a") + 1] == c[c.index("-b") + 1] == "bot" and "-single" in c
        assert c[c.index("-deckfile") + 1] == decks and c[c.index("-seed") + 1] == "1"
    # resumed: nothing left to play, no driver call
    assert G.play(decks, pairs, out, sims=0, workers=2, chunk=3, gorge_root=fake_dzgorge.root, log=io.StringIO()) == 0
    assert len(fake_dzgorge.calls()) == 3


def test_hung_game_replayed_alone(fake_dzgorge, tmp_path):
    decks, pairs, out = _inputs(tmp_path, 4)
    fake_dzgorge.env(FAKE_HANG_BUSY="1")                             # chunk line 1 hangs unless played alone
    assert G.play(decks, pairs, out, sims=0, workers=2, chunk=4, gorge_root=fake_dzgorge.root, chunk_secs=1,
                  probe_secs=30, log=io.StringIO()) == 4
    assert sorted(r["line"] for r in _rows(out)) == [0, 1, 2, 3]


def test_hung_game_skipped(fake_dzgorge, tmp_path):
    decks, pairs, out = _inputs(tmp_path, 3)
    fake_dzgorge.env(FAKE_HANG="0")                                  # line 0 hangs even alone
    log = io.StringIO()
    assert G.play(decks, pairs, out, sims=0, workers=2, chunk=3, gorge_root=fake_dzgorge.root, chunk_secs=1,
                  probe_secs=1, log=log) == 2
    rows = {r["line"]: r for r in _rows(out)}
    assert sorted(rows) == [0, 1, 2] and rows[0]["result"] == "skipped" and rows[0]["decks"] == ["d0", "e0"]
    assert "line 0" in log.getvalue()
