"""draftzero.build.games on the real driver (`pytest -m engine`): explicit pairs of the sample decks from a deckfile,
played once each through Gorge.play (-deckfile -pairs-file -single), then resumed."""
import io
import json
from pathlib import Path

import pytest

from draftzero.build import games as G

SAMPLE = Path(__file__).resolve().parents[2] / "assets" / "sample" / "decks"


@pytest.mark.engine
def test_build_games_plays_explicit_pairs(gorge, tmp_path):
    names = sorted(p.stem for p in SAMPLE.glob("*.json"))[:6]
    decks = tmp_path / "decks.jsonl"
    decks.write_text("".join(json.dumps(json.loads((SAMPLE / f"{n}.json").read_text())) + "\n" for n in names))
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text("".join(f"{a}\t{b}\t{7 + i}\n" for i, (a, b) in enumerate(zip(names[::2], names[1::2]))))
    out = tmp_path / "games.jsonl"
    n = G.play(str(decks), str(pairs), str(out), sims=0, workers=2, gorge_root=gorge.root, log=io.StringIO())
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert n == 3 and sorted(r["line"] for r in rows) == [0, 1, 2]
    assert all(r["result"] in ("A", "B", "draw", "stall") for r in rows)
    assert [r["decks"] for r in sorted(rows, key=lambda r: r["line"])] == [list(p) for p in zip(names[::2], names[1::2])]
    assert G.play(str(decks), str(pairs), str(out), sims=0, workers=2, gorge_root=gorge.root, log=io.StringIO()) == 0
