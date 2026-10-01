"""Closed decklists for experiment #4's games (docs/017 §3.4): a small HTTP service that samples the
opponent's hidden cards from belief.py's OpponentModel, the model behind every benchmark world.

The game player (mage.player.ai.BenchPlayer, through the bridge's play op) asks once per decision:

    POST /sample {"exclude": "<the opponent's deck file stem>", "seen": {name: copies}, "zones":
                  {name: copies}, "hand": <hand size>, "k": 8, "seed": <int>}
    ->   {"samples": [{"deck": [names], "hand": [names], "hidden": [names]}, ...]}

  seen    every card of the opponent's the player has seen (now: the cards in its known zones)
  zones   its cards in known zones: battlefield (cards it owns), graveyard, exile, its spells on the stack
  hidden  the sampled deck minus `zones`: what its hand and library hold in that world; `hand` is
          the model's draw of `hand` cards from it

`exclude` names the deck being played against (FDN_top_00123_WB): its draft leaves the belief pool,
or the posterior would find the real list again. The stem -> draft map comes from the deck
extraction's decks.jsonl (data/deckgen/<pool>/decks.jsonl; tools/extract_decks.py). A model per
excluded draft is cached.

    python tools/imitation_scale/belief_server.py --port 50070 [--decks-jsonl data/deckgen/FDN_PremierDraft_wr60/decks.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter, OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from draftzero.gameplay.belief import OpponentModel  # noqa: E402
from draftzero.gameplay.ids import Ids  # noqa: E402

DECKS_JSONL = REPO / "data" / "deckgen" / "FDN_PremierDraft_wr60" / "decks.jsonl"


class Beliefs:
    """OpponentModels keyed by the excluded draft (an LRU of `size`), and the stem -> draft map."""

    def __init__(self, decks_jsonl: Path, size: int = 64):
        self.draft_of = {}
        if decks_jsonl.exists():
            for line in decks_jsonl.read_text().splitlines():
                d = json.loads(line)
                if d.get("dck_file"):
                    self.draft_of[Path(d["dck_file"]).stem] = d["draft_id"]
        self.ids = Ids.load("FDN")
        self.size = size
        self.models: OrderedDict[str, OpponentModel] = OrderedDict()
        self.lock = threading.Lock()

    def model(self, exclude: str | None) -> OpponentModel:
        draft = self.draft_of.get(exclude or "", "")
        if exclude and not draft:
            raise KeyError(f"no draft for deck '{exclude}' in the decks.jsonl map: the real list would stay in the pool")
        with self.lock:
            m = self.models.get(draft)
            if m is not None:
                self.models.move_to_end(draft)
                return m
        m = OpponentModel.load(ids=self.ids, exclude_drafts=[draft] if draft else ())
        with self.lock:
            self.models[draft] = m
            while len(self.models) > self.size:
                self.models.popitem(last=False)
        return m

    def sample(self, req: dict) -> list[dict]:
        m = self.model(req.get("exclude"))
        seen, zones = Counter(req.get("seen") or {}), Counter(req.get("zones") or {})
        rng = np.random.default_rng(int(req.get("seed", 0)) & 0xFFFFFFFFFFFFFFFF)
        out = []
        for _ in range(int(req.get("k", 8))):
            deck, hand = m.sample(None, seen, int(req.get("hand", 0)), known_zone_cards=zones, rng=rng)
            hidden = Counter(deck) - zones
            out.append({"deck": deck, "hand": hand, "hidden": sorted(hidden.elements())})
        return out


def serve(port: int, beliefs: Beliefs) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"ok": True, "decks": len(beliefs.draft_of)}) if self.path == "/healthz" else self._send(404, {})

        def do_POST(self):
            if self.path != "/sample":
                return self._send(404, {"error": "unknown path"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, {"samples": beliefs.sample(req)})
            except Exception as e:  # noqa: BLE001 - the caller sees the error and plays on
                self._send(400, {"error": f"{type(e).__name__}: {e}"[:500]})

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=50070)
    ap.add_argument("--decks-jsonl", type=Path, default=DECKS_JSONL)
    a = ap.parse_args(argv)
    beliefs = Beliefs(a.decks_jsonl)
    print(f"belief service on 127.0.0.1:{a.port}: {len(beliefs.draft_of)} decks mapped to drafts", flush=True)
    srv = serve(a.port, beliefs)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
