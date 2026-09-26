"""Fetch oracle text for every card in a set from Scryfall, for classifying targets.

    python tools/fetch_oracle.py --set FDN --extra assets/vocab/FDN_SPG_extra_cards.txt

Writes data/scryfall/<SET>_oracle.json: card name -> {oracle_text, type_line, mana_cost}.
Double-faced and adventure cards are stored under the full name and under each face's name,
because XMage logs "Cast <face name>".

Local analysis input only (data/ is gitignored). Card text is Wizards of the Coast's; this
uses Scryfall's public API per its guidelines: a User-Agent, and a pause between requests.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.scryfall.com"
HEADERS = {"User-Agent": "draftzero/0.1 (github.com/danieljbrooks/draft-zero)", "Accept": "application/json"}


def get(url: str) -> dict:
    time.sleep(0.12)                     # Scryfall asks for 50-100 ms between requests
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
        return json.load(r)


def entries(card: dict) -> dict[str, dict]:
    out = {}
    faces = card.get("card_faces") or []
    text = card.get("oracle_text") or "\n//\n".join(f.get("oracle_text", "") for f in faces)
    base = {"oracle_text": text, "type_line": card.get("type_line", ""), "mana_cost": card.get("mana_cost", "")}
    out[card["name"]] = base
    for f in faces:                      # "Cast <face>" is how XMage logs one half of a card
        out.setdefault(f["name"], {"oracle_text": f.get("oracle_text", ""),
                                   "type_line": f.get("type_line", ""), "mana_cost": f.get("mana_cost", "")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="FDN")
    ap.add_argument("--extra", help="file of extra card names (one per line), e.g. bonus-sheet cards")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cards: dict[str, dict] = {}
    url = f"{API}/cards/search?" + urllib.parse.urlencode({"q": f"set:{a.set.lower()}", "unique": "cards"})
    while url:
        page = get(url)
        for c in page["data"]:
            cards.update(entries(c))
        url = page.get("next_page") if page.get("has_more") else None
    n_set = len(cards)
    missing = []
    if a.extra:
        for name in (l.strip() for l in open(a.extra) if l.strip()):
            if name in cards:
                continue
            try:
                cards.update(entries(get(f"{API}/cards/named?" + urllib.parse.urlencode({"exact": name}))))
            except Exception:
                missing.append(name)
    out = Path(a.out or f"data/scryfall/{a.set}_oracle.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cards, indent=1, sort_keys=True))
    print(f"{len(cards)} names ({n_set} from set:{a.set.lower()}, {len(cards) - n_set} extra) -> {out}"
          + (f"; not found: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
