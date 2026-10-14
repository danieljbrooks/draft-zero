#!/usr/bin/env bash
# Build gorge at the pinned commit with DraftZero's driver (cmd/dzgorge) overlaid.
#
#   bash gorge/build.sh            # -> $GORGE_DIR/bin/{dzgorge,policytrain,botbench}
#
# GORGE_DIR (default ../ext/gorge next to this repo's parent) is a gorge checkout; it is
# cloned if missing. The Forge card scripts gorge compiles are GPL-3.0: they live in
# $GORGE_DIR/.cards and are never committed here.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GORGE_DIR="${GORGE_DIR:-$(cd "$HERE/../.." && pwd)/ext/gorge}"
GORGE_REF="$(cat "$HERE/GORGE_REF")"

if [ ! -d "$GORGE_DIR/.git" ]; then
  git clone https://github.com/adams-shaun/gorge "$GORGE_DIR"
fi
cd "$GORGE_DIR"
if [ "$(git rev-parse HEAD)" != "$GORGE_REF" ]; then
  git fetch origin "$GORGE_REF" 2>/dev/null || git fetch origin
  git checkout --detach "$GORGE_REF"
fi
export GOFLAGS="${GOFLAGS:--p=4 -trimpath}"
[ -d .cards/cardsfolder ] || make fetch-cards
ls .cards/ir-*.gob.gz >/dev/null 2>&1 || make compile-cards
# DraftZero's opt-in patches to gorge (gorge/patches): applied once, kept across rebuilds.
for p in "$HERE"/patches/*.patch; do
  [ -e "$p" ] || continue
  if git apply --reverse --check "$p" 2>/dev/null; then continue; fi
  git apply "$p"
  echo "applied $(basename "$p")"
done
rm -rf cmd/dzgorge && cp -r "$HERE/cmd/dzgorge" cmd/dzgorge
go vet ./cmd/dzgorge
go build -o bin/dzgorge ./cmd/dzgorge
go build -o bin/policytrain ./cmd/policytrain
go build -o bin/botbench ./cmd/botbench
echo "built $GORGE_DIR/bin/{dzgorge,policytrain,botbench} at gorge $GORGE_REF"
