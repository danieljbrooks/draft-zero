#!/usr/bin/env bash
# Build gorge's DraftZero driver (cmd/dzgorge) at the pinned fork commit.
#
#   bash engine/gorge/build.sh                               # GORGE_DIR defaults to ext/gorge-dz next to this repo
#   GORGE_DIR=/work/gorge-fork bash engine/gorge/build.sh    # an existing checkout
#
# -> $GORGE_DIR/bin/dzgorge and $GORGE_DIR/bin/dzgorge.build.json
#    {repo, ref, dirty, go_version, built_utc, binary_sha256, cards_lock}
#
# GORGE_REPO and GORGE_REF (next to this script) pin the engine: the DraftZero fork of gorge, whose main owns
# cmd/dzgorge and whose fork commits carry DraftZero's deltas, so nothing is patched or overlaid here. GORGE_DIR is
# cloned from GORGE_REPO if missing, then checked out (detached) at GORGE_REF; a commit that is already in the
# checkout is used even when the remote does not have it (yet). The Forge card scripts gorge compiles are GPL-3.0:
# they live in $GORGE_DIR/.cards, are fetched and compiled only when missing, and are never committed here.
# Needs git, make and Go (1.25+) on PATH.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
GORGE_REPO="$(tr -d '[:space:]' < "$HERE/GORGE_REPO")"
GORGE_REF="$(tr -d '[:space:]' < "$HERE/GORGE_REF")"
GORGE_DIR="${GORGE_DIR:-$(dirname "$ROOT")/ext/gorge-dz}"

case "$GORGE_REF" in
  *[!0-9a-f]* | "") echo "build.sh: GORGE_REF must be a full commit hash, not '$GORGE_REF'" >&2; exit 1 ;;
esac
[ "${#GORGE_REF}" -eq 40 ] || { echo "build.sh: GORGE_REF must be 40 hex digits" >&2; exit 1; }

if [ ! -e "$GORGE_DIR/.git" ]; then
  mkdir -p "$(dirname "$GORGE_DIR")"
  git clone "$GORGE_REPO" "$GORGE_DIR"
fi
cd "$GORGE_DIR"
GORGE_DIR="$(pwd)"

have_ref() { git cat-file -e "$GORGE_REF^{commit}" 2>/dev/null; }
if ! have_ref; then
  git fetch "$GORGE_REPO" "$GORGE_REF" 2>/dev/null || git fetch "$GORGE_REPO" || true
fi
if ! have_ref; then
  echo "build.sh: commit $GORGE_REF is neither in $GORGE_DIR nor fetchable from $GORGE_REPO" >&2
  exit 1
fi
if [ "$(git rev-parse HEAD)" != "$GORGE_REF" ]; then
  git checkout --detach "$GORGE_REF"
fi
dirty=false
[ -z "$(git status --porcelain --untracked-files=no)" ] || dirty=true
if [ "$dirty" = true ]; then
  echo "build.sh: warning: $GORGE_DIR has uncommitted changes; the build record says dirty" >&2
fi

[ -d .cards/cardsfolder ] || make fetch-cards
ls .cards/ir-*.gob.gz > /dev/null 2>&1 || make compile-cards

mkdir -p bin
go build -trimpath -o bin/dzgorge ./cmd/dzgorge

if command -v sha256sum > /dev/null; then
  sha="$(sha256sum bin/dzgorge | cut -d' ' -f1)"
else
  sha="$(shasum -a 256 bin/dzgorge | cut -d' ' -f1)"
fi
lock=null
[ -s .cards/cards.lock ] && lock="$(cat .cards/cards.lock)"
tmp="$(mktemp bin/.dzgorge.build.json.XXXXXX)"
cat > "$tmp" <<EOF
{
  "repo": "$GORGE_REPO",
  "ref": "$GORGE_REF",
  "dirty": $dirty,
  "go_version": "$(go env GOVERSION)",
  "built_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "binary_sha256": "$sha",
  "cards_lock": $lock
}
EOF
mv "$tmp" bin/dzgorge.build.json
echo "built $GORGE_DIR/bin/dzgorge (sha256 $sha) at gorge $GORGE_REF"
