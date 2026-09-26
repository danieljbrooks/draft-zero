#!/bin/bash
# build_vocab.sh — generate a set-wide action vocabulary from an XMage bundle's own card classes.
#
#   tools/vocab/build_vocab.sh <xmage dir> <out.tsv> [set classes] [dim] [extra cards file]
#
# Defaults reproduce assets/vocab/FDN_SPG.tsv: Foundations in full, plus the Special Guests cards
# listed in assets/vocab/FDN_SPG_extra_cards.txt, at width 1024. The action keys are
# ability.toString() of every activated ability, so regenerate the vocabulary whenever the
# engine changes (e.g. v0.1 -> v0.2): if a key's text changed, its action would silently fall
# into the hashed tail.
set -euo pipefail
XMAGE="$(cd "$1" && pwd)"; OUT="$2"
SETS="${3:-mage.sets.Foundations,mage.sets.SpecialGuests}"
DIM="${4:-1024}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EXTRA="${5:-$HERE/../../assets/vocab/FDN_SPG_extra_cards.txt}"
JAVAC="${JAVA_HOME:+$JAVA_HOME/bin/}javac"; JAVA="${JAVA_HOME:+$JAVA_HOME/bin/}java"
B="$(mktemp -d)"; trap 'rm -rf "$B"' EXIT
"$JAVAC" --release 8 -nowarn -Xlint:-options -encoding UTF-8 -cp "$XMAGE/lib/*" -d "$B" "$HERE/VocabDump.java"
"$JAVA" -cp "$B:$XMAGE/lib/*" -Dlog4j.configuration=file:/dev/null VocabDump "$SETS" "$DIM" "$EXTRA" > "$OUT" 2> "$B/err"
tail -1 "$B/err"
echo "wrote $OUT"
