#!/bin/sh
# Copy MageZero's graph state encoder (FeatureGraph + StateEncoder, WillWroble/mage branch
# graph-encoder) into src/org/draftzero/mzbridge/graph/, so the bridge can encode decisions as
# graphs while the engine stays the v0.2 bundle (docs/022 §3.1). The copy is the upstream source
# with the edits below and nothing else; rerun this script to follow a newer upstream commit and
# record the commit in docs/022.
#   java/mzbridge/graph_sync.sh [commit]        (default: the pin below)
# Env: MZ_MAGE_REPO (default <repo>/../mage, a clone with WillWroble/mage fetched).
#
# Edits to the upstream files:
#   1. package org.draftzero.mzbridge.graph, importing ActionEncoder and FeatureMap from the bundle;
#   2. players are named by seat relative to the encoder's agent ("PlayerA" = the agent,
#      "PlayerB" = the opponent), as in MageZero's own data where the agent is always PlayerA.
#      Upstream calls game.getEntityName(id) (player name), which v0.2 lacks, and our seats'
#      names depend on which seat the agent sits in;
#   3. addLabeledState (MageZero's training-data path) and the FeatureMap debug hook are dropped:
#      they take v0.2 types the graph encoder no longer produces, and the bridge writes its own rows.
set -e
PIN=e4afc9c77ba7e4a6dbc24966cbdc6dc0819e0bff   # graph-encoder head, 2026-10-04 12:38 PM PT ("encoder change")
COMMIT=${1:-$PIN}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
MAGE=${MZ_MAGE_REPO:-$REPO/../mage}
SRC=Mage.Server.Plugins/Mage.Player.AI/src/main/java/mage/player/ai/encoder
OUT=$HERE/src/org/draftzero/mzbridge/graph
git -C "$MAGE" cat-file -e "$COMMIT^{commit}" 2>/dev/null || {
  echo "graph_sync: $COMMIT not in $MAGE (git -C $MAGE fetch https://github.com/WillWroble/mage graph-encoder)" >&2
  exit 2
}
mkdir -p "$OUT"
HEADER="// Vendored from WillWroble/mage@$COMMIT ($SRC) by java/mzbridge/graph_sync.sh; do not edit by hand."

git -C "$MAGE" show "$COMMIT:$SRC/FeatureGraph.java" | sed \
  -e 's/^package mage\.player\.ai\.encoder;/package org.draftzero.mzbridge.graph;/' \
  -e 's/import static mage\.player\.ai\.encoder\.StateEncoder\.stringToUUID;/import static org.draftzero.mzbridge.graph.StateEncoder.stringToUUID;/' \
  | { echo "$HEADER"; cat; } > "$OUT/FeatureGraph.java"

git -C "$MAGE" show "$COMMIT:$SRC/StateEncoder.java" | python3 -c '
import re, sys
s = sys.stdin.read()
def sub(pattern, repl, count=1, flags=0):
    global s
    s, n = re.subn(pattern, repl, s, count=count, flags=flags)
    if n == 0:
        sys.exit("graph_sync: upstream changed, pattern not found: " + pattern)
sub(r"^package mage\.player\.ai\.encoder;",
    "package org.draftzero.mzbridge.graph;\n\nimport mage.player.ai.encoder.ActionEncoder;\nimport mage.player.ai.encoder.FeatureMap;",
    flags=re.M)
sub(r"import static mage\.player\.ai\.encoder\.FeatureGraph\.\*;", "import static org.draftzero.mzbridge.graph.FeatureGraph.*;")
# 2. seat-relative player names
sub(r"game\.getEntityName\(target\)", "entityName(target, game)", count=0)
sub(r"game\.getPlayer\(myPlayerId\)\.getName\(\)", "\"PlayerA\"")
sub(r"game\.getPlayer\(opponentId\)\.getName\(\)", "\"PlayerB\"")
sub(r"(    private void processStackObject\()",
    "    /** game.getEntityName on the graph-encoder branch, with players named relative to the agent. */\n"
    "    private String entityName(UUID id, Game game) {\n"
    "        if (game.getPlayer(id) != null) return id.equals(myPlayerId) ? \"PlayerA\" : \"PlayerB\";\n"
    "        return game.getEntityName(id, myPlayerId);\n"
    "    }\n\\1")
# 3. MageZero-only hooks
sub(r"    public List<LabeledState>  labeledStates = new ArrayList<>\(\);\n", "")
sub(r"    public void addLabeledState\(.*?\n    }\n", "", flags=re.S)
sub(r"        if\(useFeatureMap\) \{\n            featureMap\.addFeature\(name, hash\);\n        }\n", "")
sys.stdout.write(s)
' | { echo "$HEADER"; cat; } > "$OUT/StateEncoder.java"
echo "graph_sync: $OUT from $COMMIT" >&2
