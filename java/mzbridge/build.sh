#!/bin/sh
# Compile the bridge worker against the draft-zero XMage build (read-only use of xmage/lib/*.jar;
# the fork itself is not changed). Output: java/mzbridge/build/mzbridge.jar (git-ignored).
#   java/mzbridge/build.sh
# Env: MZ_XMAGE_DIR (default <repo>/xmage), JAVAC (default javac), JAR (default jar).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
XMAGE=${MZ_XMAGE_DIR:-$REPO/xmage}
LIB=$XMAGE/lib
JAVAC=${JAVAC:-javac}
JARTOOL=${JAR:-jar}
if [ ! -f "$LIB/mage-1.4.58.jar" ] && [ -z "$(ls "$LIB"/mage-1*.jar 2>/dev/null)" ]; then
  echo "mzbridge build: no XMage build at $LIB (set MZ_XMAGE_DIR)" >&2
  exit 2
fi
command -v "$JAVAC" >/dev/null 2>&1 || { echo "mzbridge build: $JAVAC not on PATH (need a JDK 17+)" >&2; exit 2; }
OUT=$HERE/build
rm -rf "$OUT/classes"
mkdir -p "$OUT/classes"
# --release 17: the XMage jars are Java 8 bytecode; 17 is the oldest JDK MageZero's launchers use
"$JAVAC" -nowarn -encoding UTF-8 --release 17 -cp "$LIB/*" -d "$OUT/classes" "$HERE"/src/org/draftzero/mzbridge/*.java
"$JARTOOL" --create --file "$OUT/mzbridge.jar.tmp" -C "$OUT/classes" .
mv "$OUT/mzbridge.jar.tmp" "$OUT/mzbridge.jar"
echo "mzbridge build: $OUT/mzbridge.jar" >&2
