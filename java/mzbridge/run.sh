#!/bin/sh
# Run one long-lived bridge worker: JSON lines on stdin -> JSON lines on stdout, logs on stderr.
#   java/mzbridge/run.sh [name]          (name defaults to w0)
#   echo '{"id":1,"op":"ping"}' | java/mzbridge/run.sh w0
# Each worker runs in data/mzbridge/runtime/<name>/ with its own copy of xmage/db: H2 is opened by
# a relative path and two JVMs on one db/ crash (critique C3). The copy is an APFS clone / reflink
# when the filesystem supports it, so it costs no disk until H2 writes.
# Env: MZB_HEAP (default 3g), MZB_LOG_LEVEL (default warn), MZB_JAVA_OPTS (extra JVM flags),
#      MZB_RUNTIME_ROOT, MZ_XMAGE_DIR, MZ_ACTION_VOCAB (default <repo>/assets/vocab/FDN_SPG.tsv),
#      JAVA (default java).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
NAME=${1:-w0}
case "$NAME" in
  */*|.*|"") echo "mzbridge run: bad worker name '$NAME'" >&2; exit 2 ;;
esac
XMAGE=${MZ_XMAGE_DIR:-$REPO/xmage}
JAVA=${JAVA:-java}
RT=${MZB_RUNTIME_ROOT:-$REPO/data/mzbridge/runtime}/$NAME
JARFILE=$HERE/build/mzbridge.jar
VOCAB=${MZ_ACTION_VOCAB:-$REPO/assets/vocab/FDN_SPG.tsv}

[ -f "$XMAGE/db/cards.h2.mv.db" ] || { echo "mzbridge run: no XMage card database at $XMAGE/db (set MZ_XMAGE_DIR)" >&2; exit 2; }
command -v "$JAVA" >/dev/null 2>&1 || { echo "mzbridge run: $JAVA not on PATH (need Java 17+)" >&2; exit 2; }
[ -f "$VOCAB" ] || { echo "mzbridge run: no action vocabulary at $VOCAB (set MZ_ACTION_VOCAB)" >&2; exit 2; }
[ -f "$JARFILE" ] || "$HERE/build.sh"

mkdir -p "$RT/db"
if [ -f "$RT/worker.pid" ] && kill -0 "$(cat "$RT/worker.pid")" 2>/dev/null; then
  echo "mzbridge run: worker '$NAME' is already running (pid $(cat "$RT/worker.pid")); use another name" >&2
  exit 3
fi
# A copy smaller than the source is damaged (e.g. a worker killed during an H2 compaction), and a
# source newer than the copy means xmage/ was rebuilt (new cards): replace it either way
SRC_SIZE=$(wc -c < "$XMAGE/db/cards.h2.mv.db")
if [ -f "$RT/db/cards.h2.mv.db" ] && { [ "$(wc -c < "$RT/db/cards.h2.mv.db")" -lt "$SRC_SIZE" ] \
     || [ "$XMAGE/db/cards.h2.mv.db" -nt "$RT/db/cards.h2.mv.db" ]; }; then
  echo "mzbridge run: replacing the damaged or outdated database copy in $RT/db" >&2
  rm -f "$RT/db/cards.h2.mv.db" "$RT/db/cards.h2.lock.db"
fi
if [ ! -f "$RT/db/cards.h2.mv.db" ]; then
  cp -c "$XMAGE/db/cards.h2.mv.db" "$RT/db/" 2>/dev/null \
    || cp --reflink=auto "$XMAGE/db/cards.h2.mv.db" "$RT/db/" 2>/dev/null \
    || cp "$XMAGE/db/cards.h2.mv.db" "$RT/db/"
fi
echo $$ > "$RT/worker.pid"   # exec keeps this pid

# JDK 26+ warns when XMage's watchers mutate final fields reflectively (xmage_state.md G12);
# older JDKs do not know the flag
FINAL_FLAG=""
if "$JAVA" --enable-final-field-mutation=ALL-UNNAMED -version >/dev/null 2>&1; then
  FINAL_FLAG="--enable-final-field-mutation=ALL-UNNAMED"
fi

# G1 with a periodic collection: a worker idle for 10 s hands its search memory back to the OS
# (measured: ~1.9 GB after coach requests -> ~0.8 GB). SerialGC was slower and kept 3.3 GB.
cd "$RT"
exec "$JAVA" -Xms256m -Xmx"${MZB_HEAP:-3g}" -XX:+UseG1GC -XX:G1PeriodicGCInterval=10000 $FINAL_FLAG ${MZB_JAVA_OPTS:-} \
  -Dlog4j.configuration="file:$HERE/log4j.properties" -Dmzb.logLevel="${MZB_LOG_LEVEL:-warn}" \
  -Dmz.actionVocab="$VOCAB" \
  -cp "$JARFILE:$XMAGE/lib/*" org.draftzero.mzbridge.Worker
