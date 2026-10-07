# Shared helpers for the dzg scripts (sourced, not run).
#
#   DZ        this repo (default: the parent of gorge/)
#   GORGE_DIR the gorge checkout with bin/dzgorge (default ../ext/gorge next to the repo's parent)
#   PY        python with torch and numpy (default: python3)
#   DEVICE    the server's torch device (default auto)
#   SERVERS   dzg.serve processes per network, spread over the GPUs (default 1)
#   SERVE_ARGS extra dzg.serve arguments (e.g. --prior-temp 0.5)

DZ="${DZ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GORGE_DIR="${GORGE_DIR:-$(cd "$DZ/.." && pwd)/ext/gorge}"
PY="${PY:-python3}"
DEVICE="${DEVICE:-auto}"
SERVERS="${SERVERS:-1}"
D="$DZ/data/gorge"
SERVER_PIDS=()

# serve CKPT LOGDIR starts SERVERS servers for CKPT (a checkpoint path, or "random:<arch>" for an
# untrained network of that architecture) and sets ADDR to their remote= address list. Call it
# directly, never inside $(...): the servers must belong to this shell so stop_servers can end them.
serve() {
  local ckpt="$1" logdir="$2" addrs="" i src
  mkdir -p "$logdir"
  for ((i = 0; i < SERVERS; i++)); do
    local sock="/tmp/dzg-$$-$RANDOM-$i.sock"
    if [[ "$ckpt" == random:* ]]; then src=(--arch "${ckpt#random:}" --random-init); else src=(--ckpt "$ckpt"); fi
    local dev="$DEVICE"
    if [ "$dev" = "auto" ] && command -v nvidia-smi >/dev/null; then
      local ngpu; ngpu=$(nvidia-smi -L | wc -l)
      dev="cuda:$((i % ngpu))"
    fi
    # shellcheck disable=SC2086
    PYTHONPATH="$DZ/gorge" "$PY" -m dzg.serve "${src[@]}" --socket "$sock" --device "$dev" ${SERVE_ARGS:-} \
      > "$logdir/serve$i.log" 2>&1 &
    SERVER_PIDS+=($!)
    addrs+="${addrs:+,}unix:$sock"
  done
  ADDR="$addrs"
}

stop_servers() {
  local p
  for p in "${SERVER_PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  SERVER_PIDS=()
}
trap stop_servers EXIT

# play MINUTES ARGS... runs dzgorge play on the pool, killed after MINUTES (an external hard kill).
play() {
  local min="$1"; shift
  timeout "${min}m" "$GORGE_DIR/bin/dzgorge" play -cards "$GORGE_DIR/.cards" -decks "$D/decks" -pool "$D/pool.tsv" "$@"
}
