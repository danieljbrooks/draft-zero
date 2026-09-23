#!/bin/sh
# Final eval, then tear the worker down.
#
# Used as the watchdog's --on-complete, launched detached, because the watchdog gives that
# command only 120s and the eval takes hours:
#
#   --on-complete "setsid nohup sh deploy/final_eval.sh runs/<id> --config configs/x.yml \
#                  --vs offline gen10 --pairs 100 >> logs/final_eval.log 2>&1 < /dev/null &"
#
# The watchdog has already stopped the loop and verified the weights off-pod before calling
# this. The teardown runs however the eval ends -- success, crash, or the timeout -- so a broken
# eval can never leave the worker billing.
cd "$(dirname "$0")/.." || exit 1
RUN_DIR=$1; shift
TEARDOWN=${DZ_TEARDOWN:-/root/terminate.sh}
HOURS=${DZ_FINAL_EVAL_HOURS:-6}

echo "[final_eval.sh] $(date -u +%FT%TZ) starting (timeout ${HOURS}h, then: $TEARDOWN)"
timeout -k 10m "${HOURS}h" python -u -m draftzero.final_eval --run-dir "$RUN_DIR" "$@"
echo "[final_eval.sh] $(date -u +%FT%TZ) eval exited $?; tearing down"
sh -c "$TEARDOWN"
echo "[final_eval.sh] teardown exited $?"
