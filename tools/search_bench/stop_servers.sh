#!/usr/bin/env bash
# Stop the benchmark's inference servers (value_server.py, and serve.py with MageZero's servers).
# Matches python processes only, so it never kills the shell that runs it.
ps -eo pid,args | awk '$2 ~ /python/ && ($0 ~ /value_server\.py/ || $0 ~ /search_bench\/serve\.py/ || $0 ~ /magezero\/server\.py/) {print $1}' | xargs -r kill
sleep 2
echo "left: $(ps -eo args | awk '$1 ~ /python/ && /server/' | wc -l)"
