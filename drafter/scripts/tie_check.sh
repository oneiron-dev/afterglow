#!/usr/bin/env bash
# Near-tie readout: Glimmer's own top-2 at each point where a DFlash arm left target-only decoding.
set -uo pipefail
. "$(dirname "$0")/env.sh"
cd $W
LD_LIBRARY_PATH=$R $R/llama-server -m $T -ngl 99 -fa on -np 1 -c 8192 --no-webui --host 127.0.0.1 --port 18433 > eval-final/server-tiecheck.log 2>&1 &
SP=$!
for i in $(seq 1 240); do curl -sf http://127.0.0.1:18433/health >/dev/null && break; sleep 1; done
python3 $S/tie_check.py 18433 eval-final eval-final/tie_check.json
kill $SP; wait $SP 2>/dev/null
