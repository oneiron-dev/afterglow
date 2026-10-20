#!/usr/bin/env bash
# Training-data generation: Glimmer Q4_K_M target alone, greedy, 8 parallel slots.
set -uo pipefail
. "$(dirname "$0")/env.sh"
PORT=18431
cd $W
mkdir -p logs
DEADLINE=$(( $(date +%s) + ${GEN_MINUTES:-60} * 60 ))
LD_LIBRARY_PATH=$R $R/llama-server -m $T -ngl 99 -fa on -np 8 -c 40960 \
  --host 127.0.0.1 --port $PORT --no-webui > logs/gen-server.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 2; done
nvidia-smi --query-gpu=memory.used,power.draw,power.limit --format=csv,noheader
( while sleep 20; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done ) > logs/gen-vram.log &
VP=$!
python3 $S/gen_client.py $PORT data/prompts_train.jsonl data/gen_train.jsonl 512 8 $DEADLINE
kill $VP
echo "peak VRAM MiB: $(sort -n logs/gen-vram.log | tail -1)"
