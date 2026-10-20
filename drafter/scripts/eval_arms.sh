#!/usr/bin/env bash
# Eval on one RTX 4090 (llama.cpp 69f201a): target alone vs target + DFlash drafter arms.
# Same target GGUF, same flags, concurrency 1, greedy, 25 held-out prompts, n_predict 512, arms back to back.
# usage: eval_arms.sh OUTDIR ARM=DRAFTER_GGUF|none ...
set -uo pipefail
. "$(dirname "$0")/env.sh"
OUT=$1; shift
PORT=18432
mkdir -p $W/$OUT
cd $W
{ nvidia-smi --query-gpu=name,power.limit,temperature.gpu --format=csv,noheader; date; } | tee $OUT/gpu.txt
for spec in "$@"; do
  arm=${spec%%=*}; d=${spec#*=}
  if [ "$d" = none ]; then DR=(); else
    DR=(-md $d -ngld all --spec-type draft-dflash --spec-draft-n-max 15 --spec-draft-n-min 0 --spec-draft-p-min 0 --spec-draft-sampling greedy)
  fi
  t0=$(date +%s)
  LD_LIBRARY_PATH=$R $R/llama-server -m $T -ngl 99 -fa on -np 1 -c 8192 --metrics --no-webui \
     --host 127.0.0.1 --port $PORT "${DR[@]}" > $OUT/server-$arm.log 2>&1 &
  SP=$!
  for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 1; done
  ( while sleep 2; do nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits; done ) > $OUT/gpu-$arm.log &
  VP=$!
  python3 $S/eval_client.py $PORT $arm data/prompts_eval.jsonl data/prompts_train.jsonl $OUT/$arm.jsonl 512
  kill $VP; kill $SP; wait $SP 2>/dev/null
  echo "$arm peak_vram_mib $(cut -d, -f1 $OUT/gpu-$arm.log | sort -n | tail -1) wall_s $(( $(date +%s) - t0 ))" | tee -a $OUT/arms.txt
done
{ nvidia-smi --query-gpu=power.limit --format=csv,noheader; date; } | tee -a $OUT/gpu.txt
echo EVAL-DONE
