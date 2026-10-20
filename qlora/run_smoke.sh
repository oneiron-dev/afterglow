#!/bin/bash
# Smoke run: fresh base + GA4 short pass + before/after scoring + timed adapter reload, under a hard timeout.
#   GLIMMER_NF4  unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit@ecedda395d4d099b477d135a9a573eedaa5a7aad, downloaded
#   PYTHON       a Python with torch, unsloth, trl, peft, transformers, datasets (see requirements.txt)
#   SMK          output dir (default ./smoke)
set -u
cd "$(dirname "$0")"
DST=${GLIMMER_NF4:?set GLIMMER_NF4 to the NF4 model dir}
SMK=${SMK:-$PWD/smoke}
VENV=${PYTHON:-python3}
echo "===SMOKERUN-BEGIN=== $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv 2>&1 | head -3
for spec in "model-00001-of-00005.safetensors 2689663104" "model-00002-of-00005.safetensors 4937053799" "model-00003-of-00005.safetensors 4994172445" "model-00004-of-00005.safetensors 4994172485" "model-00005-of-00005.safetensors 4592913091"; do
  set -- $spec; a=$(stat -c%s "$DST/$1" 2>/dev/null || echo MISSING)
  [ "$a" = "$2" ] && echo "OK $1" || { echo "SIZE-FAIL $1 got=$a want=$2"; exit 3; }
done
mkdir -p "$SMK"
F=fixtures
echo "---DATA-HASHES---"
sha256sum $F/train_rows.json $F/recall_probes.json $F/general_slice.json
echo "---RUN-SMOKE(cap 1620s + grace)---"
timeout -s TERM --preserve-status -k 60 1620 "$VENV" smoke_eval_gpu.py --model-dir "$DST" --train-rows $F/train_rows.json --recall-probes $F/recall_probes.json --general-slice $F/general_slice.json --adapter-out "$SMK/smoke-adapter" --results-out "$SMK/smoke-results.json" --max-seq-length 2048
rc=$?
echo "smoke rc=$rc"
ls -la "$SMK"/ 2>&1 | head
echo "===SMOKERUN-END==="
exit $rc
