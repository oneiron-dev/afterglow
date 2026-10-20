#!/bin/bash
# QLoRA rate run: check the NF4 shards, then a timeout-wrapped rate_both.py (2,048 and 4,096 tokens).
#   GLIMMER_NF4  unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit@ecedda395d4d099b477d135a9a573eedaa5a7aad, downloaded
#   PYTHON       a Python with torch, unsloth, trl, peft, transformers, datasets (see requirements.txt)
#   LOGD         output dir (default ./logs)
set -u
cd "$(dirname "$0")"
DST=${GLIMMER_NF4:?set GLIMMER_NF4 to the NF4 model dir}
LOGD=${LOGD:-$PWD/logs}
VENV=${PYTHON:-python3}
echo "===GPURUN-BEGIN=== $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv 2>&1 | head -3
echo "---PRECHECK-SIZES---"
for spec in "model-00001-of-00005.safetensors 2689663104" "model-00002-of-00005.safetensors 4937053799" "model-00003-of-00005.safetensors 4994172445" "model-00004-of-00005.safetensors 4994172485" "model-00005-of-00005.safetensors 4592913091"; do
  set -- $spec; a=$(stat -c%s "$DST/$1" 2>/dev/null || echo MISSING)
  [ "$a" = "$2" ] && echo "OK $1" || { echo "SIZE-FAIL $1 got=$a want=$2"; exit 3; }
done
mkdir -p "$LOGD"
echo "---RUN-RATE(cap 2040s + grace)---"
GLIMMER_NF4="$DST" timeout -s TERM --preserve-status -k 60 2040 "$VENV" rate_both.py "$LOGD"
rc=$?
echo "rate rc=$rc"
echo "---LOGS---"; ls -la "$LOGD"/rate-*.json 2>&1
echo "===GPURUN-END==="
exit $rc
