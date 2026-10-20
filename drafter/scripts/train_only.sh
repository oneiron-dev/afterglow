#!/usr/bin/env bash
# Fine-tune the drafter on the cached taps (two learning rates, best on the held-out val lives kept).
set -euo pipefail
. "$(dirname "$0")/env.sh"
cd $W
mkdir -p logs
( while sleep 10; do nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits; done ) > logs/train-gpu-samples.log &
VP=$!
trap 'kill $VP' EXIT
t1=$(date +%s)
venv/bin/python -u $S/train_drafter.py $W --lrs ${LRS:-2e-5,6e-5} --batch 16 --anchors 256 --anchors-public 64 --out ckpt/drafter-tuned 2>&1 | grep --line-buffered -v "Loading weights"
echo "train wall s: $(( $(date +%s) - t1 ))"
