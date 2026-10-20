#!/usr/bin/env bash
# One GPU run: cache the target's hidden taps, then fine-tune the drafter (two learning rates, best kept).
set -euo pipefail
. "$(dirname "$0")/env.sh"
cd $W
mkdir -p logs
nvidia-smi --query-gpu=power.limit --format=csv,noheader | sed 's/^/power limit: /'
( while sleep 10; do nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits; done ) > logs/extract-gpu-samples.log &
VP=$!
trap 'kill $VP' EXIT
t0=$(date +%s)
venv/bin/python $S/extract_hs.py $W 2>&1 | grep -v "Loading weights"
t1=$(date +%s)
echo "extract wall s: $((t1 - t0))"
venv/bin/python $S/train_drafter.py $W --lrs 2e-5,6e-5 --batch 16 --anchors 256 --anchors-public 64 --out ckpt/drafter-tuned 2>&1 | grep -v "Loading weights"
echo "train wall s: $(( $(date +%s) - t1 ))"
du -sh hs
