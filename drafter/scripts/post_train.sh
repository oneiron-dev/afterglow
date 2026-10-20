#!/usr/bin/env bash
# After training: convert both tuned drafters (same pipeline as the stock control), then run the final eval arms.
set -uo pipefail
. "$(dirname "$0")/env.sh"
cd $W
S2=ckpt/drafter-tuned-sweep.json
[ -f $S2 ] || { echo "no sweep result"; exit 1; }
BEST=$(python3 -c "import json;d=json.load(open('$S2'));print(d['best_lr'])")
OTHER=$(python3 -c "import json;d=json.load(open('$S2'));print([r['lr'] for r in d['runs'] if r['lr']!=d['best_lr']][0])")
B=$(python3 -c "print(f'{float(\"$BEST\"):g}')"); O=$(python3 -c "print(f'{float(\"$OTHER\"):g}')")
echo "best lr $B, other lr $O"
t0=$(date +%s)
bash $S/convert_gguf.sh $W/ckpt/drafter-tuned-lr$B dflash-tuned-lr$B
bash $S/convert_gguf.sh $W/ckpt/drafter-tuned-lr$O dflash-tuned-lr$O
echo "convert wall s: $(( $(date +%s) - t0 ))"
bash $S/eval_arms.sh eval-final target=none \
  stock_official=$W/models/official-gguf/dflash-Muse-Glimmer-30B-Q4_K_M.gguf \
  tuned=$W/gguf/dflash-tuned-lr$B-Q4_K_M.gguf \
  stock_mine=$W/gguf/dflash-stockmine-Q4_K_M.gguf \
  tuned_lr$O=$W/gguf/dflash-tuned-lr$O-Q4_K_M.gguf
python3 $S/analyze_eval.py eval-final target stock_official stock_mine tuned tuned_lr$O
