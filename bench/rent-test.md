# Rent-test: check the RTX 3090 and RTX 5090 presets

The 3090 and 5090 presets ([../docs/GPU-PRESETS.md](../docs/GPU-PRESETS.md)) are untested. This page is one test session on a rented box (Vast.ai or RunPod), per card. It answers three questions: does Afterglow fit, how fast does it decode with and without DFlash, and does the KV snapshot round-trip match.

## The box

- One RTX 3090 (24 GB) or one RTX 5090 (32 GB).
- Image: Ubuntu with the CUDA **devel** toolkit. Use **CUDA 12.8 or later for the 5090**; any 12.x works for the 3090.
- At least 100 GB disk, 32 GB RAM and 8 CPU cores.
- Cost is the box's hourly price × about 1 to 2 hours. Check the live price when you book.

## Commands

```bash
# 0. Pick the card
export CARD=3090 ARCH=86  CFG=serve/cuda/config.rtx3090.json
# export CARD=5090 ARCH=120 CFG=serve/cuda/config.rtx5090.json
export GLIMMER_HOME=/workspace/glimmer M=/workspace/glimmer/models
mkdir -p $GLIMMER_HOME/prompts $GLIMMER_HOME/slots $M
apt-get update && apt-get install -y aria2 cmake git build-essential python3-pip curl
git clone https://github.com/oneiron-dev/afterglow && cd afterglow
{ nvidia-smi --query-gpu=name,memory.total,power.limit,driver_version --format=csv; nvcc --version | tail -2; date -u; } | tee $GLIMMER_HOME/gpu.txt

# 1. llama.cpp for this card only
git clone --depth 1 --branch v0.7.0 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j --target llama-server llama-tokenize llama-quantize
mkdir -p $GLIMMER_HOME/runtime && ln -s "$PWD/llama.cpp/build/bin" $GLIMMER_HOME/runtime/bin

# 2. Weights, sha256-checked by hf_fetch.py
T=Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf; [ $CARD = 5090 ] && T=Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf
python3 drafter/scripts/hf_fetch.py meta-models/Muse-Glimmer-30B-GGUF 70bf1b61ac09f91b24d39038091b41c582bc5d7a $M $T dflash-Muse-Glimmer-30B-Q4_K_M.gguf
python3 drafter/scripts/hf_fetch.py unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit ecedda395d4d099b477d135a9a573eedaa5a7aad $M tokenizer.json
python3 drafter/scripts/hf_fetch.py oneiron-dev/afterglow-dflash-drafter main $M dflash-tuned-lr2e-05-Q4_K_M.gguf dflash-stockmine-Q4_K_M.gguf

# 3. Config and prompts
sed "s#/srv/glimmer#$GLIMMER_HOME#g" $CFG > $GLIMMER_HOME/config.json
cp examples/prompts_eval.jsonl $GLIMMER_HOME/prompts/
pip install tokenizers==0.23.3

# 4. The harness
python3 serve/cuda/g0.py --config $GLIMMER_HOME/config.json          # must print no blocked items
python3 serve/cuda/make_prompts.py --config $GLIMMER_HOME/config.json --write-prompts
python3 serve/cuda/selfcheck.py
timeout --kill-after=10s 40m python3 serve/cuda/g1.py --config $GLIMMER_HOME/config.json
timeout --kill-after=10s 25m python3 serve/cuda/g2.py --config $GLIMMER_HOME/config.json

# 5. Fit check at the preset's serving context (32,768 on a 3090, 131,072 on a 5090)
CTX=32768; [ $CARD = 5090 ] && CTX=131072
B=$GLIMMER_HOME/runtime/bin
LD_LIBRARY_PATH=$B $B/llama-server -m $M/$T -ngl 99 -fa on -np 1 -c $CTX --metrics \
  -md $M/dflash-Muse-Glimmer-30B-Q4_K_M.gguf -ngld all --spec-type draft-dflash \
  --spec-draft-n-max 15 --spec-draft-n-min 0 --spec-draft-p-min 0 --spec-draft-sampling greedy \
  --host 127.0.0.1 --port 8099 > $GLIMMER_HOME/fit-server.log 2>&1 &
until curl -sf http://127.0.0.1:8099/health >/dev/null; do sleep 2; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/after load ctx=$CTX: /" | tee $GLIMMER_HOME/fit.txt
python3 - <<'EOF' | tee -a $GLIMMER_HOME/fit.txt
import json, os, urllib.request
rows = [json.loads(l) for l in open(os.environ["GLIMMER_HOME"] + "/prompts/prompts_g1.jsonl")]
ids = next(r["prompt_ids"] for r in rows if r["id"] == "synth-32768-v1")
body = json.dumps({"prompt": ids, "n_predict": 64, "temperature": 0, "top_k": 1, "cache_prompt": False}).encode()
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8099/completion", body, {"Content-Type": "application/json"}), timeout=900))
print("32K request:", {k: r["timings"][k] for k in ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second")})
EOF
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/after 32K request: /' | tee -a $GLIMMER_HOME/fit.txt
kill %1

# 6. Optional: the QLoRA night step (22.2 GB NF4 download). On the 5090, the torch wheel must be a CUDA 12.8+ build.
python3 drafter/scripts/hf_fetch.py unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit ecedda395d4d099b477d135a9a573eedaa5a7aad $GLIMMER_HOME/nf4
python3 -m venv $GLIMMER_HOME/venv && $GLIMMER_HOME/venv/bin/pip install -r qlora/requirements.txt
$GLIMMER_HOME/venv/bin/python -c "import torch; print(torch.cuda.get_arch_list())"     # must list sm_86 / sm_120
GLIMMER_NF4=$GLIMMER_HOME/nf4 PYTHON=$GLIMMER_HOME/venv/bin/python LOGD=$GLIMMER_HOME/results/qlora bash qlora/run_rate.sh

# 7. Bring the results home (no model files)
tar czf afterglow-rtx$CARD-results.tgz -C $GLIMMER_HOME results gpu.txt fit.txt fit-server.log
```

## What to record

| Item | Where it comes from |
|---|---|
| Card, VRAM, power limit, driver, CUDA version | `gpu.txt` |
| Provider, region, image, hourly price, total cost | by hand |
| Readiness: no blocked items | `results/g0.json` |
| Decode tok/s and DFlash acceptance per arm, at 4K and 32K | `results/g1.jsonl` |
| Peak VRAM per arm | `results/g1.jsonl` |
| KV snapshot save and restore ms and bytes at 8K and 32K; token-ID parity | `results/g2.jsonl` |
| Prefix-cache TTFT: cold, hit and miss, with `cache_n` / `prompt_n` | `results/g2.jsonl` |
| Memory at the serving context, after load and after a 32K request; compare with the estimate in GPU-PRESETS.md | `fit.txt` |
| QLoRA tok/s and peak VRAM at 2,048 and 4,096 (optional) | `results/qlora/rate-*.json` |
| Wall-clock minutes per step | by hand |

The preset passes if G0 is clean, G2 parity is identical, and `fit.txt` leaves at least 2 GiB free. Write the numbers into `docs/GPU-PRESETS.md` and remove "untested" for that card.

## Expected minutes, per card

| Step | Minutes |
|---|---:|
| apt, clone, `nvidia-smi` | 2–4 |
| Build llama.cpp, one architecture | 5–10 |
| Download target (16.8 or 19.7 GB) and three drafters (4.9 GB) | 5–20, depending on the box's bandwidth |
| G0, prompts, self-check | 2–3 |
| G1, 84 timed requests across 4 server loads | 15–30 (the cap is 40 min, for the 3090's slower prefill) |
| G2, KV snapshot and prefix cache | 10–25 |
| Fit check | 3–5 |
| **Core total** | **about 45–95** |
| Optional QLoRA: 22.2 GB download, env, rate run | +15–30 |
