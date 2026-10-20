# DFlash drafter tune for Muse Glimmer

The question: does fine-tuning Glimmer's official DFlash drafter on Glimmer's own outputs, for agent-style prompts, raise acceptance length and tokens/s on one RTX 4090?

The answer from the 14 Oct 2026 run: **not overall.** Mean acceptance length was 2.83 tuned, 2.88 stock (same converter) and 2.91 for Meta's file. On the 20 agent-turn prompts the tune was about 3.5% faster than the same-converter stock control and level with Meta's file. Retraining pays only after the target itself changes. Full numbers: [../docs/MEASUREMENTS.md](../docs/MEASUREMENTS.md).

## Setup

All scripts read their paths from `env.sh`. Set these first:

| Variable | Meaning | Default |
|---|---|---|
| `DRAFTER_WORK` | work dir with `models/`, `data/`, `hs/`, `ckpt/`, `gguf/`, `logs/`, `venv/` | current dir |
| `LLAMA_BIN` | dir with `llama-server` and `llama-quantize` | `$DRAFTER_WORK/llama.cpp/build/bin` |
| `LLAMA_SRC` | llama.cpp checkout, for `convert_hf_to_gguf.py` | `$DRAFTER_WORK/llama.cpp` |
| `TARGET_GGUF` | the served Glimmer target | `$DRAFTER_WORK/models/official-gguf/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` |

Put the official GGUFs (target and drafter, `meta-models/Muse-Glimmer-30B-GGUF@70bf1b61`) in `$DRAFTER_WORK/models/official-gguf/`. Copy the Glimmer tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`) into `$DRAFTER_WORK/data/tok/`.

## Recipe (repeat steps 2 to 6 after each nightly target fine-tune)

| Step | Command | Where | Time on one 4090 |
|---|---|---|---|
| 0. Env and pinned weights (drafter bf16 5.1 GB, NF4 target 22.2 GB) | `bash scripts/setup.sh` | CPU | 7.5 min |
| 1. Prompts, split by life **before** generation: 488 train lives, 20 eval, 20 spare, plus 500 UltraChat `train_sft` turns | `python3 scripts/fetch_public.py data/public/ultrachat_slice.json` then `python3 scripts/build_prompts.py $DRAFTER_WORK` | CPU | seconds |
| 2. Glimmer's own greedy outputs (target only, 8 slots, ≤512 tokens) | `bash scripts/gen_outputs.sh` | GPU | 24 min for 1,964 prompts |
| 3+4. Hidden taps from blocks 1, 13, 25, 37, 49 of the NF4 target, then one epoch at two learning rates; best on held-out val lives kept | `bash scripts/extract_and_train.sh` (or `LRS=2e-5 bash scripts/train_only.sh` once taps exist) | GPU + ~60 GB host RAM | 11 min taps, 14 min per LR |
| 5+6. Convert to Q4_K_M (official tensor layout) and run the eval arms | `bash scripts/post_train.sh` | GPU | 6 min |
| Near-tie readout | `bash scripts/tie_check.sh` | GPU | under 1 min |
| Checks | `python3 scripts/check_recompute.py eval-final` and `python3 scripts/check_leak.py $DRAFTER_WORK` | CPU | seconds |

Training details: all 2.556 B drafter parameters train; Glimmer's embedding table and lm_head stay frozen. Each block is an anchor inside Glimmer's reply plus 15 mask tokens. Loss is CE on raw lm_head logits, weighted exp(−(k−1)/7) over slots 1 to 15. AdamW (0.9, 0.95), no weight decay, clip 1.0, cosine schedule with 10% warmup, batch 16. fp32 master weights and AdamW state sit on the CPU, so the GPU peak is 16.5 GiB.

Eval flags (llama.cpp): `-md <drafter> -ngld all --spec-type draft-dflash --spec-draft-n-max 15 --spec-draft-n-min 0 --spec-draft-p-min 0 --spec-draft-sampling greedy --metrics`.

## Training data

`build_prompts.py` reads fictional "lives" (`data/lives/lives-*.jsonl`): one JSON object per persona with `life_id`, `lang`, `persona`, `days[].sessions[].turns[]`, a fact `ledger` with validity days, and recall `questions`. The 528 lives we used are not in this repo. `examples/prompts_eval.jsonl` shows the prompt format that results.

## What should help a retrain

1. More agent turns per life. Life chat is where acceptance is weakest (2.3).
2. Taps from the served GGUF, not from NF4 in transformers, so that training features match serving.
3. A Q8_0 or importance-matrix quant of the drafter, to win back Meta's quantisation edge (2.91 vs 2.88 on the same weights).

The tuned drafter itself is a release artifact: [RELEASE.md](RELEASE.md).
