# Measurements made before the hackathon

Every number here comes from runs on **14 October 2026**, before the Meta AI Developer Hackathon kickoff. Hardware: one RTX 4090 (24 GB) at a 400 W power limit, Linux. None of these is a hackathon result.

## Decode speed: llama.cpp, Glimmer Q4_K_M, with and without DFlash

llama.cpp `69f201a`. Settings `-np 1 -c 8192 -fa on`, concurrency 1, greedy, `n_predict` 512. 25 held-out prompts (`examples/prompts_eval.jsonl`), none seen in drafter training. Acceptance = 1 + accepted ÷ verify steps, from llama-server's `/metrics` counters.

| Arm | Mean acceptance length | Decode tok/s | Speedup | Same tokens as target-only (all 25) | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Glimmer alone | – | 51.3 | 1.00× | 25/25 | 16,100 MiB |
| + official DFlash (Meta's Q4_K_M file) | 2.91 | 83.4 | 1.62× | 14/25 | 18,594 MiB |
| + stock drafter, our conversion | 2.88 | 82.0 | 1.60× | 14/25 | 18,594 MiB |
| + tuned drafter (lr 2e-5, released) | 2.83 | 81.2 | 1.58× | 14/25 | 18,594 MiB |
| + tuned drafter (lr 6e-5, rejected on val) | 2.75 | 79.3 | 1.54× | 14/25 | 18,594 MiB |

By prompt source (acceptance length · speedup):

| Source (prompts · decoded tokens) | Official | Stock, our conversion | Tuned 2e-5 |
|---|---|---|---|
| Life chat: a turn with memory notes (15 · 462) | 2.34 · 1.29× | 2.31 · 1.29× | 2.38 · 1.32× |
| Life recall: a question answered from notes (5 · 166) | 3.29 · 1.81× | 2.88 · 1.60× | 3.10 · 1.71× |
| Public chat: UltraChat first turn (5 · 2,392) | 3.03 · 1.69× | 3.02 · 1.67× | 2.92 · 1.64× |

Limits:
- The DFlash arms decoded 2,880 tokens and target-only 3,020, because 11 outputs differ. The speedups compare rates on slightly different text.
- All four drafters gave identical token streams to each other on 25/25 prompts. Each differed from target-only on 11/25. At 10 of the 11 split points, the DFlash token was Glimmer's own second choice, but 4 of those gaps were 12 to 23 probability points. A verify-path fault in this llama.cpp build is not ruled out.
- On the 20 agent-turn prompts alone, the tuned drafter reached 1.41× against 1.36× for the same-converter stock control, level with Meta's file (1.40×).
- The 1.6× ceiling is set by low acceptance on every source, long replies included.

## Drafter tune cost

| Step | GPU time | Peak VRAM |
|---|---:|---:|
| Generate Glimmer's outputs (1,964 prompts, 237,611 reply tokens, 8 slots) | 24.2 min | 17,208 MiB |
| Extract taps (1.58 M tokens at 1,975 tok/s, bf16, 100 GB on disk) | 13.8 min | 18,412 MiB |
| Fine-tune, 2 learning rates, with checks and val | 32.3 min | 17,936 MiB |
| Convert + 5-arm eval | 5.8 min | 18,594 MiB |
| Smoke tests, failed starts, one stopped run | 16.8 min | – |
| **Total** | **93 GPU-min** | |

One nightly retrain is about 60 minutes: generation 24, taps 13, one learning rate 14, conversion 1, a 3-arm check 3, plus about 4 minutes of checks. This is a sum of measured parts, not one timed run. It needs about 60 GB of host RAM for the CPU-side optimizer state.

## QLoRA on the NF4 target

`unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit@ecedda39` (22.2 GB). torch 2.14.1+cu130, transformers 5.15.0, peft 0.21.2, trl 0.22.2, bitsandbytes 0.50.2, unsloth 2026.10.5. Rank 16, alpha 32, dropout 0.05, q/k/v/o in layers 36 to 51 of 52 (9,043,968 trainable parameters), LR 1e-4, batch 1, response-only loss, gradient checkpointing, adamw_8bit. Unsloth offloaded the frozen embeddings to the CPU.

| Sequence length | Step time (3 measured steps) | Sustained input tok/s | Peak NVML VRAM |
|---:|---|---:|---:|
| 2,048 | 2.92 to 2.93 s | 699.31 | 19.90 GiB |
| 4,096 | 5.65 to 5.69 s | 722.28 | 20.49 GiB |

At these rates, 1 M training tokens take about 23 to 24 minutes and 5 M about 2 hours. That is a training-only projection from three warmed steps. It excludes generation, evaluation, save, merge, export and requantisation.

Day-one smoke (`qlora/run_smoke.sh`, gradient accumulation 4, one epoch over 34 synthetic rows, about 9 optimizer steps):

| Arm | Facts (30 × 1 probe) | Preferences (3) | Procedure (1) | General slice (10, synthetic) |
|---|---:|---:|---:|---:|
| Before (clean base) | 0/30 | 1/3 | 0/1 | 10/10 |
| After (adapter reloaded) | 0/30 | 1/3 | 0/1 | 10/10 |

This proves the scoring runs end to end. It is **not** evidence of learning or of no forgetting. Same-base adapter reload took 0.3 s (not a cold load, merge or export). Adapter size: 117,633,880 bytes.

## Not measured yet

- The current llama.cpp release (v0.7.0) speed matrix and the KV snapshot and prefix-cache timings: the `g1.py` and `g2.py` scripts are written and CPU-validated, and the GPU runs are pending.
- Anything on the Mac: `serve/mac/` has not been run.
- EXL3, vLLM and SGLang.
- A full learning-and-forgetting study (the four-arm design with notes and weights).
