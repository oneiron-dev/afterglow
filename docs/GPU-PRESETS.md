# GPU presets: RTX 3090, 4090 and 5090

The RTX 4090 numbers are measured. The RTX 3090 and RTX 5090 presets are **untested; they need a run on rented hardware** ([../bench/rent-test.md](../bench/rent-test.md)). Their budgets below are arithmetic from the 4090 measurements, not measurements.

| | RTX 3090 | RTX 4090 | RTX 5090 |
|---|---|---|---|
| Architecture | Ampere, sm_86 | Ada, sm_89 | Blackwell, sm_120 |
| VRAM | 24 GB | 24 GB | 32 GB |
| CUDA toolkit | 12.x | 12.x | **12.8 or later** |
| llama.cpp build | `-DCMAKE_CUDA_ARCHITECTURES=86` | `=89` | `=120` |
| Harness config | `serve/cuda/config.rtx3090.json` | `serve/cuda/config.example.json` | `serve/cuda/config.rtx5090.json` |
| Target | Q4_K_M (Meta, 16.76 GB) | Q4_K_M | Dynamic Q4_K_XL (Meta, 19.65 GB); Q6_K if you make it |
| Drafter | official DFlash Q4_K_M | same | same (a Q8_0 drafter also fits) |
| Serving context | 32,768 (65,536 to try) | 8,192 measured; 32,768 | 131,072 with Q4_K_XL; 65,536 with Q6_K |
| Target-only decode, published | 45.3 tok/s at 4K | 51.2 tok/s at 4K (we measured 51.3) | 83.2 tok/s at 4K |
| DFlash speedup | not measured | 1.62× measured | 3.1× on Meta's card (its prompt set) |
| QLoRA night step (NF4) | fits, 2.9 GiB headroom | measured | fits, 10.8 GiB headroom |
| Status | **untested** | measured | **untested** |

Published decode figures: Hardware Corner, target-only Q4_K_M in llama.cpp (a different build and file from ours); the 5090 DFlash figure is Meta's model card (llama.cpp, 74.9 → 233.4 tok/s, batch 1, greedy). Our 4090 DFlash gain on our own prompts was 1.62×, so do not expect 3.1× on agent turns.

## The arithmetic

**Measured anchors** (RTX 4090, llama.cpp `69f201a`, `-np 1 -c 8192 -fa on`, peak `nvidia-smi` memory.used):

| What | MiB |
|---|---:|
| Q4_K_M target only | 16,100 |
| Q4_K_M target + official DFlash drafter | **18,594** |
| QLoRA on the NF4 target, sequence 2,048 / 4,096 (NVML peak) | 20,378 / 20,982 (19.90 / 20.49 GiB) |
| Drafter fine-tune / tap extraction / 8-slot generation | 17,936 / 18,412 / 17,208 |

**File sizes:** Q4_K_M target 16,756,683,904 B (15,980 MiB). Dynamic Q4_K_XL target 19,653,960,832 B (18,743 MiB), so +2,763 MiB. DFlash drafter 1,631,208,128 B (1,556 MiB).

Glimmer has 29.78 B parameters (BF16 index: 59,553,253,376 bytes). No official Q6_K or Q8_0 file exists. At the nominal bits per weight:
- Q6_K: 29.78 B × 6.5625 / 8 = 24.43 GB, so +7,314 MiB over Q4_K_M;
- Q8_0: 29.78 B × 8.5 / 8 = 31.64 GB, so +14,192 MiB.

Real files keep a few tensors at higher precision, so allow ±5%.

**KV growth with context.** Target KV in bytes is `1024 × [13N + 39 × min(N, 2048)]`: 39 sliding-window layers stop growing at 2,048 tokens, and 13 global layers grow with every token. Above 8,192 tokens, each extra token adds 13 × 1,024 = 13,312 B:

| Context N | Extra KV over 8,192 |
|---:|---:|
| 32,768 | 13,312 × 24,576 = 312 MiB |
| 65,536 | 13,312 × 57,344 = 728 MiB |
| 131,072 | 13,312 × 122,880 = 1,560 MiB |

**Estimate = 18,594 MiB + (target file − Q4_K_M file) + extra KV.** We plan against 24,000 MiB usable on a 24 GB card and 32,000 MiB on a 32 GB card, below the physical size, and keep **at least 2 GiB (2,048 MiB) of headroom**. That margin covers compute buffers and DFlash state at long context, which were not measured.

| Card | Target + DFlash | 8,192 | 32,768 | 65,536 | 131,072 |
|---|---|---:|---:|---:|---:|
| 3090 | Q4_K_M | 18,594 (5,406 free) | 18,906 (5,094) | 19,322 (4,678) | 20,154 (3,846) |
| 5090 | Dynamic Q4_K_XL | 21,357 (10,643) | 21,669 (10,331) | 22,085 (9,915) | 22,917 (9,083) |
| 5090 | Q6_K (self-made) | 25,908 (6,092) | 26,220 (5,780) | 26,636 (5,364) | 27,468 (4,532) |
| 5090 | Q8_0 (self-made) | 32,786 (−786) | does not fit | does not fit | does not fit |

All values are MiB; free = usable − estimate. Q8_0 does not fit with the drafter. Without the drafter it would need 16,100 + 14,192 = 30,292 MiB at 8K, which leaves 1,708 MiB, under the 2 GiB rule. So **Q6_K is the highest quant that fits a 5090 with DFlash.** There is no official Q6_K. Make one from Meta's BF16 release (about 60 GB to download, about 120 GB of scratch):

```bash
python3 llama.cpp/convert_hf_to_gguf.py Muse-Glimmer-30B --outtype bf16 --outfile glimmer-bf16.gguf
llama.cpp/build/bin/llama-quantize glimmer-bf16.gguf Muse-Glimmer-30B-Q6_K.gguf Q6_K
sha256sum Muse-Glimmer-30B-Q6_K.gguf     # put the hash and byte count into config.rtx5090.json
```

Until then, the 5090 preset uses Meta's own Dynamic Q4_K_XL file. Meta's card puts its quality loss at 0.2% against 1.0% for the 17 GB Q4_K_M.

**QLoRA night step (NF4, sequence 4,096):** 20,982 MiB peak. On a 3090: 24,000 − 20,982 = 3,018 MiB free (2.9 GiB). On a 5090: 32,000 − 20,982 = 11,018 MiB (10.8 GiB). Stop the llama.cpp server first: serving (18.6 GiB) and training (20.5 GiB) do not share one card.

## Card notes

**RTX 3090.**
- There is no FP8 or FP4 hardware, so use K-quants only. bf16 and NF4 work on Ampere.
- Decode is memory-bound: 936 GB/s against the 4090's 1,008 GB/s. The published 45.3 tok/s at 4K is close to that ratio.
- The DFlash verify step is compute-heavy, and the 3090 has less than half the 4090's tensor throughput, so expect less than the 4090's 1.62×. Measure it.
- QLoRA and the drafter retrain fit, but will run slower than the measured 4090 rates.

**RTX 5090.**
- Use CUDA 12.8 or later. Check that your PyTorch build lists `sm_120` (`python -c "import torch; print(torch.cuda.get_arch_list())"`), and that bitsandbytes and Unsloth load on it, before the night step.
- **NVFP4** is a real Blackwell path, but not in this repo. LMSYS published SGLang on its SM120 backend: an 18 GB NVFP4 checkpoint plus the 5 GB BF16 DFlash drafter fits one 5090, at 63.9 → 236.4 tok/s with DFlash. That needs SGLang, not llama.cpp.
- **FP8** does not fit: the FP8-block checkpoint (`RedHatAI/Muse-Glimmer-30B-FP8-block`) is about 34 GB.
- llama.cpp's GGUF K-quants are the path these presets cover.

## Serve commands

The flags are the same as on the 4090 (README step 3). Change the target file and `-c`:

```bash
# RTX 3090
llama-server -m $M/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf -ngl 99 -fa on -np 1 -c 32768 --metrics \
  -md $M/dflash-Muse-Glimmer-30B-Q4_K_M.gguf -ngld all --spec-type draft-dflash \
  --spec-draft-n-max 15 --spec-draft-n-min 0 --spec-draft-p-min 0 --spec-draft-sampling greedy \
  --slot-save-path $GLIMMER_HOME/slots --host 127.0.0.1 --port 8080

# RTX 5090
llama-server -m $M/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf -ngl 99 -fa on -np 1 -c 131072 --metrics \
  -md $M/dflash-Muse-Glimmer-30B-Q4_K_M.gguf -ngld all --spec-type draft-dflash \
  --spec-draft-n-max 15 --spec-draft-n-min 0 --spec-draft-p-min 0 --spec-draft-sampling greedy \
  --slot-save-path $GLIMMER_HOME/slots --host 127.0.0.1 --port 8080
```
