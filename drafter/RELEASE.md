# Release artifact: agent-tuned DFlash drafter for Muse Glimmer 30B

Weights are not stored in git. They ship as release files.

| File | Bytes | sha256 | Use |
|---|---:|---|---|
| `dflash-tuned-lr2e-05-Q4_K_M.gguf` | 1,631,205,408 | `d78021f9d342266f720377b9f3633152ad4ccad51aa90761e033671f40d8a717` | llama.cpp `-md` drafter for the official Glimmer Q4_K_M target |
| `model.safetensors` (bf16, HF format) | 5,111,976,608 | `39db46a1cb627b8a7898fe49949be9ea4ecae1ec7f36e0f90e58c58270c95d78` | further training; load with `MuseGlimmerAssistantModel` |
| `config.json` | 878 | `3aaa82631c4410c581babf3b709724bdf33862e5feb9166c8f9a6a40714dc69b` | goes with `model.safetensors` |
| `dflash-stockmine-Q4_K_M.gguf` (control) | 1,631,205,408 | `8dbc4f0313a6a1afca38fb69f81ff2ad06845c641a5edd2176236d1bdd1a19d4` | Meta's drafter through our converter and quant: the fair baseline for the tune |

- **Base:** `meta-models/Muse-Glimmer-30B-assistant` at `e8192f3a8f617f74be2ce220360c89ef4789f39f` (Apache-2.0).
- **Target it was tuned for:** `meta-models/Muse-Glimmer-30B-GGUF` at `70bf1b61ac09f91b24d39038091b41c582bc5d7a`, file `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`.
- **Training data:** Glimmer's own greedy replies to 1,964 prompts (976 life-chat turns and 488 recall questions from fictional personas, 500 UltraChat-200k `train_sft` first turns), and hidden taps from the NF4 target `unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit@ecedda39`. One epoch, 113 steps, lr 2e-5. Trained 14 Oct 2026.
- **Converter:** llama.cpp `69f201a` `convert_hf_to_gguf.py --outtype bf16`, then `llama-quantize Q4_K_M --tensor-type attn_v=q6_k --tensor-type ffn_down=q6_k`.
- **Licence:** Apache-2.0, as a modification of Meta's Apache-2.0 release, plus the [Muse Glimmer Usage Policy](../licenses/GLIMMER-USAGE-POLICY.md). Ship [NOTICE](../NOTICE), [the licence](../licenses/GLIMMER-LICENSE) and the policy with the files.

## Where it is published

**Not uploaded yet.** The plan: the primary home is a Hugging Face model repo, because the bf16 checkpoint (5.1 GB) is over GitHub's 2 GiB per-file release limit.

- `oneiron-dev/muse-glimmer-30b-dflash-agent-tuned`: both GGUFs, `model.safetensors`, `config.json`, a model card with the table above, `LICENSE`, `USAGE_POLICY.md` and `NOTICE`.
- A GitHub release on this repo, tag `drafter-v0.1.0`, carries the two GGUFs (each under 2 GiB) and a `SHA256SUMS` file, as a mirror.

Check a download with `sha256sum -c SHA256SUMS`.
