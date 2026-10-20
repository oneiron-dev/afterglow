# Provenance: what existed before the hackathon

Afterglow (Muse Glimmer Afterglow) is built from the files below.

The Meta AI Developer Hackathon starts on **26 October 2026, 17:00 UTC** (27 October, 02:00 JST). Everything in this repository was written before that time. The tag `pre-event-2026-10-21` marks the first published commit; later commits before kickoff are listed in git.

This file lists each pre-existing file with its original date and the sha256 of the original. "Unchanged" means the file is byte-identical to the original. The other files were edited only to remove private paths and host names, to read paths from environment variables or a config file, and to reword comments and messages. Three harness files also gained small config knobs so the same gate runs on other GPUs (noted per file). No algorithm, constant or measurement setting changed.

Dates are file modification times in Japan Standard Time (UTC+9).

## Pre-existing files

| Repo file | Origin | Original date (JST) | Original sha256 | Change for publication |
|---|---|---|---|---|
| `serve/cuda/common.py` | llama.cpp harness | 2026-10-21 03:25 | `e69432566a9abad7` | paths parameterised, host names and internal notes removed; adds the `server.gpu_match` default |
| `serve/cuda/g0.py` | llama.cpp harness | 2026-10-21 03:25 | `e53fe5415896e720` | paths parameterised, host names and internal notes removed; GPU-name gate reads `server.gpu_match`, drafter gate follows the configured arms |
| `serve/cuda/g1.py` | llama.cpp harness | 2026-10-21 03:26 | `25b0626ce746df7a` | paths parameterised, host names and internal notes removed |
| `serve/cuda/g2.py` | llama.cpp harness, KV snapshot | 2026-10-21 04:31 | `87d93a44e651e8ac` | paths parameterised, host names and internal notes removed |
| `serve/cuda/make_prompts.py` | llama.cpp harness | 2026-10-21 03:27 | `22ab640712620d6c` | renamed from `tokenize.py` (it shadowed the stdlib module); paths parameterised, host names and internal notes removed |
| `serve/cuda/selfcheck.py` | llama.cpp harness | 2026-10-21 04:31 | `a4fb43b196febab8` | renamed from `validate_root.py`; paths parameterised, host names and internal notes removed; cell counts follow the configured arms |
| `serve/cuda/config.example.json` | llama.cpp harness | 2026-10-21 04:27 | `cf043c7d74eeb7b3` | rewritten as an example: placeholder paths, build-host fields removed, preset header and `gpu_match` added |
| `serve/mac/run_mac_glimmer.sh` | MLX harness | 2026-10-21 03:03 | `9a69f77c8d346878` | paths parameterised, host names and internal notes removed |
| `serve/mac/infer_mac_glimmer.py` | MLX harness, KV snapshot | 2026-10-21 03:05 | `dd19d2d9f4a71637` | docstring and one message reworded |
| `drafter/scripts/analyze_eval.py` | drafter tune | 2026-10-14 19:39 | `e4c16d79ae5e2f78` | unchanged |
| `drafter/scripts/build_prompts.py` | drafter tune | 2026-10-14 19:08 | `806fec387716424c` | unchanged |
| `drafter/scripts/check_leak.py` | drafter tune | 2026-10-14 20:54 | `2ccb238b104ef23c` | unchanged |
| `drafter/scripts/check_recompute.py` | drafter tune | 2026-10-14 20:54 | `a169f20d1793fa1a` | unchanged |
| `drafter/scripts/eval_client.py` | drafter tune | 2026-10-14 19:17 | `ea3d30bf1486e245` | unchanged |
| `drafter/scripts/extract_hs.py` | drafter tune | 2026-10-14 19:16 | `cee44a1d1a7f5efc` | unchanged |
| `drafter/scripts/fetch_public.py` | drafter tune | 2026-10-14 19:07 | `4604bd854f98ad79` | unchanged |
| `drafter/scripts/gen_client.py` | drafter tune | 2026-10-14 19:09 | `79ec75b87e638d43` | unchanged |
| `drafter/scripts/hf_fetch.py` | drafter tune | 2026-10-14 19:05 | `b8ac153d54169eda` | unchanged |
| `drafter/scripts/tie_check.py` | drafter tune | 2026-10-14 20:46 | `797b8b337182d7fc` | unchanged |
| `drafter/scripts/train_drafter.py` | drafter tune | 2026-10-14 19:55 | `ebf438b558e5e8c6` | unchanged |
| `drafter/scripts/convert_gguf.sh` | drafter tune | 2026-10-14 19:26 | `4af89ae9bc8894b4` | paths come from `env.sh`; host names removed |
| `drafter/scripts/eval_arms.sh` | drafter tune | 2026-10-14 19:37 | `d190a58afd2bb640` | renamed (host-specific name dropped); paths come from `env.sh`; host names removed |
| `drafter/scripts/extract_and_train.sh` | drafter tune | 2026-10-14 19:35 | `83981940c8a7e2b5` | paths come from `env.sh`; host names removed |
| `drafter/scripts/gen_outputs.sh` | drafter tune | 2026-10-14 19:09 | `8989de2e5091be2d` | renamed (host-specific name dropped); paths come from `env.sh`; host names removed |
| `drafter/scripts/post_train.sh` | drafter tune | 2026-10-14 20:00 | `7887a3b4e4df5f3b` | paths come from `env.sh`; host names removed |
| `drafter/scripts/setup.sh` | drafter tune | 2026-10-14 19:05 | `72d11bd5fa009bf4` | renamed (host-specific name dropped); paths come from `env.sh`; host names removed |
| `drafter/scripts/tie_check.sh` | drafter tune | 2026-10-14 20:35 | `cbbf6695bbc24cd9` | paths come from `env.sh`; host names removed |
| `drafter/scripts/train_only.sh` | drafter tune | 2026-10-14 19:55 | `04249fa19644e9ae` | paths come from `env.sh`; host names removed |
| `qlora/smoke_eval_gpu.py` | QLoRA smoke | 2026-10-14 04:06 | `07a709d172548f33` | docstring and messages reworded |
| `qlora/rate_both.py` | QLoRA rate | 2026-10-14 03:58 | `1e30f12b8a0a44df` | model dir from `GLIMMER_NF4`; docstring reworded |
| `qlora/run_rate.sh` | QLoRA rate | 2026-10-14 03:52 | `dc92b98cf408955b` | renamed; paths parameterised, host names and internal notes removed |
| `qlora/run_smoke.sh` | QLoRA smoke | 2026-10-14 04:04 | `59ba0c4ebd883370` | renamed; paths parameterised, host names and internal notes removed |
| `qlora/fixtures/train_rows.json` | QLoRA smoke data (synthetic) | 2026-10-14 04:04 | `59bd6284af7d3dc2` | unchanged |
| `qlora/fixtures/recall_probes.json` | QLoRA smoke data (synthetic) | 2026-10-14 04:04 | `ea746b4dea56b21d` | unchanged |
| `qlora/fixtures/general_slice.json` | QLoRA smoke data (synthetic) | 2026-10-14 04:04 | `ae059616a3aeff99` | unchanged |
| `qlora/fixtures/synthetic-day-fixture.json` | QLoRA pilot data (synthetic) | 2026-10-14 02:33 | `94bd4ac48a5b5fa0` | unchanged |
| `examples/prompts_eval.jsonl` | drafter tune eval prompts (synthetic + UltraChat) | 2026-10-14 19:08 | `95e96226d83de3ca` | unchanged |

## Files written for publication (21 October 2026)

`README.md`, `NOTICE`, `PROVENANCE.md`, `docs/MEASUREMENTS.md`, `docs/GPU-PRESETS.md`, `bench/rent-test.md`, `serve/cuda/config.rtx3090.json`, `serve/cuda/config.rtx5090.json`, `drafter/README.md`, `drafter/RELEASE.md`, `drafter/scripts/env.sh`, `drafter/requirements.txt`, `qlora/requirements.txt`, `.gitignore`. `LICENSE` is the Apache-2.0 text. `licenses/GLIMMER-LICENSE` and `licenses/GLIMMER-USAGE-POLICY.md` are verbatim copies from `meta-models/Muse-Glimmer-30B-assistant` at `e8192f3a`.

## Pre-existing artifacts outside git

| Artifact | Made | Identity |
|---|---|---|
| Tuned DFlash drafter, Q4_K_M GGUF | 14 Oct 2026 | sha256 `d78021f9d342266f720377b9f3633152ad4ccad51aa90761e033671f40d8a717` |
| Tuned DFlash drafter, bf16 safetensors | 14 Oct 2026 | sha256 `39db46a1cb627b8a7898fe49949be9ea4ecae1ec7f36e0f90e58c58270c95d78` |
| Stock drafter through our converter (control) | 14 Oct 2026 | sha256 `8dbc4f0313a6a1afca38fb69f81ff2ad06845c641a5edd2176236d1bdd1a19d4` |
| Measurements in `docs/MEASUREMENTS.md` | 14 Oct 2026 | — |

## Not included

- The 528 fictional "lives" used to build the drafter training prompts, and the generated training set.
- Run logs, raw result files and server logs.
- Model weights (download them from Hugging Face at the pinned revisions).
