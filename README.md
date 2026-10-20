# glimmer-nightly

Tools to run Meta's **Muse Glimmer 30B** on your own hardware and retrain parts of it at night:

- **Serve** Glimmer on one RTX 4090 (llama.cpp + the official DFlash drafter) or on an Apple-silicon Mac (MLX-LM).
- **Snapshot** the KV cache to disk and restore it, so a restarted server does not prefill the whole history again.
- **Retrain at night:** a QLoRA adapter for the target, and a DFlash drafter retrain on the target's own outputs.
- **Measure** all of it with resumable benchmarks that keep exact token IDs and the server's own counters.

Everything here was written and measured **before** the Meta AI Developer Hackathon kickoff (26 Oct 2026, 17:00 UTC).
We publish it as open source so that our hackathon entry can build on it, with that dependency disclosed.
[PROVENANCE.md](PROVENANCE.md) lists every file, its original date and what changed for publication.

> Status: research tooling, not a product. [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) says what was measured, on what, and what was not.

## What is in the repo

| Path | What it does | Written |
|---|---|---|
| `serve/cuda/` | llama.cpp harness: readiness gate (`g0.py`), speed matrix with and without DFlash drafters (`g1.py`), KV snapshot save/restore and prefix-cache reuse (`g2.py`), prompt builder (`make_prompts.py`), CPU self-check (`selfcheck.py`). Python stdlib only. | 21 Oct 2026 |
| `serve/mac/` | MLX-LM benchmark: 4-bit decode speed, plus a prompt-cache save, reload and token-ID replay check. | 21 Oct 2026 |
| `drafter/` | DFlash drafter tune: prompt split, target generation, hidden-tap extraction, fine-tune, GGUF conversion, eval, leak and recompute checks. | 14 Oct 2026 |
| `qlora/` | Nightly QLoRA scaffolding: a training-rate measurement and a day-one before/after smoke on a synthetic day. | 14 Oct 2026 |
| `examples/prompts_eval.jsonl` | 25 held-out eval prompts: 20 turns from fictional personas and 5 UltraChat `test_sft` first turns. | 14 Oct 2026 |
| `qlora/fixtures/` | A synthetic day for one fictional persona (30 facts, preferences, one procedure) and a 10-item synthetic general slice. | 14 Oct 2026 |

All example data is synthetic or public. The repo holds no real user data, logs or traces.

## Run it on an RTX 4090 (Linux, CUDA)

You need a 24 GB card, CUDA 12.x and about 20 GB of free disk for the models.

**1. Build llama.cpp.** The harness targets v0.7.0. The 14 Oct measurements used commit `69f201a`.

```bash
export GLIMMER_HOME=/srv/glimmer            # any directory you own
git clone --depth 1 --branch v0.7.0 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j --target llama-server llama-tokenize llama-quantize
mkdir -p $GLIMMER_HOME/runtime && ln -s "$PWD/llama.cpp/build/bin" $GLIMMER_HOME/runtime/bin
```

**2. Get the weights** at pinned revisions. `hf_fetch.py` checks each file's sha256 against Hugging Face (it needs `aria2c`).

```bash
M=$GLIMMER_HOME/models
python3 drafter/scripts/hf_fetch.py meta-models/Muse-Glimmer-30B-GGUF 70bf1b61ac09f91b24d39038091b41c582bc5d7a $M \
  Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf dflash-Muse-Glimmer-30B-Q4_K_M.gguf
python3 drafter/scripts/hf_fetch.py unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit ecedda395d4d099b477d135a9a573eedaa5a7aad $M tokenizer.json
```

The tuned drafter (`dflash-tuned-lr2e-05-Q4_K_M.gguf`, 1.63 GB) is a release artifact, not a file in git. See [drafter/RELEASE.md](drafter/RELEASE.md).

**3. Serve.** This runs the target and the official DFlash drafter, one slot, with KV snapshots enabled:

```bash
B=$GLIMMER_HOME/runtime/bin
LD_LIBRARY_PATH=$B $B/llama-server -m $M/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf -ngl 99 -fa on -np 1 -c 8192 --metrics \
  -md $M/dflash-Muse-Glimmer-30B-Q4_K_M.gguf -ngld all --spec-type draft-dflash \
  --spec-draft-n-max 15 --spec-draft-n-min 0 --spec-draft-p-min 0 --spec-draft-sampling greedy \
  --slot-save-path $GLIMMER_HOME/slots --host 127.0.0.1 --port 8080
```

For target-only decoding, drop the flags from `-md` to `--spec-draft-sampling greedy`. Use target-only when the output must match plain decoding token for token (see the caveats).

**4. Benchmark.** Copy `serve/cuda/config.example.json` to `$GLIMMER_HOME/config.json` and replace `/srv/glimmer` with your directory. Then:

```bash
mkdir -p $GLIMMER_HOME/prompts $GLIMMER_HOME/slots && cp examples/prompts_eval.jsonl $GLIMMER_HOME/prompts/
python3 serve/cuda/g0.py --config $GLIMMER_HOME/config.json                             # readiness: GPU, flags, model hashes
python3 serve/cuda/make_prompts.py --config $GLIMMER_HOME/config.json --write-prompts   # needs tokenizers==0.23.3
python3 serve/cuda/selfcheck.py                                                          # CPU only: manifests and self-tests
timeout --kill-after=10s 30m python3 serve/cuda/g1.py --config $GLIMMER_HOME/config.json   # 84 timed requests, 4 arms
timeout --kill-after=10s 25m python3 serve/cuda/g2.py --config $GLIMMER_HOME/config.json   # KV snapshot + prefix cache
```

Results go to `$GLIMMER_HOME/results/` as JSONL, one fsynced record per cell. A rerun skips only the cells that succeeded under the same manifest hash.

## Run it on a Mac (Apple silicon, MLX)

Use an existing Python environment with `mlx` and an `mlx-lm` release that has the Glimmer port (0.32.2 was checked). You also need a local copy of
[`mlx-community/Muse-Glimmer-30B-4bit`](https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit/tree/3e7677d7a40d348a3daba263a2b1c0aa41910710), about 19.4 GB. The script never downloads or installs anything.

```bash
GLIMMER_PYTHON=/abs/path/to/venv/bin/python GLIMMER_MODEL=/abs/path/to/Muse-Glimmer-30B-4bit \
  bash serve/mac/run_mac_glimmer.sh
```

The script decodes 128 tokens three times each at 512 and 2,304 prompt tokens. It saves the prompt cache after 2,303 tokens, which is past the 2,048-token sliding window. Then it reloads the cache and checks that the continuation has the same token IDs. It stops itself after 9 minutes and caps MLX memory at 40 GiB.

This script is prepared and syntax-checked. **It has not been run yet.** MLX-LM has no DFlash support, so the Mac path is target-only.

## KV snapshots: what is covered

- **CUDA:** `g2.py` prefills 8,192 or 32,768 tokens and saves the slot (`POST /slots/0?action=save`). It erases the slot, restores it and compares the continuation's token IDs with a fresh prefill. It also measures prefix-cache reuse at about 30K tokens with the server's `cache_n` and `prompt_n` counters.
- **Mac:** MLX-LM's `save_prompt_cache` and `load_prompt_cache`. Glimmer has 39 sliding-window layers (2,048 tokens) and 13 global layers. The rotating caches carry ring offsets, which must survive the round trip.
- Snapshots are **target-only**. The DFlash drafter's state is not saved; it starts cold.
- A snapshot is valid only for the same weights, quant, tokenizer, chat template and prefix. Rebuild it after any nightly weight change.
- Snapshot files do not move between llama.cpp and MLX.

## Nightly retraining

- **QLoRA** (`qlora/`): rank 16, alpha 32, q/k/v/o in the last 16 of 52 layers, response-only loss, on the NF4 target. `run_rate.sh` measures training tok/s. `run_smoke.sh` trains on the synthetic day and scores recall and a general slice before and after. Pinned versions are in `qlora/requirements.txt`.
- **Drafter** (`drafter/`): after the target changes, retrain the DFlash drafter on the target's own greedy outputs. This takes about 60 minutes on one 4090. Recipe and commands: [drafter/README.md](drafter/README.md).

## Known caveats

- With DFlash on, llama.cpp `69f201a` output matched target-only decoding on 14 of 25 prompts. All four drafters gave the same output as each other. A fault in the speculative verify path is not ruled out. Use target-only decoding for exact comparisons.
- The tuned drafter is not faster overall than Meta's. On agent turns it is 3.5% faster than the same-converter stock control and level with Meta's file. On long public chat it is slower.
- The QLoRA smoke proves that the scoring runs. It does not show learning: held-out fact recall was 0/30 before and after one short pass.
- EXL3, vLLM and SGLang paths are not included and were not tested.

## Licence

Our code is licensed under [Apache-2.0](LICENSE). Meta releases the Muse Glimmer weights under Apache-2.0. The weights, and anything derived from them (the tuned drafter included), are also subject to the [Muse Glimmer Usage Policy](licenses/GLIMMER-USAGE-POLICY.md). [NOTICE](NOTICE) lists the third-party notices.
