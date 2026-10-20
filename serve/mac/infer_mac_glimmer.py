#!/usr/bin/env python3
"""Offline MLX-LM benchmark: Glimmer 4-bit decode speed plus a prompt-cache (KV) save, reload and replay check."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
LIMIT_SECONDS = 540


def emit(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()


def preflight(model):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("This benchmark needs an Apple-silicon Mac.")
    if os.getpriority(os.PRIO_PROCESS, 0) != 0:
        raise SystemExit("Start from a normal-priority terminal (nice must be 0).")
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit("GLIMMER_MODEL must be an existing local model directory; no downloads.")
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config", config)
    if "glimmer" not in json.dumps(config.get("architectures", [])).lower() and "glimmer" not in str(config.get("model_type", "")).lower():
        raise SystemExit("Not a Glimmer model config.")
    quant = config.get("quantization", text.get("quantization", config.get("quantization_config", {})))
    if quant.get("bits") != 4:
        raise SystemExit("Use an already-converted MLX 4-bit Glimmer model.")
    index = model / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else [p.name for p in model.glob("*.safetensors")]
    if not files:
        raise SystemExit("No local safetensors weights.")
    total = 0
    for name in files:
        path = (model / name).resolve()
        if not path.is_relative_to(model) or not path.is_file() or path.stat().st_size < 1_000_000:
            raise SystemExit("Missing, incomplete, or nonlocal weight shard: " + name)
        total += path.stat().st_size
    if total > 30 * 2**30:
        raise SystemExit("Weight size exceeds this bounded 4-bit benchmark's safety limit.")
    if not (model / "tokenizer.json").is_file() or not (model / "tokenizer_config.json").is_file():
        raise SystemExit("Tokenizer files must already be on disk.")
    return {"weight_bytes": total, "config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest()}


def worker(model, out, manifest):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache

    def sync_cache(cache):
        arrays = [value for _, value in tree_flatten([c.state for c in cache]) if isinstance(value, mx.array)]
        mx.eval(*arrays)

    def continuation(cache, token):
        start = time.perf_counter()
        times, tokens = [], []
        generator = generate_step(mx.array([token]), model_object, max_tokens=128, prompt_cache=cache)
        for token_id, _ in generator:
            tokens.append(int(token_id))
            times.append(time.perf_counter())
        generator.close()
        count = len(tokens)
        return tokens, {"first_token_s": times[0] - start if times else None,
                        "decode_tokens": max(0, count - 1),
                        "decode_s": times[-1] - times[0] if count > 1 else None,
                        "decode_tps": (count - 1) / (times[-1] - times[0]) if count > 1 else None,
                        "output_tokens": count}

    # No Core ML conversion, no custom kernels, no drafter, no KV quantization.
    mx.set_memory_limit(40 * 2**30)
    load_start = time.perf_counter()
    model_object, tokenizer = load(str(model), tokenizer_config={"trust_remote_code": False})
    mx.eval(model_object.parameters())
    emit(out / "results.jsonl", {"type": "loaded", "load_s": time.perf_counter() - load_start,
                               "mlx": importlib.metadata.version("mlx"),
                               "mlx_lm": importlib.metadata.version("mlx-lm"), **manifest})
    seed = tokenizer.encode("Synthetic benchmark. Continue a numbered list of simple observations.\n", add_special_tokens=False)
    body = tokenizer.encode("The garden has a red gate and a blue bench. Count the objects carefully.\n", add_special_tokens=False)
    for length in (512, 2304):
        ids = (seed + body * ((length // len(body)) + 2))[:length]
        for attempt in range(3):
            cache = make_prompt_cache(model_object)
            prefill_start = time.perf_counter()
            for offset in range(0, len(ids) - 1, 256):
                chunk = ids[offset:min(offset + 256, len(ids) - 1)]
                logits = model_object(mx.array([chunk]), cache=cache)
                mx.eval(logits)
                del logits
            sync_cache(cache)
            prefill_s = time.perf_counter() - prefill_start
            snapshot = out / "window-wrapped.safetensors"
            if length == 2304 and attempt == 0:
                save_start = time.perf_counter()
                save_prompt_cache(str(snapshot), cache, metadata={"model": str(model), "prefix_tokens": str(len(ids) - 1)})
                with snapshot.open("rb") as stream:
                    os.fsync(stream.fileno())
                emit(out / "results.jsonl", {"type": "snapshot_save", "prefix_tokens": len(ids) - 1,
                                           "save_fsync_s": time.perf_counter() - save_start,
                                           "bytes": snapshot.stat().st_size})
            expected, metrics = continuation(cache, ids[-1])
            emit(out / "results.jsonl", {"type": "decode", "prompt_tokens": len(ids), "try": attempt + 1,
                                       "prefill_s": prefill_s, "ttft_s": prefill_s + metrics["first_token_s"],
                                       "peak_memory_bytes": mx.get_peak_memory(), **metrics})
            del cache
            if length == 2304 and attempt == 0:
                restore_start = time.perf_counter()
                restored, metadata = load_prompt_cache(str(snapshot), return_metadata=True)
                sync_cache(restored)
                restore_s = time.perf_counter() - restore_start
                actual, replay_metrics = continuation(restored, ids[-1])
                equal = actual == expected
                emit(out / "results.jsonl", {"type": "snapshot_restore_same_process", "restore_s": restore_s,
                                           "same_token_ids": equal, "output_tokens": len(actual),
                                           "restore_to_first_token_s": restore_s + replay_metrics["first_token_s"]})
                del restored
                snapshot.unlink()
                if not equal:
                    raise SystemExit("Snapshot replay mismatch. Do not use restored KV in the demo.")
            gc.collect()
    emit(out / "results.jsonl", {"type": "DONE", "note": "Raw fixed-token decode; same-process serialization smoke, not fresh-process restore qualification."})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model, out = Path(args.model).expanduser().resolve(), Path(args.out).expanduser().resolve()
    manifest = preflight(model)
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 2**30:
        raise SystemExit("Leave at least 1 GiB of free disk for results and the temporary KV snapshot.")
    if args.worker:
        worker(model, out, manifest)
        return
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    emit(out / "results.jsonl", {"type": "START", "time": time.time(), "macos": platform.mac_ver()[0],
                               "nice": os.getpriority(os.PRIO_PROCESS, 0), "python": sys.version.split()[0],
                               "free_disk_bytes": shutil.disk_usage(out).free, "worker_cap_s": LIMIT_SECONDS})
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--model", str(model), "--out", str(out)]
    with (out / "run.log").open("w") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=LIMIT_SECONDS)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            emit(out / "results.jsonl", {"type": "STOPPED", "cap_s": LIMIT_SECONDS})
            raise SystemExit(124)
    print("Results:", out)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
