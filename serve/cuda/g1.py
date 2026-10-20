#!/usr/bin/env python3
"""G1 llama.cpp matrix: 6 items x 4 arms x 3 tries = 72 + 1x4x3 = 12 = 84 timed requests.

Standard CLI: python3 serve/cuda/g1.py --config $GLIMMER_HOME/config.json
Outputs: <results>/g1.jsonl (append+fsync each cell incl failed) + g1.manifest.json + g1.log
Resumable: skips ONLY successful matching-manifest cells. Failed/blocked cells re-run.
One untimed warmup per loaded arm. Concurrency 1. Greedy. Real TTFT via stream:true first chunk.
Full exact input/output IDs stored per cell. Native counters + acceptance. Peak VRAM per arm.
GPU cap 30 min (wrap the run in `timeout`). No wholesale rerun; use --only-cells for G6 repair.

Cohort (fixed): 5 heldout IDs + synth-4096-v1 (group 4K) + synth-32768-v1 (group 32K).
Context: server.context_4k/context_32k (default both 33536). One server per unique
(arm, context): 4 loads when equal (one untimed warmup each), else one per pair.
Strict success: native error, empty output, missing predicted timing, or
predicted_n != len(gen_ids) all FAIL the cell (never fake-skipped).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

HELDOUT_N = 512   # fallback if config missing (config sampling.n_predict_heldout wins)
SYNTH_N = 256     # fallback if config missing


def load_prompts_g1(path):
    """Load prompts_g1.jsonl -> {id: row}. Row needs prompt_ids + kind (heldout/synthetic)."""
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def build_manifest(cfg, prompts):
    """Build exact-count manifest bound to config+promptIDs+generation constants."""
    heldout = list(cfg["prompts"]["heldout_ids"])
    arms = list(cfg["arms"])
    samp = cfg["sampling"]
    n_held = int(samp.get("n_predict_heldout", HELDOUT_N))
    n_syn = int(samp.get("n_predict_synthetic", SYNTH_N))
    items_4k = heldout + ["synth-4096-v1"]
    items_32k = ["synth-32768-v1"]
    cells = []
    for item in items_4k:
        row = prompts.get(item)
        if row is None:
            # manifest still lists the cell as missing-prompt (blocked at runtime, preserved)
            prompt_sha, n_prompt, kind = "missing-prompt", 0, ("heldout" if item in heldout else "synthetic")
        else:
            ids = row.get("prompt_ids", [])
            prompt_sha = common.sha256_ids(ids) if ids else "empty"
            n_prompt = len(ids)
            kind = row.get("kind", "heldout" if item in heldout else "synthetic")
        for arm in arms:
            for t in (1, 2, 3):
                cells.append({
                    "cell_id": common.build_cell_id(item, arm, t),
                    "item_id": item, "arm": arm, "trial": t,
                    "group": "4K", "kind": kind,
                    "n_prompt": n_prompt, "prompt_ids_sha": prompt_sha,
                    "n_predict": n_held if kind == "heldout" else n_syn,
                    "context": int(cfg["server"].get("context_4k", 33536)),
                })
    for item in items_32k:
        row = prompts.get(item)
        if row is None:
            prompt_sha, n_prompt = "missing-prompt", 0
        else:
            ids = row.get("prompt_ids", [])
            prompt_sha = common.sha256_ids(ids) if ids else "empty"
            n_prompt = len(ids)
        for arm in arms:
            for t in (1, 2, 3):
                cells.append({
                    "cell_id": common.build_cell_id(item, arm, t),
                    "item_id": item, "arm": arm, "trial": t,
                    "group": "32K", "kind": "synthetic",
                    "n_prompt": n_prompt, "prompt_ids_sha": prompt_sha,
                    "n_predict": n_syn,
                    "context": int(cfg["server"].get("context_32k", 33536)),
                })
    manifest = {
        "job": "G1",
        "schema": 1,
        "config_hash": common.config_hash(cfg),
        "generation": {
            "temperature": samp.get("temperature", 0.0), "top_k": samp.get("top_k", 1),
            "top_p": samp.get("top_p", 1.0), "seed": samp.get("seed", 42),
            "cache_prompt": samp.get("cache_prompt", False), "stream": True,
            "n_predict_heldout": n_held, "n_predict_synthetic": n_syn,
        },
        "draft": dict(cfg.get("draft", {})),
        "arms": arms,
        "items_4k": items_4k, "items_32k": items_32k,
        "counts": {"cells_4k": 72, "cells_32k": 12, "total": 84},
        "cells": cells,
    }
    manifest["manifest_hash"] = common.manifest_hash(
        {k: v for k, v in manifest.items() if k != "manifest_hash"})
    return manifest


def plan_groups(cfg):
    """Server load plan: unique (arm, context) pairs, arm-major. Returns [(arm, ctx, [groups])].

    Equal contexts (33536/33536) -> 4 loads; differing -> one per distinct pair.
    Pure function (CPU-testable, no server).
    """
    c4 = int(cfg["server"].get("context_4k", 33536))
    c32 = int(cfg["server"].get("context_32k", 33536))
    groups = []
    for arm in list(cfg["arms"]):
        if c4 == c32:
            groups.append((arm, c4, ["4K", "32K"]))
        else:
            groups.append((arm, c4, ["4K"]))
            groups.append((arm, c32, ["32K"]))
    return groups


def server_cmd(cfg, arm, context, port):
    rt = cfg["runtime"]
    sv = cfg["server"]
    models = cfg["models"]
    draft_cfg = cfg.get("draft", {})
    binpath = rt.get("llama_server", "")
    cmd = [binpath, "-m", models["target"], "-ngl", str(sv.get("n_gpu_layers", 99)),
           "-fa", str(sv.get("flash_attn", "on")), "-np", str(sv.get("n_parallel", 1)),
           "-c", str(context), "--metrics", "--no-webui",
           "--host", sv.get("host", "127.0.0.1"), "--port", str(port)]
    if arm != "target-only":
        dkey = arm  # official/stock/tuned map to models keys
        dpath = models.get(dkey, "")
        cmd += ["-md", dpath, "-ngld", "all",
                "--spec-type", str(draft_cfg.get("spec_type", "draft-dflash")),
                "--spec-draft-n-max", str(draft_cfg.get("n_max", 15)),
                "--spec-draft-n-min", str(draft_cfg.get("n_min", 0)),
                "--spec-draft-p-min", str(draft_cfg.get("p_min", 0.0)),
                "--spec-draft-sampling", str(draft_cfg.get("sampling", "greedy"))]
    return cmd


def run_cell(base_url, cfg, cell, prompt_ids):
    """Run one timed cell. Returns record dict (success True/False), always with IDs + timings."""
    samp = cfg["sampling"]
    body = {"prompt": prompt_ids, "n_predict": int(cell["n_predict"]),
            "temperature": float(samp.get("temperature", 0.0)),
            "top_k": int(samp.get("top_k", 1)),
            "top_p": float(samp.get("top_p", 1.0)),
            "seed": int(samp.get("seed", 42)),
            "cache_prompt": bool(samp.get("cache_prompt", False)),
            "return_tokens": True, "stream": True, "id_slot": 0}
    m0 = common.get_metrics(base_url)
    try:
        r = common.completion_stream(base_url, body, timeout=1200)
    except Exception as e:
        m1 = common.get_metrics(base_url)
        return {"success": False, "error": "completion:%s" % str(e)[:800],
                "client_concurrency": 1, "queue_native": None,
                "spec": common.metrics_deltas(m0, m1)}
    m1 = common.get_metrics(base_url)
    final = r.get("final", {}) if isinstance(r.get("final"), dict) else {}
    # native error object in stream -> fail (no fake success)
    if isinstance(final.get("error"), dict) or isinstance(final.get("error"), str):
        return {"success": False, "error": "native:%s" % str(final.get("error"))[:500],
                "client_concurrency": 1, "queue_native": None,
                "ttft_ms": r.get("first_token_ms"), "wall_ms": r.get("wall_ms"),
                "chunks": r.get("chunks"), "timings": final.get("timings", {}),
                "spec": common.metrics_deltas(m0, m1)}
    timings = final.get("timings", {}) if isinstance(final.get("timings"), dict) else {}
    gen_ids = final.get("tokens", [])
    if not isinstance(gen_ids, list):
        gen_ids = []
    try:
        gen_ids = [int(x) for x in gen_ids]
    except Exception:
        pass
    spec = common.metrics_deltas(m0, m1)
    acc = common.acceptance(spec)
    # strict success: nonempty output, native predicted timing present, ID count match
    fails = []
    if len(gen_ids) == 0:
        fails.append("empty gen_ids (no tokens, no EOS evidence)")
    pred_n = timings.get("predicted_n")
    pred_ms = timings.get("predicted_ms")
    if pred_n is None or pred_ms is None:
        fails.append("missing native predicted_n/predicted_ms")
    elif int(pred_n) != len(gen_ids):
        fails.append("output-ID mismatch: predicted_n=%s != len(gen_ids)=%d" % (pred_n, len(gen_ids)))
    if len(gen_ids) > int(cell["n_predict"]):
        fails.append("n_gen=%d exceeds n_predict=%s" % (len(gen_ids), cell["n_predict"]))
    warns = common.check_lengths(cell, gen_ids, timings, cell["n_predict"])
    # explicit native cache counters (verified in server source: cache_n=reused, prompt_n=computed)
    cache = {"cache_n": timings.get("cache_n"), "prompt_n": timings.get("prompt_n"),
             "prompt_ms": timings.get("prompt_ms"),
             "predicted_n": pred_n, "predicted_ms": pred_ms}
    end_native = {"stop_type": final.get("stop_type"), "stopping_word": final.get("stopping_word"),
                  "truncated": final.get("truncated")}
    base = {"client_concurrency": 1, "queue_native": None,
            "ttft_ms": r.get("first_token_ms"), "wall_ms": r.get("wall_ms"),
            "chunks": r.get("chunks"), "stream_warnings": r.get("stream_warnings", []),
            "timings": timings, "native_cache": cache,
            "gen_ids": gen_ids, "n_gen": len(gen_ids),
            "gen_ids_sha": common.sha256_ids(gen_ids) if gen_ids else "empty",
            "spec": spec, "acceptance": acc,
            "length_warnings": warns, "end_native": end_native}
    if fails:
        base["success"] = False
        base["error"] = "; ".join(fails)[:800]
    else:
        base["success"] = True
    return base


def main():
    ap = argparse.ArgumentParser(description="G1 84-cell matrix (resumable, native counters).")
    ap.add_argument("--config", default=os.path.join(common.REMOTE_DIR, "config.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompts", default=None, help="prompts_g1.jsonl (default from config)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--only-cells", default=None, help="comma-separated cell_ids for G6 repair")
    ap.add_argument("--write-manifest", action="store_true", help="write manifest and exit (no server)")
    ap.add_argument("--dry-run", action="store_true", help="plan only: manifest counts + skip set, no server")
    ap.add_argument("--no-warmup", action="store_true", help="skip untimed warmup (not recommended)")
    ap.add_argument("--health-secs", type=int, default=240)
    args = ap.parse_args()
    common.install_signal_handlers()
    cfg = common.load_config(args.config)
    if cfg.get("_warnings"):
        print("G1 config warnings: %s" % cfg["_warnings"], flush=True)
    outdir = args.out or cfg.get("results_dir")
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "g1.log")
    logf = open(log_path, "a")

    def log(msg):
        line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    prompts_path = args.prompts or cfg["prompts"].get("g1_prompts")
    log("G1 config=%s prompts=%s outdir=%s" % (args.config, prompts_path, outdir))
    log("G1 config_hash=%s" % common.config_hash(cfg))
    if not prompts_path or not os.path.exists(prompts_path):
        log("G1 BLOCKED: prompts file missing: %s" % prompts_path)
        if args.write_manifest or args.dry_run:
            return 2
        # still write manifest with missing-prompt cells so the block is preserved
        prompts = {}
    else:
        prompts = load_prompts_g1(prompts_path)
        log("G1 loaded %d prompt rows" % len(prompts))
    manifest = build_manifest(cfg, prompts)
    man_path = os.path.join(outdir, "g1.manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    mhash = manifest["manifest_hash"]
    log("G1 manifest %d cells hash=%s" % (len(manifest["cells"]), mhash[:16]))
    assert len(manifest["cells"]) == 84, "manifest must be exactly 84 cells"
    results_path = os.path.join(outdir, "g1.jsonl")
    done = common.load_completed(results_path, mhash)
    log("G1 resumable: %d/%d already successful+matching, will skip" % (len(done), len(manifest["cells"])))
    only = set(x.strip() for x in (args.only_cells or "").split(",") if x.strip()) or None
    if only:
        log("G1 only-cells filter: %d" % len(only))
    if args.write_manifest or args.dry_run:
        todo = [c for c in manifest["cells"] if c["cell_id"] not in done and (only is None or c["cell_id"] in only)]
        log("G1 dry-run: %d todo, 0 servers started" % len(todo))
        for c in todo[:10]:
            log("  todo %s" % c["cell_id"])
        logf.close()
        return 0

    # group cells by (arm, context): restart server when context differs between groups
    base_port = args.port or int(cfg["server"].get("port_base", 18501))
    host = cfg["server"].get("host", "127.0.0.1")
    port = common.pick_free_port(base_port, host=host)
    log("G1 port=%d (base %d)" % (port, base_port))
    base_url = "http://%s:%d" % (host, port)
    ld_path = cfg["runtime"].get("ld_library_path", "")
    env_extra = {"LD_LIBRARY_PATH": ld_path} if ld_path and os.path.isdir(ld_path) else None
    # static per-run facts attached to every cell
    run_facts = {"manifest_hash": mhash, "config_hash": common.config_hash(cfg),
                 "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
                 "build_sha256": cfg["runtime"].get("build_sha256", "unknown"),
                 "llama_cpp_rev": cfg["runtime"].get("llama_cpp_rev", "unknown")}
    try:
        r = common.run_cmd(["nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader"], timeout=30)
        run_facts["gpu"] = r["out"].strip()[:300] if r["rc"] == 0 else "unknown"
    except Exception:
        run_facts["gpu"] = "unknown"

    # One server per unique (arm, context): 4 loads when contexts equal (33536/33536),
    # else one per distinct pair. One untimed warmup per LOADED server.
    groups = plan_groups(cfg)
    log("G1 server plan: %d loads %s" % (len(groups), [(a, c) for a, c, _ in groups]))
    n_done = 0
    for arm, ctx, group_names in groups:
        cells = [c for c in manifest["cells"] if c["arm"] == arm and c["group"] in group_names]
        todo = [c for c in cells if c["cell_id"] not in done and (only is None or c["cell_id"] in only)]
        if not todo:
            log("G1 skip %s/ctx%d: all %d done or filtered" % (arm, ctx, len(cells)))
            continue
        # need server for this (arm, ctx)
        cmd = server_cmd(cfg, arm, ctx, port)
        # pre-check model files exist (blocked cells, no silent substitution, no fallback)
        missing = []
        if not os.path.exists(cfg["models"]["target"]):
            missing.append("target")
        if arm != "target-only" and not os.path.exists(cfg["models"].get(arm, "")):
            missing.append(arm)
        if missing or not os.path.exists(cmd[0]):
            for c in todo:
                rec = dict(run_facts)
                rec.update({"job": "G1", "cell_id": c["cell_id"], "item_id": c["item_id"],
                            "arm": arm, "trial": c["trial"], "group": c["group"],
                            "success": False, "error": "blocked:missing %s bin=%s" % (missing, cmd[0]),
                            "n_prompt": c["n_prompt"], "prompt_ids_sha": c["prompt_ids_sha"],
                            "n_predict": c["n_predict"], "context": ctx})
                common.append_jsonl(results_path, rec)
                log("G1 BLOCKED %s missing=%s" % (c["cell_id"], missing))
            continue
        tag = "%s-ctx%d" % (arm, ctx)
        slog = os.path.join(outdir, "server-g1-%s.log" % tag)
        vlog = os.path.join(outdir, "gpu-g1-%s.log" % tag)
        # binary identity per loaded server (no fallback: primary only)
        server_sha = common.sha256_file(cmd[0])
        server_facts = {"server_sha": server_sha,
                        "server_sha_expected": cfg["runtime"].get("build_sha256", "unknown")}
        log("G1 start arm=%s ctx=%d groups=%s sha=%s" % (arm, ctx, group_names, server_sha[:16]))
        srv = common.start_server(cmd, slog, env_extra=env_extra)
        vsamp = common.start_vram_sampler(vlog)
        try:
            h = common.wait_health(base_url, secs=args.health_secs)
            if not h["ok"]:
                log("G1 server unhealthy %s last=%s" % (tag, h["last_status"]))
                for c in todo:
                    rec = dict(run_facts)
                    rec.update(server_facts)
                    rec.update({"job": "G1", "cell_id": c["cell_id"], "item_id": c["item_id"],
                                "arm": arm, "trial": c["trial"], "group": c["group"],
                                "success": False, "error": "blocked:server-unhealthy %s" % h["last_status"],
                                "n_prompt": c["n_prompt"], "prompt_ids_sha": c["prompt_ids_sha"],
                                "n_predict": c["n_predict"], "context": ctx})
                    common.append_jsonl(results_path, rec)
                continue
            log("G1 healthy %s waited=%.1fs" % (tag, h["waited_s"]))
            # one untimed warmup per LOADED server (128 tokens, uncounted)
            if not args.no_warmup:
                try:
                    wrow = prompts.get(todo[0]["item_id"], {})
                    wids = wrow.get("prompt_ids", [])[:2048] or [1, 2, 3]
                    wb = {"prompt": wids, "n_predict": 128, "temperature": 0.0, "top_k": 1,
                          "cache_prompt": False, "return_tokens": False, "stream": False}
                    common.completion_blocking(base_url, wb, timeout=600)
                    log("G1 warmup done %s (untimed, uncounted)" % tag)
                except Exception as e:
                    log("G1 warmup failed %s: %s" % (tag, str(e)[:300]))
            for c in todo:
                row = prompts.get(c["item_id"])
                if row is None or not row.get("prompt_ids"):
                    rec = dict(run_facts)
                    rec.update(server_facts)
                    rec.update({"job": "G1", "cell_id": c["cell_id"], "item_id": c["item_id"],
                                "arm": arm, "trial": c["trial"], "group": c["group"],
                                "success": False, "error": "blocked:prompt-missing",
                                "n_prompt": c["n_prompt"], "prompt_ids_sha": c["prompt_ids_sha"],
                                "n_predict": c["n_predict"], "context": ctx})
                    common.append_jsonl(results_path, rec)
                    log("G1 BLOCKED %s prompt-missing" % c["cell_id"])
                    continue
                pids = row["prompt_ids"]
                out = run_cell(base_url, cfg, c, pids)
                rec = dict(run_facts)
                rec.update(server_facts)
                rec.update({"job": "G1", "cell_id": c["cell_id"], "item_id": c["item_id"],
                            "arm": arm, "trial": c["trial"], "group": c["group"],
                            "kind": c["kind"], "n_prompt": len(pids),
                            "prompt_ids": pids, "prompt_ids_sha": common.sha256_ids(pids),
                            "n_predict": c["n_predict"], "context": ctx})
                rec.update(out)
                rec["vram_sample_peak_mib"] = common.peak_vram_mib(vlog)
                common.append_jsonl(results_path, rec)
                if rec.get("success"):
                    done.add(c["cell_id"])
                n_done += 1
                t = (rec.get("timings") or {})
                log("G1 %s gen=%s ttft=%s dec=%s/%s acc=%s %s" % (
                    c["cell_id"], rec.get("n_gen"), rec.get("ttft_ms"),
                    t.get("predicted_n"), t.get("predicted_ms"),
                    (rec.get("acceptance") or {}).get("accept_len"), "OK" if rec.get("success") else "FAIL"))
        finally:
            common.stop_vram_sampler(vsamp)
            common.stop_server(srv)
            peak = common.peak_vram_mib(vlog)
            log("G1 stop %s peak_vram=%s" % (tag, peak))
    log("G1 finished: %d cells newly written, %d/%d successful+matching" % (
        n_done, len(common.load_completed(results_path, mhash)), len(manifest["cells"])))
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
