#!/usr/bin/env python3
"""G2 KV snapshot + 30K prefix cache: 12 timed save/restore ops + 12 continuations + 9 prefix = 33 records.

Standard CLI: python3 serve/cuda/g2.py --config $GLIMMER_HOME/config.json
Outputs: <results>/g2.jsonl (append+fsync each cell incl failed) + g2.manifest.json + g2.log
Resumable: skips ONLY successful matching-manifest cells. Failed/blocked cells re-run.
Target-only (no speculation), concurrency 1, greedy temp0/top_k1/seed42. One untimed warmup per load.
GPU cap 25 min (wrap the run in `timeout`). No wholesale rerun; use --only-cells for G6 repair.

Snapshot (24 records): occupied 8192/32768 x (save + fresh-ref + restore + restored) x 3 tries.
  - Prefill P_N (n_predict=0, cache_prompt:false) -> prove boundary via GET /slots n_prompt_tokens==N.
  - Save -> verify n_saved==N, stat bytes + explicit fsync ms.
  - Fresh ref: erase+verify empty, prompt P_N+S_fix(64), n_predict=64, cache_prompt:false, stream:true.
  - Restore: erase+verify empty, restore file, verify n_restored==N + slot N.
  - Restored: prompt P_N+S_fix, n_predict=64, cache_prompt:true, stream:true. Expect cache_n~=N, prompt_n~=64.
    If cache_n<N or prompt_n>64+tol -> pure_restore=false + recomputation note (SWA ring not purely restored).
  - Parity: raw token-ID equality (no retokenization). Empty on either side = parity FAIL, not success.
Prefix (9 records): cold empty / hit same-slot P+S2 / miss P'+S2 early-change x 3 tries, n_predict=16, stream TTFT.
  - Reuse ONLY from timings.cache_n (reused) / timings.prompt_n (computed). tokens_cached is TOTAL consumed, not reuse.
  - Cold: erase+verify empty, cache_prompt:false. Hit/miss: same slot, cache_prompt:true, no intermediate erase.

Prompts: prefers <prompts>/prompts_g2.jsonl (7 explicit rows) if present; else deterministic fallback from
  prompts_g1.jsonl (synth-32768-v1 + synth-4096-v1 slices, valid-IDs-only).
  Derivation method recorded per prompt in manifest + each cell.
Context: server.context_g2 (default 33536). One server load for all G2 (target-only, --slot-save-path DIR).
Restore mode: empty-slot (erase + GET /slots verify). Process restart not needed; erase+verify proves empty.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

# ---- G2 documented constants (new benchmark settings, not model changes) ----
G2_DEFAULTS = {
    "server": {
        "context_g2": 33536,
        "slot_save_dir": common.REMOTE_DIR + "/slots",
    },
    "prompts": {
        "g2_prompts": common.REMOTE_DIR + "/prompts/prompts_g2.jsonl",
    },
    "g2": {
        "snap_lengths": [8192, 32768],
        "snap_suffix": 64,
        "snap_continuation": 64,
        "prefix_len": 30720,
        "prefix_suffix": 64,
        "prefix_n_predict": 16,
        "prefix_miss_head": 16,
        "slot": 0,
        "trials": [1, 2, 3],
        "recompute_tol": 16,
    },
}

ARM = "target-only"


def load_config_g2(path):
    cfg = common.load_config(path)
    # merge G2 defaults (file wins)
    for section, defaults in G2_DEFAULTS.items():
        cur = cfg.get(section, {})
        if not isinstance(cur, dict):
            cur = {}
        merged = dict(defaults)
        merged.update(cur)
        cfg[section] = merged
    return cfg


def load_prompts_g1_rows(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def load_prompts_g2(cfg):
    """Return (prompts_dict, derivation_dict, source_note).

    prompts_dict keys: snap_8192, snap_32768, g2_suffix, prefix, suffix_a, suffix_b, prefix_miss.
    Each value: {"ids": [...], "id": row_id, "sha": ...}.
    derivation: key -> "explicit:<row>" or "fallback:<rule>".
    """
    g2_path = cfg["prompts"].get("g2_prompts", "")
    g = cfg["g2"]
    want_len = {
        "snap_8192": g["snap_lengths"][0],
        "snap_32768": g["snap_lengths"][1],
        "g2_suffix": g["snap_suffix"],
        "prefix": g["prefix_len"],
        "suffix_a": g["prefix_suffix"],
        "suffix_b": g["prefix_suffix"],
        "prefix_miss": g["prefix_len"],
    }
    # Preferred: explicit prompts_g2.jsonl
    explicit_ids = {
        "snap_8192": "snap-8192-v1",
        "snap_32768": "snap-32768-v1",
        "g2_suffix": "g2-suffix-64-v1",
        "prefix": "prefix-30720-v1",
        "suffix_a": "prefix-suffix-a-64-v1",
        "suffix_b": "prefix-suffix-b-64-v1",
        "prefix_miss": "prefix-miss-30720-v1",
    }
    if isinstance(g2_path, str) and g2_path and os.path.exists(g2_path):
        rows = load_prompts_g1_rows(g2_path)
        prompts, deriv = {}, {}
        missing = []
        for key, row_id in explicit_ids.items():
            row = rows.get(row_id)
            if row is None or not row.get("prompt_ids"):
                missing.append(row_id)
                continue
            ids = row["prompt_ids"]
            prompts[key] = {"ids": ids, "id": row_id,
                            "sha": common.sha256_ids(ids) if ids else "empty"}
            deriv[key] = "explicit:%s" % row_id
        if not missing:
            # validate lengths
            for key, exp in want_len.items():
                if len(prompts[key]["ids"]) != exp:
                    raise ValueError("explicit %s len %d != %d" % (key, len(prompts[key]["ids"]), exp))
            # validate suffix_b differs + miss head differs
            if prompts["suffix_a"]["ids"] == prompts["suffix_b"]["ids"]:
                raise ValueError("explicit suffix_a == suffix_b")
            if prompts["prefix"]["ids"][:g["prefix_miss_head"]] == prompts["prefix_miss"]["ids"][:g["prefix_miss_head"]]:
                raise ValueError("explicit prefix_miss head == prefix head")
            return prompts, deriv, "explicit:%s" % g2_path
        # fall through to fallback if explicit file incomplete (recorded as blocked? no: use fallback, note missing)
        note_missing = "explicit file incomplete, missing %s; using fallback" % missing
    else:
        note_missing = "explicit file absent (%s); using fallback" % (g2_path or "none",)
    # Fallback: derive from prompts_g1.jsonl
    g1_path = cfg["prompts"].get("g1_prompts", "")
    if not isinstance(g1_path, str) or not g1_path or not os.path.exists(g1_path):
        raise FileNotFoundError("fallback needs prompts_g1.jsonl path string, got %r (%s)" % (g1_path, note_missing))
    rows = load_prompts_g1_rows(g1_path)
    s32 = (rows.get("synth-32768-v1") or {}).get("prompt_ids", [])
    s4 = (rows.get("synth-4096-v1") or {}).get("prompt_ids", [])
    if len(s32) != 32768 or len(s4) != 4096:
        raise ValueError("fallback needs synth-32768-v1(32768)+synth-4096-v1(4096), got %d/%d" % (len(s32), len(s4)))
    n8192, n32768 = g["snap_lengths"]
    plen, slen, mhead = g["prefix_len"], g["prefix_suffix"], g["prefix_miss_head"]
    snap_8192 = s32[:n8192]
    snap_32768 = list(s32)
    g2_suffix = s4[0:g["snap_suffix"]]
    prefix = s32[:plen]
    suffix_a = s4[64:64 + slen]
    suffix_b = s4[128:128 + slen]
    miss_head = list(s4[192:192 + mhead])
    if len(suffix_a) != slen or len(suffix_b) != slen or len(miss_head) != mhead:
        raise ValueError("fallback slice short (pool layout changed)")
    if suffix_a == suffix_b:
        # shift b by one (still valid IDs, deterministic)
        suffix_b = s4[129:129 + slen]
    prefix_miss = list(prefix)
    if miss_head == prefix[:mhead]:
        miss_head = list(s4[193:193 + mhead])
    prefix_miss[:mhead] = miss_head
    prompts = {
        "snap_8192": {"ids": snap_8192, "id": "fallback:snap-8192", "sha": common.sha256_ids(snap_8192)},
        "snap_32768": {"ids": snap_32768, "id": "fallback:snap-32768", "sha": common.sha256_ids(snap_32768)},
        "g2_suffix": {"ids": g2_suffix, "id": "fallback:g2-suffix", "sha": common.sha256_ids(g2_suffix)},
        "prefix": {"ids": prefix, "id": "fallback:prefix", "sha": common.sha256_ids(prefix)},
        "suffix_a": {"ids": suffix_a, "id": "fallback:suffix_a", "sha": common.sha256_ids(suffix_a)},
        "suffix_b": {"ids": suffix_b, "id": "fallback:suffix_b", "sha": common.sha256_ids(suffix_b)},
        "prefix_miss": {"ids": prefix_miss, "id": "fallback:prefix_miss", "sha": common.sha256_ids(prefix_miss)},
    }
    deriv = {
        "snap_8192": "fallback:S32[:8192]",
        "snap_32768": "fallback:S32(full)",
        "g2_suffix": "fallback:S4[0:64]",
        "prefix": "fallback:S32[:30720]",
        "suffix_a": "fallback:S4[64:128]",
        "suffix_b": "fallback:S4[128:192]",
        "prefix_miss": "fallback:prefix+head=S4[192:208]",
    }
    return prompts, deriv, "fallback:%s (%s)" % (g1_path, note_missing)


def build_manifest_g2(cfg, prompts, deriv, source_note):
    g = cfg["g2"]
    samp = cfg["sampling"]
    ctx = int(cfg["server"].get("context_g2", 33536))
    trials = list(g["trials"])
    cells = []
    # snapshot cells
    for n in g["snap_lengths"]:
        key = "snap_8192" if n == 8192 else "snap_32768"
        pkey = prompts.get(key, {})
        sfix = prompts.get("g2_suffix", {})
        full_sha = "missing"
        if pkey.get("ids") and sfix.get("ids"):
            full_sha = common.sha256_ids(list(pkey["ids"]) + list(sfix["ids"]))
        for t in trials:
            base = {"arm": ARM, "trial": t, "group": "snapshot", "occupied": n,
                    "context": ctx, "n_prompt_save": n,
                    "prompt_save_sha": pkey.get("sha", "missing"),
                    "suffix_sha": sfix.get("sha", "missing"),
                    "n_prompt_full": n + int(g["snap_suffix"]) if pkey.get("ids") else 0,
                    "prompt_full_sha": full_sha,
                    "n_predict_cont": int(g["snap_continuation"]),
                    "derivation": "%s|%s" % (deriv.get(key, "?"), deriv.get("g2_suffix", "?"))}
            for kind in ("snap-save-%d" % n, "snap-ref-%d" % n, "snap-restore-%d" % n, "snap-restored-%d" % n):
                cells.append(dict(base, cell_id=common.build_cell_id(kind, ARM, t), item_id=kind))
    # prefix cells
    p = prompts.get("prefix", {})
    pm = prompts.get("prefix_miss", {})
    sa = prompts.get("suffix_a", {})
    sb = prompts.get("suffix_b", {})
    cold_sha = common.sha256_ids(list(p["ids"]) + list(sa["ids"])) if p.get("ids") and sa.get("ids") else "missing"
    hit_sha = common.sha256_ids(list(p["ids"]) + list(sb["ids"])) if p.get("ids") and sb.get("ids") else "missing"
    miss_sha = common.sha256_ids(list(pm["ids"]) + list(sb["ids"])) if pm.get("ids") and sb.get("ids") else "missing"
    for t in trials:
        for kind, sha in (("prefix-cold", cold_sha), ("prefix-hit", hit_sha), ("prefix-miss", miss_sha)):
            cells.append({"cell_id": common.build_cell_id(kind, ARM, t), "item_id": kind,
                          "arm": ARM, "trial": t, "group": "prefix",
                          "n_prompt": int(g["prefix_len"]) + int(g["prefix_suffix"]),
                          "prompt_ids_sha": sha, "prefix_sha": p.get("sha", "missing"),
                          "prefix_miss_sha": pm.get("sha", "missing"),
                          "suffix_a_sha": sa.get("sha", "missing"), "suffix_b_sha": sb.get("sha", "missing"),
                          "n_predict": int(g["prefix_n_predict"]), "context": ctx,
                          "derivation": "%s|%s|%s|%s" % (deriv.get("prefix", "?"), deriv.get("prefix_miss", "?"),
                                                         deriv.get("suffix_a", "?"), deriv.get("suffix_b", "?"))})
    manifest = {
        "job": "G2",
        "schema": 1,
        "config_hash": common.config_hash(cfg),
        "prompts_source": source_note,
        "prompts_sha": {k: v.get("sha", "missing") for k, v in prompts.items()},
        "prompts_derivation": dict(deriv),
        "generation": {
            "temperature": float(samp.get("temperature", 0.0)),
            "top_k": int(samp.get("top_k", 1)),
            "top_p": float(samp.get("top_p", 1.0)),
            "seed": int(samp.get("seed", 42)),
            "snap_suffix": int(g["snap_suffix"]),
            "snap_continuation": int(g["snap_continuation"]),
            "prefix_n_predict": int(g["prefix_n_predict"]),
            "slot": int(g["slot"]),
        },
        "server": {"context_g2": ctx,
                   "slot_save_dir": cfg["server"].get("slot_save_dir", "")},
        "arms": [ARM],
        "counts": {"snapshot": 24, "prefix": 9, "total": 33,
                   "timed_save_restore": 12, "continuations": 12, "prefix_requests": 9},
        "cells": cells,
    }
    manifest["manifest_hash"] = common.manifest_hash(
        {k: v for k, v in manifest.items() if k != "manifest_hash"})
    return manifest


def server_cmd_g2(cfg, context, port):
    rt = cfg["runtime"]
    sv = cfg["server"]
    models = cfg["models"]
    binpath = rt.get("llama_server", "")
    slot_dir = sv.get("slot_save_dir", G2_DEFAULTS["server"]["slot_save_dir"])
    cmd = [binpath, "-m", models["target"], "-ngl", str(sv.get("n_gpu_layers", 99)),
           "-fa", str(sv.get("flash_attn", "on")), "-np", str(sv.get("n_parallel", 1)),
           "-c", str(context), "--metrics", "--no-webui",
           "--slot-save-path", slot_dir,
           "--host", sv.get("host", "127.0.0.1"), "--port", str(port)]
    return cmd


# ---- slot helpers (local; filenames are basenames only) ----

def get_slots(base_url, timeout=60):
    st, obj = common.get_json(base_url.rstrip("/") + "/slots", timeout=timeout)
    if st != 200:
        raise RuntimeError("GET /slots HTTP %s: %s" % (st, str(obj)[:400]))
    return obj


def slot_n_tokens(base_url, slot_id, timeout=60):
    slots = get_slots(base_url, timeout=timeout)
    if not isinstance(slots, list):
        raise RuntimeError("GET /slots: expected list, got %s" % type(slots))
    for s in slots:
        if isinstance(s, dict) and int(s.get("id", -1)) == int(slot_id):
            # native total consumed in slot (prompt tokens held)
            for k in ("n_prompt_tokens", "n_tokens", "n_past"):
                if k in s:
                    return int(s[k]), s
            return None, s
    raise RuntimeError("slot %d not found in /slots (%d slots)" % (slot_id, len(slots)))


def slot_erase(base_url, slot_id, timeout=120):
    st, obj = common.post_json(base_url.rstrip("/") + "/slots/%d?action=erase" % int(slot_id),
                               {}, timeout=timeout)
    if st != 200:
        raise RuntimeError("erase HTTP %s: %s" % (st, str(obj)[:400]))
    return obj


def slot_save(base_url, slot_id, filename, timeout=300):
    if "/" in filename or "\\" in filename or not filename:
        raise ValueError("filename must be a basename: %r" % filename)
    st, obj = common.post_json(base_url.rstrip("/") + "/slots/%d?action=save" % int(slot_id),
                               {"filename": filename}, timeout=timeout)
    if st != 200:
        raise RuntimeError("save HTTP %s: %s" % (st, str(obj)[:500]))
    return obj


def slot_restore(base_url, slot_id, filename, timeout=300):
    if "/" in filename or "\\" in filename or not filename:
        raise ValueError("filename must be a basename: %r" % filename)
    t0 = time.time()
    st, obj = common.post_json(base_url.rstrip("/") + "/slots/%d?action=restore" % int(slot_id),
                               {"filename": filename}, timeout=timeout)
    wall_ms = (time.time() - t0) * 1000.0
    if st != 200:
        raise RuntimeError("restore HTTP %s: %s" % (st, str(obj)[:500]))
    if isinstance(obj, dict):
        obj["_wall_ms"] = wall_ms
    return obj


def stat_and_fsync(path):
    """Return {bytes, fsync_ms, fsync_ok, error?}. Opens read-only, fsyncs the fd."""
    t0 = time.time()
    try:
        n = os.path.getsize(path)
    except Exception as e:
        return {"bytes": None, "fsync_ms": None, "fsync_ok": False, "error": "stat:%s" % str(e)[:300]}
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return {"bytes": n, "fsync_ms": (time.time() - t0) * 1000.0, "fsync_ok": True}
    except Exception as e:
        return {"bytes": n, "fsync_ms": (time.time() - t0) * 1000.0, "fsync_ok": False,
                "error": "fsync:%s" % str(e)[:300]}


def ensure_empty_slot(base_url, slot_id):
    """Erase + verify 0. Returns {erase_resp, n_after} or raises."""
    er = slot_erase(base_url, slot_id)
    n_after, _ = slot_n_tokens(base_url, slot_id)
    if n_after is None or int(n_after) != 0:
        raise RuntimeError("slot %d emptiness not proved after erase: n=%s" % (slot_id, n_after))
    return {"erase_resp": er, "n_after": n_after}


# ---- completion helpers ----

def greedy_body(prompt_ids, n_predict, seed, cache_prompt, slot_id, stream, return_tokens=True):
    return {"prompt": prompt_ids, "n_predict": int(n_predict),
            "temperature": 0.0, "top_k": 1, "top_p": 1.0,
            "seed": int(seed), "cache_prompt": bool(cache_prompt),
            "return_tokens": bool(return_tokens), "stream": bool(stream),
            "id_slot": int(slot_id)}


def run_prefill_blocking(base_url, prompt_ids, seed, slot_id, timeout=1200):
    """Prefill-only (n_predict=0). Returns {wall_ms, resp} or raises."""
    body = greedy_body(prompt_ids, 0, seed, False, slot_id, False, return_tokens=False)
    wall_ms, resp = common.completion_blocking(base_url, body, timeout=timeout)
    return {"wall_ms": wall_ms, "resp": resp}


def run_continuation_stream(base_url, prompt_ids, n_predict, seed, cache_prompt, slot_id, timeout=1200):
    """Stream continuation with real TTFT. Returns dict with ttft/wall/timings/tokens/cached/computed."""
    body = greedy_body(prompt_ids, n_predict, seed, cache_prompt, slot_id, True, return_tokens=True)
    m0 = common.get_metrics(base_url)
    r = common.completion_stream(base_url, body, timeout=timeout)
    m1 = common.get_metrics(base_url)
    final = r.get("final", {}) if isinstance(r.get("final"), dict) else {}
    timings = final.get("timings", {}) if isinstance(final.get("timings"), dict) else {}
    toks = final.get("tokens", [])
    if not isinstance(toks, list):
        toks = []
    if final.get("error"):
        raise RuntimeError("native completion error: %s" % final["error"])
    if not toks or timings.get("predicted_n") != len(toks) or timings.get("predicted_ms") is None:
        raise RuntimeError("incomplete output IDs/native timings: tokens=%d predicted_n=%r" % (len(toks), timings.get("predicted_n")))
    spec = common.metrics_deltas(m0, m1)
    return {"ttft_ms": r.get("first_token_ms"), "wall_ms": r.get("wall_ms"),
            "chunks": r.get("chunks"), "timings": timings, "tokens": toks,
            "cache_n": timings.get("cache_n"), "prompt_n": timings.get("prompt_n"),
            "prompt_ms": timings.get("prompt_ms"),
            "tokens_cached_total": final.get("tokens_cached"),
            "tokens_evaluated": final.get("tokens_evaluated"),
            "tokens_predicted": final.get("tokens_predicted"),
            "stop_type": final.get("stop_type"), "stopping_word": final.get("stopping_word"),
            "truncated": final.get("truncated"), "spec": spec,
            "acceptance": common.acceptance(spec)}


def main():
    ap = argparse.ArgumentParser(description="G2 snapshot+prefix (33 records, resumable, native counters).")
    ap.add_argument("--config", default=os.path.join(common.REMOTE_DIR, "config.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--prompts", default=None, help="prompts_g2.jsonl override (default from config)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--slot", type=int, default=None, help="slot id (default from g2.slot)")
    ap.add_argument("--only-cells", default=None, help="comma-separated cell_ids for G6 repair")
    ap.add_argument("--write-manifest", action="store_true", help="write manifest and exit (no server)")
    ap.add_argument("--dry-run", action="store_true", help="plan only: manifest counts + skip set, no server")
    ap.add_argument("--no-warmup", action="store_true", help="skip untimed warmup (not recommended)")
    ap.add_argument("--health-secs", type=int, default=240)
    args = ap.parse_args()
    common.install_signal_handlers()
    cfg = load_config_g2(args.config)
    if args.prompts:
        cfg["prompts"]["g2_prompts"] = args.prompts
    g = cfg["g2"]
    slot_id = int(args.slot if args.slot is not None else g.get("slot", 0))
    seed = int(cfg["sampling"].get("seed", 42))
    ctx = int(cfg["server"].get("context_g2", 33536))
    slot_dir = cfg["server"].get("slot_save_dir", G2_DEFAULTS["server"]["slot_save_dir"])
    outdir = args.out or cfg.get("results_dir")
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "g2.log")
    logf = open(log_path, "a")

    def log(msg):
        line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log("G2 config=%s outdir=%s slot=%d ctx=%d" % (args.config, outdir, slot_id, ctx))
    log("G2 config_hash=%s" % common.config_hash(cfg))
    try:
        prompts, deriv, source_note = load_prompts_g2(cfg)
        log("G2 prompts source: %s" % source_note)
        for k, v in prompts.items():
            log("G2 prompt %s len=%d sha=%s (%s)" % (k, len(v["ids"]), v["sha"][:16], deriv.get(k)))
    except Exception as e:
        log("G2 BLOCKED: prompts load failed: %s" % str(e)[:600])
        if args.write_manifest or args.dry_run:
            logf.close()
            return 2
        # write manifest with missing prompts so the block is preserved as data
        prompts, deriv, source_note = {}, {}, "missing:%s" % e
        # fabricate empty prompts so manifest lists blocked cells
        for k in ("snap_8192", "snap_32768", "g2_suffix", "prefix", "suffix_a", "suffix_b", "prefix_miss"):
            prompts[k] = {"ids": [], "id": "missing", "sha": "missing-prompt"}
            deriv[k] = "missing"
    manifest = build_manifest_g2(cfg, prompts, deriv, source_note)
    man_path = os.path.join(outdir, "g2.manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    mhash = manifest["manifest_hash"]
    log("G2 manifest %d cells hash=%s" % (len(manifest["cells"]), mhash[:16]))
    assert len(manifest["cells"]) == 33, "manifest must be exactly 33 cells"
    results_path = os.path.join(outdir, "g2.jsonl")
    done = common.load_completed(results_path, mhash)
    log("G2 resumable: %d/%d already successful+matching, will skip" % (len(done), len(manifest["cells"])))
    only = set(x.strip() for x in (args.only_cells or "").split(",") if x.strip()) or None
    if only:
        log("G2 only-cells filter: %d" % len(only))
    if args.write_manifest or args.dry_run:
        todo = [c for c in manifest["cells"] if c["cell_id"] not in done and (only is None or c["cell_id"] in only)]
        log("G2 dry-run: %d todo, 0 servers started" % len(todo))
        for c in todo[:12]:
            log("  todo %s" % c["cell_id"])
        logf.close()
        return 0

    # ---- server lifecycle (one load for all G2) ----
    base_port = args.port or int(cfg["server"].get("port_base", 18501))
    host = cfg["server"].get("host", "127.0.0.1")
    port = common.pick_free_port(base_port, host=host)
    log("G2 port=%d (base %d) slot_dir=%s" % (port, base_port, slot_dir))
    base_url = "http://%s:%d" % (host, port)
    ld_path = cfg["runtime"].get("ld_library_path", "")
    env_extra = {"LD_LIBRARY_PATH": ld_path} if ld_path and os.path.isdir(ld_path) else None
    run_facts = {"manifest_hash": mhash, "config_hash": common.config_hash(cfg),
                 "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
                 "build_sha256": cfg["runtime"].get("build_sha256", "unknown"),
                 "llama_cpp_rev": cfg["runtime"].get("llama_cpp_rev", "unknown"),
                 "slot": slot_id, "context": ctx, "seed": seed}
    try:
        r = common.run_cmd(["nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader"], timeout=30)
        run_facts["gpu"] = r["out"].strip()[:300] if r["rc"] == 0 else "unknown"
    except Exception:
        run_facts["gpu"] = "unknown"

    # pre-checks (blocked cells, no silent substitution)
    binpath = cfg["runtime"].get("llama_server", "")
    target = cfg["models"].get("target", "")
    pre_block = None
    if not binpath or not os.path.exists(binpath):
        pre_block = "blocked:runtime-missing %s" % binpath
    elif not target or not os.path.exists(target):
        pre_block = "blocked:model-missing target=%s" % target
    elif any(not prompts.get(k, {}).get("ids") for k in
             ("snap_8192", "snap_32768", "g2_suffix", "prefix", "suffix_a", "suffix_b", "prefix_miss")):
        pre_block = "blocked:prompt-missing (see manifest prompts_sha)"
    if pre_block:
        cells = [c for c in manifest["cells"] if c["cell_id"] not in done and (only is None or c["cell_id"] in only)]
        for c in cells:
            rec = dict(run_facts)
            rec.update({"job": "G2", "cell_id": c["cell_id"], "item_id": c["item_id"],
                        "arm": ARM, "trial": c.get("trial"), "group": c.get("group"),
                        "success": False, "error": pre_block})
            common.append_jsonl(results_path, rec)
        log("G2 %s: wrote %d blocked cells" % (pre_block, len(cells)))
        logf.close()
        return 0

    try:
        os.makedirs(slot_dir, exist_ok=True)
    except Exception as e:
        log("G2 slot_dir create failed: %s" % str(e)[:300])
    cmd = server_cmd_g2(cfg, ctx, port)
    slog = os.path.join(outdir, "server-g2.log")
    vlog = os.path.join(outdir, "gpu-g2.log")
    log("G2 start ctx=%d cmd=%s" % (ctx, " ".join(cmd[:8]) + " ..."))
    srv = common.start_server(cmd, slog, env_extra=env_extra)
    vsamp = common.start_vram_sampler(vlog)
    n_new = 0
    try:
        h = common.wait_health(base_url, secs=args.health_secs)
        if not h["ok"]:
            log("G2 server unhealthy last=%s" % (h["last_status"],))
            for c in manifest["cells"]:
                if c["cell_id"] in done or (only is not None and c["cell_id"] not in only):
                    continue
                rec = dict(run_facts)
                rec.update({"job": "G2", "cell_id": c["cell_id"], "item_id": c["item_id"],
                            "arm": ARM, "trial": c.get("trial"), "group": c.get("group"),
                            "success": False, "error": "blocked:server-unhealthy %s" % h["last_status"]})
                common.append_jsonl(results_path, rec)
            logf.close()
            return 0
        log("G2 healthy waited=%.1fs" % h["waited_s"])
        # props for identity (build/model/ctx)
        try:
            _, props = common.get_json(base_url.rstrip("/") + "/props", timeout=60)
            if isinstance(props, dict):
                run_facts["build_info"] = str(props.get("build_info", ""))[:300]
                run_facts["model_path"] = str(props.get("model_path", ""))[:500]
                dgs = props.get("default_generation_settings", {})
                if isinstance(dgs, dict):
                    run_facts["n_ctx"] = dgs.get("n_ctx")
                run_facts["total_slots"] = props.get("total_slots")
        except Exception as e:
            run_facts["props_error"] = str(e)[:300]
        log("G2 props build=%s model=%s n_ctx=%s" % (
            run_facts.get("build_info"), run_facts.get("model_path"), run_facts.get("n_ctx")))
        # one untimed warmup (small slice, uncounted, not written)
        if not args.no_warmup:
            try:
                wids = prompts["snap_8192"]["ids"][:512]
                wb = greedy_body(wids, 32, seed, False, slot_id, False, return_tokens=False)
                common.completion_blocking(base_url, wb, timeout=600)
                try:
                    slot_erase(base_url, slot_id)
                except Exception:
                    pass
                log("G2 warmup done (untimed, uncounted, slot erased after)")
            except Exception as e:
                log("G2 warmup failed: %s" % str(e)[:300])

        def need(cell_id):
            return cell_id not in done and (only is None or cell_id in only)

        # ---- snapshot: for each N, for each trial: save -> ref -> restore -> restored ----
        for n in g["snap_lengths"]:
            key = "snap_8192" if n == 8192 else "snap_32768"
            pN = prompts[key]["ids"]
            sfix = prompts["g2_suffix"]["ids"]
            pFull = list(pN) + list(sfix)
            pN_sha = common.sha256_ids(pN)
            pFull_sha = common.sha256_ids(pFull)
            wraps = n // 2048
            for t in g["trials"]:
                fname = "g2-snap-%d-t%d.bin" % (n, t)
                fpath = os.path.join(slot_dir, fname)
                # --- save cell (prefill + save) ---
                cid_save = common.build_cell_id("snap-save-%d" % n, ARM, t)
                if need(cid_save):
                    rec = dict(run_facts)
                    rec.update({"job": "G2", "cell_id": cid_save, "item_id": "snap-save-%d" % n,
                                "arm": ARM, "trial": t, "group": "snapshot", "occupied": n,
                                "wraps_2k": wraps, "n_prompt": n, "prompt_ids": pN,
                                "prompt_ids_sha": pN_sha, "boundary": n,
                                "filename": fname, "derivation": deriv.get(key, "?")})
                    try:
                        ensure_empty_slot(base_url, slot_id)
                        pr = run_prefill_blocking(base_url, pN, seed, slot_id)
                        resp = pr["resp"] if isinstance(pr["resp"], dict) else {}
                        timings = resp.get("timings", {}) if isinstance(resp.get("timings"), dict) else {}
                        n_slot, _ = slot_n_tokens(base_url, slot_id)
                        rec["prefill"] = {"wall_ms": pr["wall_ms"], "timings": timings,
                                          "tokens_evaluated": resp.get("tokens_evaluated"),
                                          "tokens_cached_total": resp.get("tokens_cached"),
                                          "truncated": resp.get("truncated"),
                                          "slot_n_tokens": n_slot}
                        if resp.get("truncated"):
                            raise RuntimeError("prefill truncated (ctx %d < %d)" % (ctx, n))
                        if n_slot is not None and int(n_slot) != n:
                            raise RuntimeError("boundary not proved: slot n=%s != %d" % (n_slot, n))
                        sv = slot_save(base_url, slot_id, fname)
                        rec["save"] = sv
                        if int(sv.get("n_saved", -1)) != n:
                            raise RuntimeError("n_saved %s != %d" % (sv.get("n_saved"), n))
                        st = stat_and_fsync(fpath)
                        rec["file"] = {"path": fpath, "bytes": st.get("bytes"),
                                       "fsync_ms": st.get("fsync_ms"), "fsync_ok": st.get("fsync_ok"),
                                       "n_written_native": sv.get("n_written")}
                        if not st.get("fsync_ok") or not st.get("bytes"):
                            raise RuntimeError("snapshot durable write not proved: %r" % st)
                        rec["success"] = True
                    except Exception as e:
                        rec["success"] = False
                        rec["error"] = str(e)[:800]
                    rec["vram_sample_peak_mib"] = common.peak_vram_mib(vlog)
                    common.append_jsonl(results_path, rec)
                    if rec.get("success"):
                        done.add(cid_save)
                    n_new += 1
                    log("G2 %s %s" % (cid_save, "OK" if rec.get("success") else "FAIL %s" % rec.get("error", "")[:120]))
                # --- fresh ref cell (independent empty state) ---
                cid_ref = common.build_cell_id("snap-ref-%d" % n, ARM, t)
                ref_ids, ref_stop = None, None
                if need(cid_ref):
                    rec = dict(run_facts)
                    rec.update({"job": "G2", "cell_id": cid_ref, "item_id": "snap-ref-%d" % n,
                                "arm": ARM, "trial": t, "group": "snapshot", "occupied": n,
                                "wraps_2k": wraps, "n_prompt": len(pFull), "prompt_ids": pFull,
                                "prompt_ids_sha": pFull_sha, "n_predict": int(g["snap_continuation"]),
                                "derivation": "%s|%s" % (deriv.get(key, "?"), deriv.get("g2_suffix", "?"))})
                    try:
                        ensure_empty_slot(base_url, slot_id)
                        cr = run_continuation_stream(base_url, pFull, int(g["snap_continuation"]),
                                                     seed, False, slot_id)
                        rec.update({"ttft_ms": cr["ttft_ms"], "wall_ms": cr["wall_ms"],
                                    "chunks": cr["chunks"], "timings": cr["timings"],
                                    "cache_n": cr["cache_n"], "prompt_n": cr["prompt_n"],
                                    "prompt_ms": cr["prompt_ms"],
                                    "tokens_cached_total": cr["tokens_cached_total"],
                                    "tokens_evaluated": cr["tokens_evaluated"],
                                    "gen_ids": cr["tokens"], "n_gen": len(cr["tokens"]),
                                    "gen_ids_sha": common.sha256_ids(cr["tokens"]) if cr["tokens"] else "empty",
                                    "stop_type": cr["stop_type"], "truncated": cr["truncated"],
                                    "spec": cr["spec"], "acceptance": cr["acceptance"]})
                        if cr["truncated"]:
                            raise RuntimeError("ref truncated")
                        if not cr["tokens"]:
                            raise RuntimeError("ref empty tokens (not parity success)")
                        rec["success"] = True
                        ref_ids, ref_stop = list(cr["tokens"]), cr["stop_type"]
                    except Exception as e:
                        rec["success"] = False
                        rec["error"] = str(e)[:800]
                    rec["vram_sample_peak_mib"] = common.peak_vram_mib(vlog)
                    common.append_jsonl(results_path, rec)
                    if rec.get("success"):
                        done.add(cid_ref)
                    n_new += 1
                    log("G2 %s %s gen=%s" % (cid_ref, "OK" if rec.get("success") else "FAIL", rec.get("n_gen")))
                else:
                    # ref already done: load its gen_ids from jsonl for parity (matching manifest only)
                    try:
                        with open(results_path) as f:
                            for line in f:
                                try:
                                    r = json.loads(line)
                                except Exception:
                                    continue
                                if r.get("cell_id") == cid_ref and r.get("manifest_hash") == mhash and r.get("success"):
                                    ref_ids = r.get("gen_ids", [])
                                    ref_stop = r.get("stop_type")
                                    break
                    except Exception:
                        pass
                # --- restore cell ---
                cid_rs = common.build_cell_id("snap-restore-%d" % n, ARM, t)
                if need(cid_rs):
                    rec = dict(run_facts)
                    rec.update({"job": "G2", "cell_id": cid_rs, "item_id": "snap-restore-%d" % n,
                                "arm": ARM, "trial": t, "group": "snapshot", "occupied": n,
                                "wraps_2k": wraps, "n_prompt": n, "prompt_ids_sha": pN_sha,
                                "boundary": n, "filename": fname,
                                "derivation": deriv.get(key, "?")})
                    try:
                        if not os.path.exists(fpath):
                            raise RuntimeError("snapshot file missing: %s (save cell must succeed first)" % fpath)
                        pre_bytes = os.path.getsize(fpath)
                        rec["file_pre"] = {"path": fpath, "bytes": pre_bytes}
                        ensure_empty_slot(base_url, slot_id)
                        rs = slot_restore(base_url, slot_id, fname)
                        rec["restore"] = {k: v for k, v in rs.items() if not k.startswith("_")}
                        rec["restore_wall_ms"] = rs.get("_wall_ms")
                        if int(rs.get("n_restored", -1)) != n:
                            raise RuntimeError("n_restored %s != %d" % (rs.get("n_restored"), n))
                        n_slot, _ = slot_n_tokens(base_url, slot_id)
                        rec["slot_n_tokens_after"] = n_slot
                        if n_slot is not None and int(n_slot) != n:
                            raise RuntimeError("restored slot n=%s != %d" % (n_slot, n))
                        rec["success"] = True
                    except Exception as e:
                        rec["success"] = False
                        rec["error"] = str(e)[:800]
                    rec["vram_sample_peak_mib"] = common.peak_vram_mib(vlog)
                    common.append_jsonl(results_path, rec)
                    if rec.get("success"):
                        done.add(cid_rs)
                    n_new += 1
                    log("G2 %s %s" % (cid_rs, "OK" if rec.get("success") else "FAIL %s" % rec.get("error", "")[:120]))
                # --- restored continuation cell (no erase; reuse restored KV) ---
                cid_rd = common.build_cell_id("snap-restored-%d" % n, ARM, t)
                if need(cid_rd):
                    rec = dict(run_facts)
                    rec.update({"job": "G2", "cell_id": cid_rd, "item_id": "snap-restored-%d" % n,
                                "arm": ARM, "trial": t, "group": "snapshot", "occupied": n,
                                "wraps_2k": wraps, "n_prompt": len(pFull), "prompt_ids": pFull,
                                "prompt_ids_sha": pFull_sha, "n_predict": int(g["snap_continuation"]),
                                "filename": fname,
                                "derivation": "%s|%s" % (deriv.get(key, "?"), deriv.get("g2_suffix", "?"))})
                    try:
                        # do NOT erase: slot must still hold restored N (verify before generating)
                        n_before, _ = slot_n_tokens(base_url, slot_id)
                        rec["slot_n_before"] = n_before
                        # If a previous restored cell or prefix left state, the slot may not hold N.
                        # Re-restore deterministically (no new timed op; reuse same file) to guarantee the
                        # restored prefix is resident. This keeps parity meaningful under --only-cells repair.
                        if n_before is None or int(n_before) != n:
                            rec["resume_empty_state"] = ensure_empty_slot(base_url, slot_id)
                            rs2 = slot_restore(base_url, slot_id, fname)
                            if int(rs2.get("n_restored", -1)) != n:
                                raise RuntimeError("resume restore boundary mismatch")
                            rec["re_restore"] = {k: v for k, v in rs2.items() if not k.startswith("_")}
                            n_before2, _ = slot_n_tokens(base_url, slot_id)
                            rec["slot_n_before2"] = n_before2
                            if n_before2 is None or int(n_before2) != n:
                                raise RuntimeError("slot before restored gen n=%s != %d" % (n_before2, n))
                        cr = run_continuation_stream(base_url, pFull, int(g["snap_continuation"]),
                                                     seed, True, slot_id)
                        rec.update({"ttft_ms": cr["ttft_ms"], "wall_ms": cr["wall_ms"],
                                    "chunks": cr["chunks"], "timings": cr["timings"],
                                    "cache_n": cr["cache_n"], "prompt_n": cr["prompt_n"],
                                    "prompt_ms": cr["prompt_ms"],
                                    "tokens_cached_total": cr["tokens_cached_total"],
                                    "tokens_evaluated": cr["tokens_evaluated"],
                                    "gen_ids": cr["tokens"], "n_gen": len(cr["tokens"]),
                                    "gen_ids_sha": common.sha256_ids(cr["tokens"]) if cr["tokens"] else "empty",
                                    "stop_type": cr["stop_type"], "truncated": cr["truncated"],
                                    "spec": cr["spec"], "acceptance": cr["acceptance"]})
                        if cr["truncated"]:
                            raise RuntimeError("restored truncated")
                        if not cr["tokens"]:
                            raise RuntimeError("restored empty tokens (not parity success)")
                        # purity: restored prefix must NOT be invisibly re-prefilled
                        try:
                            cn = int(cr["cache_n"]) if cr["cache_n"] is not None else -1
                            pn = int(cr["prompt_n"]) if cr["prompt_n"] is not None else -1
                        except Exception:
                            cn, pn = -1, -1
                        tol = int(g.get("recompute_tol", 16))
                        pure = (cn >= n) and (pn <= int(g["snap_suffix"]) + tol)
                        rec["pure_restore"] = bool(pure)
                        if not pure:
                            rec["recomputation"] = "cache_n=%s prompt_n=%s expected cache>=%d prompt<=%d" % (
                                cn, pn, n, int(g["snap_suffix"]) + tol)
                        # parity vs fresh ref (raw IDs, no retokenization)
                        if ref_ids is None:
                            rec["parity"] = None
                            rec["parity_note"] = "ref IDs unavailable (ref cell not successful this run); compare offline via manifest"
                        else:
                            same = (list(cr["tokens"]) == list(ref_ids))
                            rec["parity"] = bool(same)
                            rec["parity_n"] = min(len(cr["tokens"]), len(ref_ids))
                            rec["ref_sha"] = common.sha256_ids(ref_ids) if ref_ids else "empty"
                            rec["ref_stop"] = ref_stop
                            if not same:
                                fd = None
                                for i, (a, b) in enumerate(zip(cr["tokens"], ref_ids)):
                                    if a != b:
                                        fd = i
                                        break
                                if fd is None and len(cr["tokens"]) != len(ref_ids):
                                    fd = min(len(cr["tokens"]), len(ref_ids))
                                rec["first_diff"] = fd
                        rec["success"] = True
                    except Exception as e:
                        rec["success"] = False
                        rec["error"] = str(e)[:800]
                    rec["vram_sample_peak_mib"] = common.peak_vram_mib(vlog)
                    common.append_jsonl(results_path, rec)
                    if rec.get("success"):
                        done.add(cid_rd)
                    n_new += 1
                    log("G2 %s %s parity=%s pure=%s" % (
                        cid_rd, "OK" if rec.get("success") else "FAIL",
                        rec.get("parity"), rec.get("pure_restore")))
                # Completed trials no longer need their disposable snapshot files.
                if all(cid in done for cid in (cid_save, cid_ref, cid_rs, cid_rd)) and os.path.exists(fpath):
                    os.unlink(fpath)
                    log("G2 removed completed disposable snapshot %s" % fname)
                # erase after each snapshot trial to leave a clean slot for the next
                try:
                    slot_erase(base_url, slot_id)
                except Exception:
                    pass

        # ---- prefix: for each trial: cold (erase) -> hit (same slot) -> miss (same slot) ----
        pP = prompts["prefix"]["ids"]
        pM = prompts["prefix_miss"]["ids"]
        sA = prompts["suffix_a"]["ids"]
        sB = prompts["suffix_b"]["ids"]
        cold_ids = list(pP) + list(sA)
        hit_ids = list(pP) + list(sB)
        miss_ids = list(pM) + list(sB)
        cold_sha = common.sha256_ids(cold_ids)
        hit_sha = common.sha256_ids(hit_ids)
        miss_sha = common.sha256_ids(miss_ids)
        npred = int(g["prefix_n_predict"])
        for t in g["trials"]:
            cold_ran_this_trial = False
            hit_ran_this_trial = False
            # cold
            cid_cold = common.build_cell_id("prefix-cold", ARM, t)
            if need(cid_cold):
                rec = dict(run_facts)
                rec.update({"job": "G2", "cell_id": cid_cold, "item_id": "prefix-cold",
                            "arm": ARM, "trial": t, "group": "prefix",
                            "condition": "cold", "n_prompt": len(cold_ids),
                            "prompt_ids": cold_ids, "prompt_ids_sha": cold_sha,
                            "n_predict": npred,
                            "derivation": "%s|%s" % (deriv.get("prefix", "?"), deriv.get("suffix_a", "?"))})
                try:
                    ensure_empty_slot(base_url, slot_id)
                    cr = run_continuation_stream(base_url, cold_ids, npred, seed, False, slot_id)
                    rec.update({"ttft_ms": cr["ttft_ms"], "wall_ms": cr["wall_ms"],
                                "chunks": cr["chunks"], "timings": cr["timings"],
                                "cache_n": cr["cache_n"], "prompt_n": cr["prompt_n"],
                                "prompt_ms": cr["prompt_ms"],
                                "tokens_cached_total": cr["tokens_cached_total"],
                                "tokens_evaluated": cr["tokens_evaluated"],
                                "gen_ids": cr["tokens"], "n_gen": len(cr["tokens"]),
                                "gen_ids_sha": common.sha256_ids(cr["tokens"]) if cr["tokens"] else "empty",
                                "stop_type": cr["stop_type"], "truncated": cr["truncated"],
                                "spec": cr["spec"], "acceptance": cr["acceptance"]})
                    if cr["truncated"]:
                        raise RuntimeError("cold truncated")
                    if not cr["tokens"]:
                        raise RuntimeError("cold empty tokens")
                    rec["success"] = True
                except Exception as e:
                    rec["success"] = False
                    rec["error"] = str(e)[:800]
                rec["vram_sample_peak_mib"] = common.peak_vram_mib(vlog)
                common.append_jsonl(results_path, rec)
                if rec.get("success"):
                    done.add(cid_cold)
                    cold_ran_this_trial = True
                n_new += 1
                log("G2 %s %s ttft=%s cache=%s" % (cid_cold, "OK" if rec.get("success") else "FAIL",
                                                   rec.get("ttft_ms"), rec.get("cache_n")))
            # hit (same slot, no erase)
            cid_hit = common.build_cell_id("prefix-hit", ARM, t)
            if need(cid_hit):
                rec = dict(run_facts)
                rec.update({"job": "G2", "cell_id": cid_hit, "item_id": "prefix-hit",
                            "arm": ARM, "trial": t, "group": "prefix",
                            "condition": "hit", "n_prompt": len(hit_ids),
                            "prompt_ids": hit_ids, "prompt_ids_sha": hit_sha,
                            "n_predict": npred,
                            "derivation": "%s|%s" % (deriv.get("prefix", "?"), deriv.get("suffix_b", "?"))})
                try:
                    if cid_cold in done and not cold_ran_this_trial:
                        ensure_empty_slot(base_url, slot_id)
                        rec["resume_setup"] = run_prefill_blocking(base_url, cold_ids, seed, slot_id)
                    n_before, _ = slot_n_tokens(base_url, slot_id)
                    rec["slot_n_before"] = n_before
                    cr = run_continuation_stream(base_url, hit_ids, npred, seed, True, slot_id)
                    rec.update({"ttft_ms": cr["ttft_ms"], "wall_ms": cr["wall_ms"],
                                "chunks": cr["chunks"], "timings": cr["timings"],
                                "cache_n": cr["cache_n"], "prompt_n": cr["prompt_n"],
                                "prompt_ms": cr["prompt_ms"],
                                "tokens_cached_total": cr["tokens_cached_total"],
                                "tokens_evaluated": cr["tokens_evaluated"],
                                "gen_ids": cr["tokens"], "n_gen": len(cr["tokens"]),
                                "gen_ids_sha": common.sha256_ids(cr["tokens"]) if cr["tokens"] else "empty",
                                "stop_type": cr["stop_type"], "truncated": cr["truncated"],
                                "spec": cr["spec"], "acceptance": cr["acceptance"]})
                    if cr["truncated"]:
                        raise RuntimeError("hit truncated")
                    if not cr["tokens"]:
                        raise RuntimeError("hit empty tokens")
                    try:
                        cn = int(cr["cache_n"]) if cr["cache_n"] is not None else -1
                    except Exception:
                        cn = -1
                    rec["hit_proved"] = bool(cn >= int(g["prefix_len"]) - int(g.get("recompute_tol", 16)))
                    rec["success"] = True
                except Exception as e:
                    rec["success"] = False
                    rec["error"] = str(e)[:800]
                rec["vram_sample_peak_mib"] = common.peak_vram_mib(vlog)
                common.append_jsonl(results_path, rec)
                if rec.get("success"):
                    done.add(cid_hit)
                    hit_ran_this_trial = True
                n_new += 1
                log("G2 %s %s ttft=%s cache=%s hit=%s" % (cid_hit, "OK" if rec.get("success") else "FAIL",
                                                          rec.get("ttft_ms"), rec.get("cache_n"), rec.get("hit_proved")))
            # miss (same slot, no erase; early head change breaks reuse)
            cid_miss = common.build_cell_id("prefix-miss", ARM, t)
            if need(cid_miss):
                rec = dict(run_facts)
                rec.update({"job": "G2", "cell_id": cid_miss, "item_id": "prefix-miss",
                            "arm": ARM, "trial": t, "group": "prefix",
                            "condition": "miss", "n_prompt": len(miss_ids),
                            "prompt_ids": miss_ids, "prompt_ids_sha": miss_sha,
                            "n_predict": npred,
                            "derivation": "%s|%s" % (deriv.get("prefix_miss", "?"), deriv.get("suffix_b", "?"))})
                try:
                    if cid_hit in done and not hit_ran_this_trial:
                        ensure_empty_slot(base_url, slot_id)
                        rec["resume_setup"] = run_prefill_blocking(base_url, hit_ids, seed, slot_id)
                    n_before, _ = slot_n_tokens(base_url, slot_id)
                    rec["slot_n_before"] = n_before
                    cr = run_continuation_stream(base_url, miss_ids, npred, seed, True, slot_id)
                    rec.update({"ttft_ms": cr["ttft_ms"], "wall_ms": cr["wall_ms"],
                                "chunks": cr["chunks"], "timings": cr["timings"],
                                "cache_n": cr["cache_n"], "prompt_n": cr["prompt_n"],
                                "prompt_ms": cr["prompt_ms"],
                                "tokens_cached_total": cr["tokens_cached_total"],
                                "tokens_evaluated": cr["tokens_evaluated"],
                                "gen_ids": cr["tokens"], "n_gen": len(cr["tokens"]),
                                "gen_ids_sha": common.sha256_ids(cr["tokens"]) if cr["tokens"] else "empty",
                                "stop_type": cr["stop_type"], "truncated": cr["truncated"],
                                "spec": cr["spec"], "acceptance": cr["acceptance"]})
                    if cr["truncated"]:
                        raise RuntimeError("miss truncated")
                    if not cr["tokens"]:
                        raise RuntimeError("miss empty tokens")
                    rec["success"] = True
                except Exception as e:
                    rec["success"] = False
                    rec["error"] = str(e)[:800]
                rec["vram_sample_peak_mib"] = common.peak_vram_mib(vlog)
                common.append_jsonl(results_path, rec)
                if rec.get("success"):
                    done.add(cid_miss)
                n_new += 1
                log("G2 %s %s ttft=%s cache=%s" % (cid_miss, "OK" if rec.get("success") else "FAIL",
                                                   rec.get("ttft_ms"), rec.get("cache_n")))
            # erase after each prefix trial
            try:
                slot_erase(base_url, slot_id)
            except Exception:
                pass

        log("G2 finished: %d cells newly written, %d/%d successful+matching" % (
            n_new, len(common.load_completed(results_path, mhash)), len(manifest["cells"])))
    finally:
        common.stop_vram_sampler(vsamp)
        common.stop_server(srv)
        try:
            peak = common.peak_vram_mib(vlog)
        except Exception:
            peak = None
        log("G2 stop peak_vram=%s" % peak)
        logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
