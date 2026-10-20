#!/usr/bin/env python3
"""G0 readiness gate: strict, no fallback. Run this gate before G1/G2.

Standard CLI: python3 serve/cuda/g0.py --config $GLIMMER_HOME/config.json
Outputs: <results>/g0.jsonl (one durable record, append+fsync) + <results>/g0.json (pretty).
GPU cap 5 min (wrap the run in `timeout`). No download/compile. No model load beyond
--version/--help probes (run with config LD_LIBRARY_PATH when set).
Exit 0 ONLY when gate passes; exit 2 when blocked (record success == gate_pass).
No fallback runtime: the CURRENT release binary in config is required; 69f201a is
never substituted. Put the GGUF hashes in config; G0 reverifies sizes + shas
(target 16GB hash fits the 5-min cap; --skip-target-hash records config sha only).

Gate verifies:
- GPU present AND name contains 4090 (+ power limit recorded)
- runtime binary exists, sha256 == config, --version ok, version mentions expected
  tag/short-rev, FULL --help contains spec-type, metrics, cache-prompt,
  slot-save-path, spec-draft (no truncation)
- all 4 GGUFs exist with exact config bytes + sha256 match
- server contexts >= 33536 and occupied 32768 + 256 output fits
- prompts: eval source has the 5 heldout IDs (nonempty IDs); g1_prompts file has
  7 rows with exact occupied lengths (4096/32768 synthetic)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

HELP_FLAGS = ["slot-save-path", "spec-type", "metrics", "cache-prompt", "spec-draft"]
EXPECTED_SYNTH = {"synth-4096-v1": 4096, "synth-32768-v1": 32768}


def _is_pending(v):
    return (not isinstance(v, str)) or v.strip() == "" or v == "unknown" or v.startswith("PENDING")


def collect_nvidia():
    out = {"available": False}
    r = common.run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,power.limit,driver_version",
                        "--format=csv,noheader"], timeout=30)
    if r["rc"] == 0 and r["out"].strip():
        out["available"] = True
        out["raw"] = r["out"].strip()[:2000]
        try:
            parts = [x.strip() for x in r["out"].strip().splitlines()[0].split(",")]
            out["name"] = parts[0] if len(parts) > 0 else "unknown"
            out["mem_total"] = parts[1] if len(parts) > 1 else "unknown"
            out["power_limit"] = parts[2] if len(parts) > 2 else "unknown"
            out["driver"] = parts[3] if len(parts) > 3 else "unknown"
        except Exception:
            pass
        out["is_4090"] = "4090" in out.get("name", "")
    else:
        out["error"] = (r["err"] or r["out"])[:1000]
        out["is_4090"] = False
    return out


def collect_build(server_bin, cfg):
    """Probe current-release binary with FULL help capture + config LD_LIBRARY_PATH."""
    out = {"server": server_bin, "exists": os.path.exists(server_bin)}
    rt = cfg.get("runtime", {})
    env_extra = None
    ld = rt.get("ld_library_path", "")
    if ld and os.path.isdir(ld):
        env_extra = {"LD_LIBRARY_PATH": ld}
        out["ld_library_path_used"] = ld
    else:
        out["ld_library_path_used"] = None
    if not out["exists"]:
        return out
    try:
        out["bytes"] = os.path.getsize(server_bin)
    except Exception as e:
        out["bytes_error"] = str(e)[:300]
    v = common.run_cmd_full([server_bin, "--version"], timeout=120, env_extra=env_extra)
    out["version_rc"] = v["rc"]
    out["version_out"] = (v["out"] + v["err"])[:3000]
    # expected tag/short-rev mention check
    exp_tag = rt.get("llama_cpp_tag", "") or ""
    exp_rev = rt.get("llama_cpp_rev", "") or ""
    short_rev = exp_rev[:7] if isinstance(exp_rev, str) and len(exp_rev) >= 7 else ""
    blob = out["version_out"]
    out["rev_match"] = bool((short_rev and short_rev in blob) or (exp_tag and exp_tag in blob))
    out["rev_expected"] = {"tag": exp_tag, "short_rev": short_rev}
    h = common.run_cmd_full([server_bin, "--help"], timeout=120, env_extra=env_extra)
    help_text = h["out"] + h["err"]
    out["help_rc"] = h["rc"]
    out["help_len"] = len(help_text)
    out["help_truncated"] = h["truncated"]
    out["flags"] = {f: (f in help_text) for f in HELP_FLAGS}
    out["build_sha256"] = common.sha256_file(server_bin)
    out["build_sha256_expected"] = rt.get("build_sha256", "unknown")
    return out


def collect_models(models_cfg, hash_target):
    out = {}
    for key in ("target", "official", "stock", "tuned"):
        path = models_cfg.get(key, "")
        rec = {"path": path, "exists": os.path.exists(path)}
        exp_bytes = models_cfg.get(key + "_bytes")
        exp_sha = models_cfg.get(key + "_sha256", "unknown")
        rec["bytes_expected"] = exp_bytes
        rec["sha256_expected"] = exp_sha
        if rec["exists"]:
            try:
                rec["bytes"] = os.path.getsize(path)
            except Exception as e:
                rec["bytes_error"] = str(e)[:300]
            rec["bytes_match"] = (rec.get("bytes") == exp_bytes)
            if key == "target" and not hash_target:
                rec["sha256"] = "deferred(--skip-target-hash; config value recorded, NOT reverified)"
                rec["sha_match"] = None
            else:
                rec["sha256"] = common.sha256_file(path)
                rec["sha_match"] = (not _is_pending(exp_sha)) and rec["sha256"] == exp_sha
        out[key] = rec
    return out


def collect_context(cfg):
    sv = cfg.get("server", {})
    c4 = int(sv.get("context_4k", 0))
    c32 = int(sv.get("context_32k", 0))
    return {"context_4k": c4, "context_32k": c32, "min_ok": c4 >= 33536 and c32 >= 33536,
            "room_32k": c32 - 32768, "room_ok": (c32 - 32768) >= 256}


def collect_ports(base):
    out = {"base": base}
    try:
        out["picked"] = common.pick_free_port(base)
    except Exception as e:
        out["picked_error"] = str(e)[:300]
    r = common.run_cmd("ss -ltn 2>&1 | head -n 50", timeout=30)
    out["ss_rc"] = r["rc"]
    out["ss"] = r["out"][:3000]
    return out


def collect_prompts(cfg):
    out = {"heldout_ids": list(cfg["prompts"].get("heldout_ids", []))}
    # eval source: configured path, else the default under GLIMMER_HOME (record which)
    configured = cfg["prompts"].get("eval_source", "")
    default_src = common.DEFAULT_CONFIG["prompts"]["eval_source"]
    src, _ = common.resolve_first_existing(configured, default_src)
    if configured and os.path.exists(configured):
        src = configured
    elif os.path.exists(default_src):
        src = default_src
    else:
        src = configured
    out["eval_source_configured"] = configured
    out["eval_source_used"] = src
    out["eval_exists"] = os.path.exists(src)
    out["heldout_check"] = []
    if out["eval_exists"]:
        try:
            want = set(out["heldout_ids"])
            found = {}
            with open(src) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("id") in want:
                        ids = r.get("prompt_ids", [])
                        found[r["id"]] = {"n_prompt": len(ids),
                                          "prompt_ids_sha": common.sha256_ids(ids) if ids else "empty"}
            for wid in out["heldout_ids"]:
                if wid in found and found[wid]["n_prompt"] > 0:
                    out["heldout_check"].append({"id": wid, "ok": True, **found[wid]})
                else:
                    out["heldout_check"].append({"id": wid, "ok": False,
                                                 **found.get(wid, {})})
        except Exception as e:
            out["heldout_error"] = str(e)[:1000]
    # g1 prompts file: must exist with 7 rows + exact synthetic lengths
    g1p = cfg["prompts"].get("g1_prompts", "")
    out["g1_prompts"] = g1p
    out["g1_exists"] = isinstance(g1p, str) and os.path.exists(g1p)
    out["g1_check"] = []
    if out["g1_exists"]:
        try:
            rows = {}
            with open(g1p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    rows[r.get("id")] = r
            for hid in out["heldout_ids"]:
                r = rows.get(hid)
                out["g1_check"].append({"id": hid, "ok": bool(r and r.get("prompt_ids")),
                                        "n_prompt": len((r or {}).get("prompt_ids", []))})
            for sid, exp_n in EXPECTED_SYNTH.items():
                r = rows.get(sid)
                n = len((r or {}).get("prompt_ids", []))
                out["g1_check"].append({"id": sid, "ok": n == exp_n, "n_prompt": n,
                                        "expected": exp_n})
            out["g1_rows"] = len(rows)
        except Exception as e:
            out["g1_error"] = str(e)[:1000]
    return out


def main():
    ap = argparse.ArgumentParser(description="G0 strict readiness gate (exit 2 when blocked).")
    ap.add_argument("--config", default=os.path.join(common.REMOTE_DIR, "config.json"))
    ap.add_argument("--out", default=None, help="results dir (default from config)")
    ap.add_argument("--skip-target-hash", action="store_true",
                    help="do NOT reverify the 16GB target sha (records config value only)")
    ap.add_argument("--dry-run", action="store_true", help="print gate verdict, write nothing")
    args = ap.parse_args()
    common.install_signal_handlers()
    cfg = common.load_config(args.config)
    if cfg.get("_warnings"):
        print("G0 config warnings: %s" % cfg["_warnings"], flush=True)
    outdir = args.out or cfg.get("results_dir")
    t0 = time.time()
    print("G0 config=%s outdir=%s" % (args.config, outdir), flush=True)
    print("G0 config_hash=%s" % common.config_hash(cfg), flush=True)

    nvidia = collect_nvidia()
    print("G0 gpu=%s is_4090=%s" % (nvidia.get("name", "unavailable"), nvidia.get("is_4090")), flush=True)
    server_bin = cfg["runtime"].get("llama_server", "")
    build = collect_build(server_bin, cfg)
    models = collect_models(cfg.get("models", {}), hash_target=(not args.skip_target_hash))
    ctx = collect_context(cfg)
    disk_r = common.run_cmd(["df", "-h", "/data", "/"], timeout=30)
    disk = {"rc": disk_r["rc"], "out": disk_r["out"][:2000]}
    ports = collect_ports(cfg["server"].get("port_base", 18501))
    prompts = collect_prompts(cfg)

    blocked = []
    if not nvidia.get("available"):
        blocked.append("gpu-unavailable")
    elif not nvidia.get("is_4090"):
        blocked.append("gpu-wrong-model:%s" % nvidia.get("name"))
    if not build.get("exists"):
        blocked.append("runtime-missing:%s" % server_bin)
    else:
        exp_sha = cfg["runtime"].get("build_sha256", "unknown")
        if _is_pending(exp_sha):
            blocked.append("runtime-sha-pending-in-config")
        elif build.get("build_sha256") != exp_sha:
            blocked.append("runtime-sha-mismatch")
        if build.get("version_rc") != 0:
            blocked.append("runtime-version-probe-failed")
        elif not build.get("rev_match"):
            blocked.append("runtime-rev-mismatch:expected %s" % build.get("rev_expected"))
        for f, present in (build.get("flags") or {}).items():
            if not present:
                blocked.append("flag-missing:%s" % f)
        if build.get("help_truncated"):
            blocked.append("help-capture-truncated")
    for k in ("target", "official", "stock", "tuned"):
        m = models[k]
        if not m.get("exists"):
            blocked.append("model-missing:%s" % k)
            continue
        if m.get("bytes_match") is False:
            blocked.append("model-bytes-mismatch:%s" % k)
        if _is_pending(m.get("sha256_expected")):
            blocked.append("model-sha-pending:%s" % k)
        elif m.get("sha_match") is False:
            blocked.append("model-sha-mismatch:%s" % k)
        elif m.get("sha_match") is None and k == "target":
            pass  # deferred by flag; recorded, not verified
    if not ctx["min_ok"]:
        blocked.append("context-room:4k=%s,32k=%s need >=33536" % (ctx["context_4k"], ctx["context_32k"]))
    if not ctx["room_ok"]:
        blocked.append("context-output-room:32k=%s leaves %s, need >=256" % (ctx["context_32k"], ctx["room_32k"]))
    for h in prompts.get("heldout_check", []):
        if not h.get("ok"):
            blocked.append("heldout-missing:%s" % h.get("id"))
    if not prompts.get("g1_exists"):
        blocked.append("prompts-g1-missing:%s" % prompts.get("g1_prompts"))
    else:
        for c in prompts.get("g1_check", []):
            if not c.get("ok"):
                blocked.append("prompts-g1-bad:%s n=%s" % (c.get("id"), c.get("n_prompt")))

    gate_pass = len(blocked) == 0
    record = {
        "job": "G0",
        "cell_id": "G0|readiness|t0",
        "manifest_hash": common.config_hash(cfg),
        "success": gate_pass,
        "blocked": blocked,
        "gate_pass": gate_pass,
        "no_fallback": True,
        "elapsed_s": time.time() - t0,
        "time_unix": int(time.time()),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "nvidia": nvidia,
        "build": build,
        "models": models,
        "context": ctx,
        "disk": disk,
        "ports": ports,
        "prompts": prompts,
        "config": cfg,
    }
    if args.dry_run:
        print(json.dumps({"job": "G0", "blocked": blocked, "gate_pass": gate_pass}, indent=1))
        return 0 if gate_pass else 2
    os.makedirs(outdir, exist_ok=True)
    common.append_jsonl(os.path.join(outdir, "g0.jsonl"), record)
    with open(os.path.join(outdir, "g0.json"), "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    print("G0 gate_pass=%s blocked=%s" % (gate_pass, blocked), flush=True)
    print("G0 wrote %s/g0.jsonl + g0.json" % outdir, flush=True)
    return 0 if gate_pass else 2


if __name__ == "__main__":
    sys.exit(main())
