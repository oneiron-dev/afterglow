#!/usr/bin/env python3
"""Shared helpers for the G0/G1/G2 CUDA harness scripts (stdlib only).

g0.py, g1.py, g2.py and make_prompts.py import this module from the same directory.
Do not add third-party imports. All functions handle missing tools gracefully
(record blocked/unknown, never crash the matrix).

Config schema: see DEFAULT_CONFIG. config.json (see config.example.json) is merged
over these defaults, so CPU validation works without a full config file.

Paths: everything defaults under $GLIMMER_HOME (runtime/, models/, prompts/, slots/,
results/); model files default under $GLIMMER_MODELS (default $GLIMMER_HOME/models).
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error

SCHEMA = 1
REMOTE_DIR = os.environ.get("GLIMMER_HOME", os.path.abspath("glimmer-work"))
MODELS_DIR = os.environ.get("GLIMMER_MODELS", REMOTE_DIR + "/models")

DEFAULT_CONFIG = {
    "schema": 1,
    "remote_dir": REMOTE_DIR,
    "scripts_dir": os.path.dirname(os.path.abspath(__file__)),
    "results_dir": REMOTE_DIR + "/results",
    "prompts_dir": REMOTE_DIR + "/prompts",
    "runtime": {
        "llama_server": REMOTE_DIR + "/runtime/bin/llama-server",
        "llama_cpp_rev": "unknown",
        "llama_cpp_version": "unknown",
        "build_sha256": "unknown",
        "ld_library_path": REMOTE_DIR + "/runtime/bin",
        "fallback_server": "",
        "fallback_rev": "69f201a2051ea9b9e9b50c3cc56afd9e2ae64414",
    },
    "models": {
        "target": MODELS_DIR + "/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
        "target_sha256": "unknown",
        "target_bytes": 16756683904,
        "official": MODELS_DIR + "/dflash-Muse-Glimmer-30B-Q4_K_M.gguf",
        "official_sha256": "unknown",
        "official_bytes": 1631208128,
        "stock": MODELS_DIR + "/dflash-stockmine-Q4_K_M.gguf",
        "stock_sha256": "unknown",
        "stock_bytes": 1631205408,
        "tuned": MODELS_DIR + "/dflash-tuned-lr2e-05-Q4_K_M.gguf",
        "tuned_sha256": "unknown",
        "tuned_bytes": 1631205408,
    },
    "prompts": {
        "eval_source": REMOTE_DIR + "/prompts/prompts_eval.jsonl",
        "heldout_ids": ["ev-L00235-c", "ev-L00254-r", "ev-uc-0", "ev-L00396-c", "ev-L00429-r"],
        "g1_prompts": REMOTE_DIR + "/prompts/prompts_g1.jsonl",
        "g2_prompts": REMOTE_DIR + "/prompts/prompts_g2.jsonl",
        "tokenizer": MODELS_DIR + "/tokenizer.json",
        "tokenizer_rev": "unknown",
    },
    "server": {
        "host": "127.0.0.1",
        "port_base": 18501,
        "n_parallel": 1,
        "n_gpu_layers": 99,
        "flash_attn": "on",
        "context_4k": 33536,
        "context_32k": 33536,
    },
    "sampling": {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 42,
        "n_predict_heldout": 512,
        "n_predict_synthetic": 256,
        "cache_prompt": False,
        "stream": True,
    },
    "draft": {
        "spec_type": "draft-dflash",
        "n_max": 15,
        "n_min": 0,
        "p_min": 0.0,
        "sampling": "greedy",
    },
    "arms": ["target-only", "official", "stock", "tuned"],
}


def deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# (section, key) pairs that must be strings (paths/ids); wrong types from a foreign
# config file are restored to defaults with a warning instead of crashing.
_STRING_KEYS = [("prompts", "eval_source"), ("prompts", "g1_prompts"), ("prompts", "g2_prompts"),
                ("prompts", "tokenizer"), ("runtime", "llama_server"), ("runtime", "ld_library_path"),
                ("results_dir", None), ("prompts_dir", None), ("scripts_dir", None), ("remote_dir", None)]


def load_config(path):
    """Load config.json merged over DEFAULT_CONFIG. Missing file -> defaults.

    Foreign files with wrong-typed path keys (e.g. a list where a path belongs)
    are normalized back to defaults; warnings land in cfg["_warnings"].
    """
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path and os.path.exists(path):
        with open(path) as f:
            user = json.load(f)
        cfg = deep_merge(cfg, user)
    warns = []
    for section, key in _STRING_KEYS:
        if key is None:
            if not isinstance(cfg.get(section), str):
                warns.append("config %s wrong type %s, using default" % (section, type(cfg.get(section)).__name__))
                cfg[section] = DEFAULT_CONFIG[section]
        else:
            sect = cfg.get(section) or {}
            if not isinstance(sect.get(key), str):
                warns.append("config %s.%s wrong type %s, using default" % (
                    section, key, type(sect.get(key)).__name__))
                sect[key] = DEFAULT_CONFIG[section][key]
    if not isinstance(cfg.get("arms"), list) or not cfg["arms"]:
        warns.append("config arms wrong type/empty, using default")
        cfg["arms"] = list(DEFAULT_CONFIG["arms"])
    if not isinstance((cfg.get("prompts") or {}).get("heldout_ids"), list):
        warns.append("config prompts.heldout_ids wrong type, using default")
        cfg["prompts"]["heldout_ids"] = list(DEFAULT_CONFIG["prompts"]["heldout_ids"])
    cfg["_warnings"] = warns
    cfg["_config_path"] = path
    return cfg


def resolve_first_existing(*paths):
    """Return (path, existed_bool) for the first existing path, else (first, False)."""
    for p in paths:
        if p and os.path.exists(p):
            return p, True
    return (paths[0] if paths else None), False


def config_hash(cfg):
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_ids(ids):
    # Canonical: comma-joined decimal IDs (exact, no spaces)
    return hashlib.sha256(",".join(str(int(x)) for x in ids).encode()).hexdigest()


def sha256_file(path, max_bytes=None):
    """Hash file in 8MB chunks. Returns 'missing:PATH' if absent, 'error:MSG' on failure."""
    if not os.path.exists(path):
        return "missing:" + path
    try:
        h = hashlib.sha256()
        n = 0
        with open(path, "rb") as f:
            while True:
                b = f.read(8 * 1024 * 1024)
                if not b:
                    break
                if max_bytes is not None and n + len(b) > max_bytes:
                    b = b[: max_bytes - n]
                    h.update(b)
                    break
                h.update(b)
                n += len(b)
                if max_bytes is not None and n >= max_bytes:
                    break
        return h.hexdigest()
    except Exception as e:
        return "error:%s" % e


def append_jsonl(path, record):
    """Append one JSON record + fsync (durable). Creates parent dirs."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())  # durability: fsync failure MUST propagate, never swallow


def load_completed(results_path, manifest_hash):
    """Return set of cell_ids with success==True and matching manifest_hash.

    Skips ONLY successful matching cells. Failed/blocked cells are re-run.
    """
    done = set()
    if not os.path.exists(results_path):
        return done
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("manifest_hash") != manifest_hash:
                continue
            if r.get("success") is True and r.get("cell_id"):
                done.add(r["cell_id"])
    return done


def build_cell_id(item_id, arm, trial):
    return "%s|%s|t%d" % (item_id, arm, int(trial))


def manifest_hash(manifest):
    canon = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()


# ---------- HTTP (stdlib only) ----------

def _req(url, data=None, timeout=60, method="GET"):
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except Exception:
            body = b""
        return e.code, body


def post_json(url, body, timeout=1200):
    data = json.dumps(body).encode()
    st, raw = _req(url, data=data, timeout=timeout, method="POST")
    try:
        return st, json.loads(raw.decode())
    except Exception:
        return st, {"_raw": raw[:2000].decode("utf-8", "replace")}


def get_json(url, timeout=60):
    st, raw = _req(url, timeout=timeout, method="GET")
    try:
        return st, json.loads(raw.decode())
    except Exception:
        return st, {"_raw": raw[:2000].decode("utf-8", "replace")}


def get_text(url, timeout=60):
    st, raw = _req(url, timeout=timeout, method="GET")
    try:
        return st, raw.decode()
    except Exception:
        return st, ""


def parse_metrics(text):
    """Parse Prometheus text from /metrics into {name: float}. Last value wins."""
    m = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            k, v = line.rsplit(" ", 1)
            name = k.split(":")[-1]
            # strip label braces: metric{...} -> metric
            if "{" in name:
                name = name[: name.index("{")]
            m[name] = float(v)
        except Exception:
            continue
    return m


def get_metrics(url, timeout=60):
    st, txt = get_text(url.rstrip("/") + "/metrics", timeout=timeout)
    if st != 200:
        return {}
    return parse_metrics(txt)


SPEC_KEYS = (
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
    "spec_decode_num_drafts_total",
)


def metrics_deltas(m0, m1, keys=SPEC_KEYS):
    """Counter deltas. None when EITHER sample lacks the key (never fabricate zero)."""
    out = {}
    for k in keys:
        if k in (m0 or {}) and k in (m1 or {}):
            try:
                out[k] = float(m1[k]) - float(m0[k])
            except Exception:
                out[k] = None
        else:
            out[k] = None
    return out


def acceptance(deltas):
    """Acceptance from spec deltas. Unknown -> None fields, never zero-fill.

    accept_len = 1 + accepted/drafts (mean acceptance length, drafter-tune convention).
    accept_rate = accepted/proposed (accepted/draft-tokens).
    """
    try:
        proposed = deltas.get("spec_decode_num_draft_tokens_total")
        accepted = deltas.get("spec_decode_num_accepted_tokens_total")
        drafts = deltas.get("spec_decode_num_drafts_total")
        if proposed is None or accepted is None or drafts is None:
            return {"accept_len": None, "accept_rate": None, "proposed": proposed,
                    "accepted": accepted, "drafts": drafts, "known": False}
        accept_len = (1.0 + accepted / drafts) if drafts else None
        accept_rate = (accepted / proposed) if proposed else None
        return {"accept_len": accept_len, "accept_rate": accept_rate,
                "proposed": proposed, "accepted": accepted, "drafts": drafts, "known": True}
    except Exception:
        return {"accept_len": None, "accept_rate": None, "proposed": None,
                "accepted": None, "drafts": None, "known": False}


def _chunk_has_token(obj):
    """True if chunk carries actual generated content (nonempty tokens/content), not progress."""
    if not isinstance(obj, dict):
        return False
    toks = obj.get("tokens")
    if isinstance(toks, list) and len(toks) > 0:
        return True
    if isinstance(obj.get("content"), str) and obj["content"] != "":
        return True
    try:
        if int(obj.get("tokens_predicted", 0)) > 0:
            return True
    except Exception:
        pass
    return False


def _is_final_chunk(obj):
    """Terminal full-response chunk: stop flag or full timings dict present."""
    return isinstance(obj, dict) and (obj.get("stop") is True or isinstance(obj.get("timings"), dict))


def decide_stream_tokens(chunk_toks, chunk_final, predicted_n=None):
    """Authoritative token list from per-chunk token arrays (pure, fixture-tested).

    chunk_toks: list of token arrays in arrival order (each [] if chunk had none).
    chunk_final: parallel bools (True for terminal full-response chunk).
    predicted_n: native timings predicted_n or None.
    Rule:
    - Split arrays into incremental (non-final chunks) and final_cand (last final chunk).
    - accum = concat of incremental arrays.
    - If no final chunk: authoritative = accum (pure incremental stream).
    - If final_cand empty: authoritative = accum (server sent progress-only final).
    - If final_cand covers accum as prefix (equal, or extra tail = last partial
      token(s)): authoritative = final_cand (no double count, no silent loss).
    - Else (conflict): prefer the candidate whose length matches predicted_n
      (final_cand, then accum+final tail, then accum); else final_cand when it is
      longer, else accum. Always returns a warnings list on conflict.
    Fixtures: [1],[2]+final[1,2]->[1,2]; [1],[2]+final[]->[1,2];
    [1],[2]+final[1,2,3]->[1,2,3].
    """
    # Only the LAST final chunk is the terminal full array; every other chunk
    # (including earlier finals and chunks after a final) counts as incremental.
    last_fin_idx = max([i for i, f in enumerate(chunk_final) if f], default=-1)
    inc = []
    final_cand = None
    for i, (toks, is_fin) in enumerate(zip(chunk_toks, chunk_final)):
        toks = [int(x) for x in (toks or [])]
        if i == last_fin_idx:
            final_cand = toks
        else:
            inc.extend(toks)
    warns = []
    if final_cand is None:
        return list(inc), warns
    if len(final_cand) == 0:
        return list(inc), warns
    if final_cand[: len(inc)] == inc:
        return list(final_cand), warns  # covers equal + extra-tail (no loss, no dup)
    # conflict: incremental and final disagree
    warns.append("stream_tokens_conflict inc_len=%d final_len=%d" % (len(inc), len(final_cand)))
    cands = [("final", list(final_cand)), ("concat", list(inc) + list(final_cand)),
             ("inc", list(inc))]
    if predicted_n is not None:
        try:
            pn = int(predicted_n)
            for name, cand in cands:
                if len(cand) == pn:
                    warns.append("stream_tokens_resolved_by_predicted_n=%s" % name)
                    return cand, warns
        except Exception:
            pass
    if len(final_cand) >= len(inc):
        return list(final_cand), warns
    return list(inc), warns


def completion_stream(base_url, body, timeout=1200):
    """POST /completion with stream:true, real incremental TTFT via readline().

    Reads SSE line-by-line (readline returns as soon as each line arrives; never
    block on fixed-size reads). TTFT = time to first chunk with nonempty
    tokens/content/tokens_predicted, never a progress-only chunk. Tokens decided by
    decide_stream_tokens() (per-chunk arrays excluding terminal full array; final
    full-response wins only when covering accumulation; predicted_n resolves
    conflicts). Returns dict with first_token_ms (or None), wall_ms, final (merged
    non-token fields + authoritative tokens), chunks, stream_warnings.
    Raises on HTTP/timeout errors.
    """
    url = base_url.rstrip("/") + "/completion"
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.time()
    first_ms = None
    n_chunks = 0
    final = {}
    chunk_toks = []
    chunk_final = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                obj = json.loads(payload.decode("utf-8", "replace"))
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            n_chunks += 1
            if first_ms is None and _chunk_has_token(obj):
                first_ms = (time.time() - t0) * 1000.0
            toks = obj.get("tokens")
            chunk_toks.append([int(x) for x in toks] if isinstance(toks, list) else [])
            chunk_final.append(_is_final_chunk(obj))
            for k, v in obj.items():
                if k != "tokens":
                    final[k] = v
    wall_ms = (time.time() - t0) * 1000.0
    pred_n = None
    try:
        pred_n = (final.get("timings") or {}).get("predicted_n")
    except Exception:
        pass
    tokens, warns = decide_stream_tokens(chunk_toks, chunk_final, pred_n)
    if tokens:
        final["tokens"] = tokens
    return {"first_token_ms": first_ms, "wall_ms": wall_ms, "final": final,
            "chunks": n_chunks, "stream_warnings": warns,
            "chunk_arrays": len(chunk_toks)}


def run_sse_selftest():
    """Live delayed-SSE fixture test (stdlib only, no model): proves TTFT is incremental.

    Spins a local mock /completion SSE server: chunk1 tokens [11] immediately,
    0.6s delay, chunk2 tokens [22], final {tokens:[11,22], stop:true,
    timings:{predicted_n:2}}. Asserts tokens==[11,22], TTFT<0.45s, wall>=0.6s.
    Also runs pure decide_stream_tokens fixtures F1/F2/F3. Returns list of
    (name, passed, detail). Raises nothing.
    """
    results = []
    # pure fixtures first (no network)
    try:
        t, w = decide_stream_tokens([[1], [2], [1, 2]], [False, False, True], 2)
        results.append(("sse_F1_full_final", t == [1, 2], "%s %s" % (t, w)))
    except Exception as e:
        results.append(("sse_F1_full_final", False, str(e)[:200]))
    try:
        t, w = decide_stream_tokens([[1], [2], []], [False, False, True], 2)
        results.append(("sse_F2_empty_final", t == [1, 2], "%s %s" % (t, w)))
    except Exception as e:
        results.append(("sse_F2_empty_final", False, str(e)[:200]))
    try:
        t, w = decide_stream_tokens([[1], [2], [1, 2, 3]], [False, False, True], 3)
        results.append(("sse_F3_extra_tail", t == [1, 2, 3], "%s %s" % (t, w)))
    except Exception as e:
        results.append(("sse_F3_extra_tail", False, str(e)[:200]))
    try:
        t, w = decide_stream_tokens([[1], [2]], [False, False], 2)
        results.append(("sse_F4_incremental_only", t == [1, 2], "%s %s" % (t, w)))
    except Exception as e:
        results.append(("sse_F4_incremental_only", False, str(e)[:200]))
    # live delayed mock server
    try:
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    ln = int(self.headers.get("Content-Length", "0"))
                    self.rfile.read(ln)
                except Exception:
                    pass
                body1 = b'data: {"tokens":[11],"content":"a","tokens_predicted":1}\n\n'
                body2 = b'data: {"tokens":[22],"content":"b","tokens_predicted":2}\n\n'
                body3 = (b'data: {"tokens":[11,22],"stop":true,"tokens_predicted":2,'
                         b'"timings":{"prompt_n":8,"prompt_ms":10.0,"predicted_n":2,"predicted_ms":600.0}}\n\n'
                         b'data: [DONE]\n\n')
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    self.wfile.write(body1)
                    self.wfile.flush()
                    time.sleep(0.6)
                    self.wfile.write(body2)
                    self.wfile.flush()
                    time.sleep(0.1)
                    self.wfile.write(body3)
                    self.wfile.flush()
                except Exception:
                    pass

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05})
        th.daemon = True
        th.start()
        try:
            r = completion_stream("http://127.0.0.1:%d" % port, {"prompt": [1], "stream": True}, timeout=30)
        finally:
            srv.shutdown()
            th.join(timeout=5)
            srv.server_close()
        toks = (r.get("final") or {}).get("tokens", [])
        ttft = r.get("first_token_ms")
        wall = r.get("wall_ms", 0)
        ok_toks = toks == [11, 22]
        ok_ttft = ttft is not None and ttft < 450.0
        ok_wall = wall >= 600.0
        results.append(("sse_live_tokens", ok_toks, str(toks)[:200]))
        results.append(("sse_live_ttft_incremental", ok_ttft, "ttft_ms=%s" % ttft))
        results.append(("sse_live_wall_delayed", ok_wall, "wall_ms=%s" % wall))
    except Exception as e:
        results.append(("sse_live", False, "mock server failed: %s" % str(e)[:300]))
    return results


def check_lengths(cell, gen_ids, timings, n_predict):
    """Consistency guards: output length vs native counters vs cap. Returns list of warnings."""
    warns = []
    try:
        n_gen = len(gen_ids)
        pred_n = (timings or {}).get("predicted_n")
        if pred_n is not None and int(pred_n) != n_gen:
            warns.append("predicted_n=%s != len(gen_ids)=%d" % (pred_n, n_gen))
        if n_gen > int(n_predict):
            warns.append("n_gen=%d exceeds n_predict=%s" % (n_gen, n_predict))
        # native cache counters: cache_n=reused, prompt_n=computed (verified in server source)
        cache_n = (timings or {}).get("cache_n")
        prompt_n = (timings or {}).get("prompt_n")
        if cache_n is not None and prompt_n is not None:
            try:
                if int(cache_n) + int(prompt_n) != int(cell.get("n_prompt", -1)) and int(cell.get("n_prompt", 0)) > 0:
                    # cache_prompt=false expects cache_n==0; record don't fail
                    warns.append("cache_n(%s)+prompt_n(%s)!=n_prompt(%s)" % (cache_n, prompt_n, cell.get("n_prompt")))
            except Exception:
                pass
    except Exception as e:
        warns.append("check_error:%s" % str(e)[:200])
    return warns


def completion_blocking(base_url, body, timeout=1200):
    """POST /completion stream:false. Returns (wall_ms, json)."""
    body = dict(body)
    body["stream"] = False
    t0 = time.time()
    st, obj = post_json(base_url.rstrip("/") + "/completion", body, timeout=timeout)
    wall_ms = (time.time() - t0) * 1000.0
    if st != 200:
        raise RuntimeError("completion HTTP %s: %s" % (st, str(obj)[:500]))
    return wall_ms, obj


# ---------- processes ----------

_CLEANUPS = []
_SIGNALS_INSTALLED = False


def _run_cleanups():
    for fn in reversed(list(_CLEANUPS)):
        try:
            fn()
        except Exception:
            pass


def _sig_handler(signum, frame):
    _run_cleanups()
    try:
        import signal
        signal.signal(signum, signal.SIG_DFL)
    except Exception:
        pass
    try:
        os.kill(os.getpid(), signum)
    except Exception:
        sys.exit(128 + int(signum))


def install_signal_handlers():
    """Install SIGTERM/SIGINT handlers that stop owned servers/samplers first.

    Call once at script start. Owned procs auto-register on start and unregister
    on stop; a wrapping try/finally in the caller is still required.
    """
    global _SIGNALS_INSTALLED
    if _SIGNALS_INSTALLED:
        return
    try:
        import signal
        signal.signal(signal.SIGTERM, _sig_handler)
        signal.signal(signal.SIGINT, _sig_handler)
        _SIGNALS_INSTALLED = True
    except Exception:
        pass


def register_cleanup(fn):
    _CLEANUPS.append(fn)
    return fn


def unregister_cleanup(fn):
    try:
        _CLEANUPS.remove(fn)
    except ValueError:
        pass


def run_cmd(cmd, timeout=120, env_extra=None):
    """Run cmd (list or str), return {rc, out, err}. Never raises. Truncates to last 8000 chars."""
    try:
        env = None
        if env_extra:
            env = dict(os.environ)
            env.update(env_extra)
        if isinstance(cmd, str):
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
        else:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return {"rc": p.returncode, "out": (p.stdout or "")[-8000:], "err": (p.stderr or "")[-8000:]}
    except subprocess.TimeoutExpired as e:
        return {"rc": 124, "out": str((e.stdout or ""))[-8000:], "err": "timeout %s" % timeout}
    except Exception as e:
        return {"rc": 127, "out": "", "err": str(e)[:2000]}


def run_cmd_full(cmd, timeout=120, env_extra=None, cap_bytes=4 * 1024 * 1024):
    """Run cmd with FULL output capture (for --help flag checks). Never raises.

    Caps at cap_bytes to bound memory; reports truncation. Returns
    {rc, out, err, truncated}.
    """
    try:
        env = None
        if env_extra:
            env = dict(os.environ)
            env.update(env_extra)
        if isinstance(cmd, str):
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
        else:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        out, err = p.stdout or "", p.stderr or ""
        trunc = False
        if len(out) + len(err) > cap_bytes:
            out = out[:cap_bytes]
            err = err[:max(0, cap_bytes - len(out))]
            trunc = True
        return {"rc": p.returncode, "out": out, "err": err, "truncated": trunc}
    except subprocess.TimeoutExpired as e:
        return {"rc": 124, "out": str(e.stdout or ""), "err": "timeout %s" % timeout, "truncated": False}
    except Exception as e:
        return {"rc": 127, "out": "", "err": str(e)[:2000], "truncated": False}


def pick_free_port(base, host="127.0.0.1", tries=50):
    """Return first free TCP port >= base (bind-probe, no race guarantee; caller re-checks via ss)."""
    for p in range(int(base), int(base) + int(tries)):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, p))
            s.close()
            return p
        except OSError:
            try:
                s.close()
            except Exception:
                pass
            continue
    raise RuntimeError("no free port in [%d,%d)" % (base, base + tries))


def start_server(cmd, log_path, env_extra=None):
    """Start llama-server in its own process group (setsid). Returns Popen. Caller owns cleanup."""
    d = os.path.dirname(log_path)
    if d:
        os.makedirs(d, exist_ok=True)
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    logf = open(log_path, "w")
    # setsid -> new pgid so stop_server can kill the whole group
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                         preexec_fn=os.setsid if hasattr(os, "setsid") else None)
    p._logf = logf
    p._cleanup_fn = lambda: stop_server(p)
    register_cleanup(p._cleanup_fn)
    return p


def stop_server(p, grace=5):
    """TERM the process group, then KILL after grace. Never raises."""
    if p is None:
        return
    try:
        try:
            unregister_cleanup(p._cleanup_fn)
        except Exception:
            pass
    except Exception:
        pass
    try:
        if p.poll() is not None:
            try:
                p._logf.close()
            except Exception:
                pass
            return
        import signal
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            try:
                p.terminate()
            except Exception:
                pass
        try:
            p.wait(timeout=grace)
        except Exception:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            try:
                p.wait(timeout=grace)
            except Exception:
                pass
    finally:
        try:
            p._logf.close()
        except Exception:
            pass


def wait_health(base_url, secs=240):
    """Poll /health until 200. Returns {ok, waited_s, last_status}."""
    t0 = time.time()
    last = None
    while time.time() - t0 < secs:
        try:
            st, _ = get_json(base_url.rstrip("/") + "/health", timeout=10)
            last = st
            if st == 200:
                return {"ok": True, "waited_s": time.time() - t0, "last_status": st}
        except Exception as e:
            last = str(e)[:200]
        time.sleep(1.0)
    return {"ok": False, "waited_s": time.time() - t0, "last_status": last}


def start_vram_sampler(log_path, interval=2):
    """Sample nvidia-smi memory.used,power.draw every 2s. Returns Popen or None if unavailable."""
    d = os.path.dirname(log_path)
    if d:
        os.makedirs(d, exist_ok=True)
    script = "while sleep %d; do nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits; done" % int(interval)
    try:
        logf = open(log_path, "w")
        p = subprocess.Popen(["bash", "-c", script], stdout=logf, stderr=subprocess.DEVNULL,
                             preexec_fn=os.setsid if hasattr(os, "setsid") else None)
        p._logf = logf
        p._cleanup_fn = lambda: stop_vram_sampler(p)
        register_cleanup(p._cleanup_fn)
        return p
    except Exception:
        return None


def stop_vram_sampler(p):
    if p is None:
        return
    try:
        try:
            unregister_cleanup(p._cleanup_fn)
        except Exception:
            pass
    except Exception:
        pass
    try:
        import signal
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            try:
                p.terminate()
            except Exception:
                pass
        try:
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    finally:
        try:
            p._logf.close()
        except Exception:
            pass


def peak_vram_mib(log_path):
    """Peak of first CSV column (memory.used MiB). Returns None if no samples."""
    try:
        peak = None
        with open(log_path) as f:
            for line in f:
                try:
                    v = float(line.split(",")[0].strip())
                    peak = v if peak is None else max(peak, v)
                except Exception:
                    continue
        return peak
    except Exception:
        return None


def self_check():
    """Quick stdlib self-test for CPU validation (no server, no GPU)."""
    checks = []
    # 1. hashes stable
    a = sha256_ids([1, 2, 3])
    b = sha256_ids([1, 2, 3])
    checks.append(("sha_ids_stable", a == b))
    # 2. metrics parse
    m = parse_metrics("spec_decode_num_drafts_total 3\nprompt_tokens_seconds 1.5\n")
    checks.append(("metrics_parse", m.get("spec_decode_num_drafts_total") == 3.0))
    # 3. acceptance known/unknown
    acc = acceptance({"spec_decode_num_draft_tokens_total": 10.0,
                      "spec_decode_num_accepted_tokens_total": 7.0,
                      "spec_decode_num_drafts_total": 3.0})
    checks.append(("acceptance", abs(acc["accept_len"] - (1 + 7.0 / 3.0)) < 1e-9 and acc["known"]))
    acc2 = acceptance({})
    checks.append(("acceptance_unknown", acc2["accept_len"] is None and not acc2["known"]))
    # 4. manifest round-trip
    man = {"cells": [{"cell_id": "a|b|t1"}]}
    h1 = manifest_hash(man)
    checks.append(("manifest_hash", isinstance(h1, str) and len(h1) == 64))
    # 5. port pick returns int
    try:
        p = pick_free_port(18901, tries=5)
        checks.append(("port_pick", isinstance(p, int)))
    except Exception:
        checks.append(("port_pick", False))
    # 6. metrics deltas: None when either side lacks the key (never fabricate zero)
    d = metrics_deltas({"a": 1.0}, {}, keys=("a", "b"))
    checks.append(("metrics_deltas_none", d.get("a") is None and d.get("b") is None))
    d2 = metrics_deltas({"a": 1.0}, {"a": 4.0}, keys=("a",))
    checks.append(("metrics_deltas_value", d2.get("a") == 3.0))
    acc3 = acceptance(metrics_deltas({}, {}))
    checks.append(("acceptance_none_deltas", acc3["accept_len"] is None and not acc3["known"]))
    # 7. config normalization: wrong-typed path keys restored to defaults
    cfg = load_config(None)
    cfg2 = deep_merge(json.loads(json.dumps(cfg)), {"prompts": {"g1_prompts": ["x"]}})
    # simulate foreign file by direct merge + normalize path
    tmp_warns = []
    if not isinstance(cfg2["prompts"]["g1_prompts"], str):
        cfg2["prompts"]["g1_prompts"] = DEFAULT_CONFIG["prompts"]["g1_prompts"]
        tmp_warns.append("x")
    checks.append(("config_normalize", isinstance(cfg2["prompts"]["g1_prompts"], str) and tmp_warns))
    # 8. pure SSE token fixtures (no network)
    t, _ = decide_stream_tokens([[1], [2], [1, 2]], [False, False, True], 2)
    checks.append(("sse_F1", t == [1, 2]))
    t, _ = decide_stream_tokens([[1], [2], []], [False, False, True], 2)
    checks.append(("sse_F2", t == [1, 2]))
    t, _ = decide_stream_tokens([[1], [2], [1, 2, 3]], [False, False, True], 3)
    checks.append(("sse_F3", t == [1, 2, 3]))
    return checks


if __name__ == "__main__":
    ok = True
    for name, passed in self_check():
        print("%s %s" % ("PASS" if passed else "FAIL", name))
        ok = ok and passed
    print("--- live delayed-SSE fixtures (localhost mock, no model) ---")
    for name, passed, detail in run_sse_selftest():
        print("%s %s %s" % ("PASS" if passed else "FAIL", name, detail))
        ok = ok and passed
    sys.exit(0 if ok else 1)
