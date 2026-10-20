#!/usr/bin/env python3
"""One eval arm against a running llama-server: concurrency 1, greedy, the held-out prompts (stdlib only).
Per prompt it records decode timings, the generated ids and the speculative counters from /metrics.
usage: eval_client.py PORT ARM PROMPTS_EVAL.jsonl WARMUP_PROMPTS.jsonl OUT.jsonl N_PREDICT"""
import json, sys, time, urllib.request
port, arm, src, warm_src, out, n_predict = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
url = f"http://127.0.0.1:{port}"
def post(path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1200))
def metrics():
    m = {}
    for line in urllib.request.urlopen(url + "/metrics", timeout=60).read().decode().splitlines():
        if line and not line.startswith("#"):
            k, v = line.rsplit(" ", 1)
            m[k.split(":")[-1]] = float(v)
    return m
def gen(ids, n):
    return post("/completion", {"prompt": ids, "n_predict": n, "temperature": 0.0, "top_k": 1, "cache_prompt": False,
                                "return_tokens": True, "stream": False})
warm = [json.loads(l) for l in open(warm_src)][:2]
for w in warm:                       # warm-up, not counted
    gen(w["prompt_ids"], 128)
prompts = [json.loads(l) for l in open(src)]
with open(out, "w") as f:
    for p in prompts:
        m0 = metrics(); t0 = time.time()
        r = gen(p["prompt_ids"], n_predict)
        wall = time.time() - t0; m1 = metrics()
        d = {k: m1.get(k, 0) - m0.get(k, 0) for k in ("spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total",
                                                       "spec_decode_num_drafts_total")}
        rec = {"arm": arm, "id": p["id"], "src": p["src"], "lang": p["lang"], "gen_ids": r["tokens"], "n_gen": len(r["tokens"]),
               "timings": r["timings"], "wall_s": wall, "spec": d}
        f.write(json.dumps(rec) + "\n"); f.flush()
        t = r["timings"]
        acc = (1 + d["spec_decode_num_accepted_tokens_total"] / d["spec_decode_num_drafts_total"]) if d["spec_decode_num_drafts_total"] else None
        print(f"{arm} {p['id']:<14} gen {len(r['tokens']):4d}  decode {t['predicted_per_second']:7.1f} tok/s  "
              f"acc_len {acc if acc is None else round(acc, 2)}", flush=True)
