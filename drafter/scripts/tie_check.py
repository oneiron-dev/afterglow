#!/usr/bin/env python3
"""Where a DFlash arm's greedy output leaves the target-alone output, read the target's own top-2 there two ways:
(a) single-token decode (the target-alone path), (b) the same prefix evaluated as one batched prompt (the verify path).
usage: tie_check.py PORT EVAL_DIR OUT.json"""
import json, math, sys, urllib.request
port, d, out = sys.argv[1], sys.argv[2], sys.argv[3]
url = f"http://127.0.0.1:{port}"
def post(body):
    req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))
prompts = {json.loads(l)["id"]: json.loads(l) for l in open("data/prompts_eval.jsonl")}
T = {json.loads(l)["id"]: json.loads(l) for l in open(f"{d}/target.jsonl")}
S = {json.loads(l)["id"]: json.loads(l) for l in open(f"{d}/stock_official.jsonl")}
def top(e):
    lst = e.get("top_logprobs") or e.get("top_probs") or e.get("probs") or []
    return [(x.get("id"), round(math.exp(x["logprob"]) if "logprob" in x else x.get("prob", 0.0), 4)) for x in lst[:2]]
res = []
for pid, t in T.items():
    a, b = t["gen_ids"], S[pid]["gen_ids"]
    if a == b:
        continue
    k = next(j for j in range(min(len(a), len(b))) if a[j] != b[j])
    pre = prompts[pid]["prompt_ids"] + a[:k]
    # (a) decode path: generate k+1 tokens one by one from the prompt, read the probs of token k
    r1 = post({"prompt": prompts[pid]["prompt_ids"], "n_predict": k + 1, "temperature": 0.0, "top_k": 1, "n_probs": 2,
               "cache_prompt": False, "return_tokens": True})
    p_dec = top(r1["completion_probabilities"][k])
    # (b) batched path: the whole prefix as one prompt, next token
    r2 = post({"prompt": pre, "n_predict": 1, "temperature": 0.0, "top_k": 1, "n_probs": 2, "cache_prompt": False, "return_tokens": True})
    p_bat = top(r2["completion_probabilities"][0])
    res.append({"id": pid, "pos": k, "target_tok": a[k], "dflash_tok": b[k],
                "decode_top2": p_dec, "batched_top2": p_bat})
    print(json.dumps(res[-1]), flush=True)
json.dump(res, open(out, "w"), indent=1)
