#!/usr/bin/env python3
"""Generate the target's own greedy outputs for a prompt set through a running llama-server (stdlib only).
usage: gen_client.py PORT PROMPTS.jsonl OUT.jsonl N_PREDICT CONCURRENCY [DEADLINE_EPOCH]"""
import json, sys, time, urllib.request, concurrent.futures as cf, os, threading
port, src, out, n_predict, conc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
deadline = float(sys.argv[6]) if len(sys.argv) > 6 else 1e18
url = f"http://127.0.0.1:{port}"
def post(path, body, timeout=900):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))
prompts = [json.loads(l) for l in open(src)]
# tokenizer parity: the GGUF vocab must give the same ids as the HF tokenizer used to build the prompts
mism = 0
for p in prompts[:8]:
    ids = post("/tokenize", {"content": p["prompt"], "add_special": False, "parse_special": True})["tokens"]
    mism += ids != p["prompt_ids"]
print(f"tokenizer parity: {8 - mism}/8 identical", flush=True)
assert mism == 0, "GGUF tokenizer != HF tokenizer"
done = set()
if os.path.exists(out):
    done = {json.loads(l)["id"] for l in open(out)}
todo = [p for p in prompts if p["id"] not in done]
lock = threading.Lock(); t0 = time.time(); ntok = [0, 0]
def run(p):
    if time.time() > deadline: return None
    r = post("/completion", {"prompt": p["prompt_ids"], "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
                             "cache_prompt": False, "return_tokens": True, "stream": False})
    rec = {"id": p["id"], "src": p["src"], "gen_ids": r["tokens"], "text": r["content"], "n_gen": len(r["tokens"]),
           "stop": r.get("stop_type"), "timings": r.get("timings")}
    with lock:
        with open(out, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        ntok[0] += 1; ntok[1] += rec["n_gen"]
        if ntok[0] % 50 == 0:
            el = time.time() - t0
            print(f"{ntok[0]}/{len(todo)} done, {ntok[1]} gen tok, {ntok[1]/el:.1f} tok/s aggregate, {el/60:.1f} min", flush=True)
    return rec
with cf.ThreadPoolExecutor(conc) as ex:
    list(ex.map(run, todo))
el = time.time() - t0
print(f"GEN-DONE {ntok[0]} prompts, {ntok[1]} tokens, {el/60:.1f} min, {ntok[1]/max(el,1):.1f} tok/s", flush=True)
