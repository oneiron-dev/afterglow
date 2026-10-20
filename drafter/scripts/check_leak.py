import json, sys
D = sys.argv[1]
tr = [json.loads(l) for l in open(f"{D}/data/prompts_train.jsonl")]
ev = [json.loads(l) for l in open(f"{D}/data/prompts_eval.jsonl")]
gen = [json.loads(l) for l in open(f"{D}/data/gen_train.jsonl")]
seqs = json.load(open(f"{D}/data/seqs.json"))
sp = json.load(open(f"{D}/data/split.json"))
trid = {p["id"] for p in tr}; genid = {g["id"] for g in gen}; seqid = {s["id"] for s in seqs}
print("train", len(tr), "eval", len(ev), "gen", len(gen), "seqs", len(seqs), "seq train/val", sum(s["split"]=="train" for s in seqs), sum(s["split"]=="val" for s in seqs))
print("gen ids subset of train prompts:", genid <= trid, "| seqs subset of gen:", seqid <= genid)
evid = {p["id"] for p in ev}
print("eval id in train/gen/seqs:", evid & trid, evid & genid, evid & seqid)
# id stems (strip prefix) e.g. ev-L00235-c vs tr-L00235-c0
evlife = {p["life_id"] for p in ev if p["life_id"]}; trlife = {p["life_id"] for p in tr if p["life_id"]}
print("eval lives", len(evlife), "overlap with train lives:", evlife & trlife)
print("eval lives == split.eval_lives:", evlife <= set(sp["eval_lives"]), "| split eval ∩ train:", set(sp["eval_lives"]) & set(sp["train_lives"]), "| spare ∩ train:", set(sp["spare_lives"]) & set(sp["train_lives"]))
print("public_train/public_eval:", sp["public_train"], "/", sp["public_eval"], "| counts", sp["counts"])
# exact text
trtext = {p["prompt"] for p in tr}; trids = {tuple(p["prompt_ids"]) for p in tr}
print("eval prompt text exactly in train:", sum(p["prompt"] in trtext for p in ev), "| eval prompt_ids exactly in train:", sum(tuple(p["prompt_ids"]) in trids for p in ev))
big = "\n\x00".join(p["prompt"] for p in tr) + "\n\x00" + "\n\x00".join(g["text"] for g in gen)
# substring: full eval prompt body (after the chat header), and every user/system message content of each eval prompt
def contents(p):
    out = []
    for m in p["messages"]:
        c = m["content"]
        if isinstance(c, list):
            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
        if c and len(c) >= 40:
            out.append((m["role"], c))
    return out
hits = []
for p in ev:
    body = p["prompt"].split("<|message|>", 1)[1] if "<|message|>" in p["prompt"] else p["prompt"]
    if body in big: hits.append((p["id"], "whole prompt body"))
    for role, c in contents(p):
        if c in big: hits.append((p["id"], role, c[:60]))
print("eval message contents (>=40 chars) found verbatim in training prompts/replies:", len(hits))
for h in hits[:20]: print("  ", h)
# last user turn of each eval prompt, any length >= 15 chars
lh = []
for p in ev:
    us = [m["content"] for m in p["messages"] if m["role"] == "user"]
    u = us[-1] if us else ""
    if isinstance(u, str) and len(u) >= 15 and u in big: lh.append((p["id"], u[:60]))
print("eval final user turns found in training text:", lh)
# token-id containment: eval prompt_ids (after the shared prefix) as a sublist of any training sequence
import itertools
def contains(hay, needle):
    n = len(needle); f = needle[0]
    return any(hay[i:i+n] == needle for i in range(len(hay)-n+1) if hay[i] == f)
trseq = [p["prompt_ids"] for p in tr]
tid = sum(any(contains(h, p["prompt_ids"][8:]) for h in trseq) for p in ev)
print("eval prompt_ids[8:] contained in any training prompt_ids:", tid)
print("eval srcs:", {s: sum(p['src']==s for p in ev) for s in ['life-chat','life-recall','public-chat']}, "train srcs:", {s: sum(p['src']==s for p in tr) for s in ['life-chat','life-recall','public-chat']})
