import json, re, sys
D = sys.argv[1]
arms = ["target", "stock_official", "stock_mine", "tuned", "tuned_lr6e-05"]
R = {a: [json.loads(l) for l in open(f"{D}/{a}.jsonl")] for a in arms}
ev = [json.loads(l) for l in open(f"{D}/../data/prompts_eval.jsonl")]
print("eval ids order matches jsonl:", all([r["id"] for r in R[a]] == [p["id"] for p in ev] for a in arms))
# --- server-log parse (independent of /metrics and the jsonl)
pt = re.compile(r"task (\d+) \|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
pa = re.compile(r"task (\d+) \| draft acceptance = [\d.]+ \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)")
def logparse(a):
    txt = open(f"{D}/server-{a}.log").read()
    ev_ = [(int(t), float(ms), int(n)) for t, ms, n in pt.findall(txt)]
    ac = [(int(t), int(x), int(g), float(m)) for t, x, g, m in pa.findall(txt)]
    return ev_, ac
base = None
out = {}
for a in arms:
    rows = R[a]
    tok = sum(r["timings"]["predicted_n"] for r in rows); ms = sum(r["timings"]["predicted_ms"] for r in rows)
    ng = sum(len(r["gen_ids"]) for r in rows)
    assert all(r["n_gen"] == len(r["gen_ids"]) == r["timings"]["predicted_n"] for r in rows), a
    tps = tok / ms * 1000
    # metrics-based
    acc = sum(r["spec"]["spec_decode_num_accepted_tokens_total"] for r in rows)
    st = sum(r["spec"]["spec_decode_num_drafts_total"] for r in rows)
    dr = sum(r["spec"]["spec_decode_num_draft_tokens_total"] for r in rows)
    # timings-based
    acc_t = sum(r["timings"].get("draft_n_accepted", 0) for r in rows); dr_t = sum(r["timings"].get("draft_n", 0) for r in rows)
    # identity tokens = 1 + steps + accepted per request
    ident = all(r["timings"]["predicted_n"] == 1 + r["spec"]["spec_decode_num_drafts_total"] + r["spec"]["spec_decode_num_accepted_tokens_total"] for r in rows) if st else None
    # alt acc_len: tokens after first / steps
    alt = (tok - len(rows)) / st if st else None
    # server log, skip the 2 warm-ups
    evl, acl = logparse(a)
    evl = evl[2:]; acl = acl[2:] if acl else acl
    log_tok = sum(n for _, _, n in evl); log_ms = sum(ms_ for _, ms_, _ in evl)
    log_acc = sum(x for _, x, _, _ in acl); log_gen = sum(g for _, _, g, _ in acl)
    perprompt_mean = sum(m for *_, m in acl) / len(acl) if acl else None
    out[a] = dict(n=len(rows), tok=tok, ms=ms, tps=tps, acc_len=(1 + acc / st) if st else None, alt_acc=alt,
                  acc_rate=acc / dr if dr else None, steps=st, acc=acc, drafted=dr, acc_t=acc_t, dr_t=dr_t, ident=ident,
                  log_n=len(evl), log_tok=log_tok, log_tps=log_tok / log_ms * 1000, log_acc=log_acc, log_gen=log_gen,
                  perprompt_mean=perprompt_mean)
b = out["target"]["tps"]; bl = out["target"]["log_tps"]
for a, s in out.items():
    s["speedup"] = s["tps"] / b; s["log_speedup"] = s["log_tps"] / bl
    print(a, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
# per source
for src in ["life-chat", "life-recall", "public-chat"]:
    line = [src]
    bt = [r for r in R["target"] if r["src"] == src]
    btps = sum(r["timings"]["predicted_n"] for r in bt) / sum(r["timings"]["predicted_ms"] for r in bt) * 1000
    for a in arms:
        rs = [r for r in R[a] if r["src"] == src]
        tok = sum(r["timings"]["predicted_n"] for r in rs); ms = sum(r["timings"]["predicted_ms"] for r in rs)
        st = sum(r["spec"]["spec_decode_num_drafts_total"] for r in rs); acc = sum(r["spec"]["spec_decode_num_accepted_tokens_total"] for r in rs)
        line.append(f"{a}: n={len(rs)} tok={tok} acc={(1+acc/st) if st else 0:.3f} sp={tok/ms*1000/btps:.3f}")
    print(" | ".join(line))
# per-prompt win/loss tuned vs stock_mine
w = l = t = 0
for x, y in zip(R["tuned"], R["stock_mine"]):
    ax = 1 + x["spec"]["spec_decode_num_accepted_tokens_total"] / x["spec"]["spec_decode_num_drafts_total"]
    ay = 1 + y["spec"]["spec_decode_num_accepted_tokens_total"] / y["spec"]["spec_decode_num_drafts_total"]
    if abs(ax - ay) < 1e-9: t += 1
    elif ax > ay: w += 1
    else: l += 1
print("tuned vs stock_mine per prompt: win", w, "loss", l, "tie", t)
# losslessness
ref = {r["id"]: r["gen_ids"] for r in R["target"]}
print("min gen len per arm:", {a: min(len(r["gen_ids"]) for r in R[a]) for a in arms})
print("all gen_ids are int lists:", all(isinstance(r["gen_ids"], list) and all(isinstance(i, int) for i in r["gen_ids"]) for a in arms for r in R[a]))
for a in arms[1:]:
    same = [r["gen_ids"] == ref[r["id"]] for r in R[a]]
    div = []
    for r in R[a]:
        g, h = r["gen_ids"], ref[r["id"]]
        if g != h:
            k = next((i for i, (u, v) in enumerate(zip(g, h)) if u != v), min(len(g), len(h)))
            div.append((r["id"], k))
    print(a, "first5", sum(same[:5]), "all", sum(same), "first5 ids", [r["id"] for r in R[a]][:5], "divergences", div)
for a in ["stock_mine", "tuned", "tuned_lr6e-05"]:
    print(f"{a} == stock_official on", sum(x["gen_ids"] == y["gen_ids"] for x, y in zip(R[a], R["stock_official"])), "/25")
print("tuned == stock_mine on", sum(x["gen_ids"] == y["gen_ids"] for x, y in zip(R["tuned"], R["stock_mine"])), "/25")
