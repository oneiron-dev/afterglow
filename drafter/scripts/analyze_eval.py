#!/usr/bin/env python3
"""Summarise an eval dir: acceptance length, decode tok/s, speedup vs the target alone, losslessness, VRAM.
usage: analyze_eval.py EVAL_DIR [ARM ...]   (first arm = the target-alone baseline)"""
import json, os, sys
d = sys.argv[1]
arms = sys.argv[2:] or ["target", "stock_official", "stock_mine", "tuned"]
R = {a: [json.loads(l) for l in open(f"{d}/{a}.jsonl")] for a in arms if os.path.exists(f"{d}/{a}.jsonl")}
vram = {}
if os.path.exists(f"{d}/arms.txt"):
    for line in open(f"{d}/arms.txt"):
        p = line.split()
        vram[p[0]] = (int(p[2]), int(p[4]))
base = arms[0]
def agg(rows):
    tok = sum(r["timings"]["predicted_n"] for r in rows)
    ms = sum(r["timings"]["predicted_ms"] for r in rows)
    acc = sum(r["spec"]["spec_decode_num_accepted_tokens_total"] for r in rows)
    steps = sum(r["spec"]["spec_decode_num_drafts_total"] for r in rows)
    drafted = sum(r["spec"]["spec_decode_num_draft_tokens_total"] for r in rows)
    return {"tokens": tok, "decode_tok_s": tok / ms * 1000 if ms else 0.0,
            "acc_len": (1 + acc / steps) if steps else None, "accept_rate": (acc / drafted) if drafted else None,
            "verify_steps": steps}
out = {"arms": {}, "by_src": {}}
srcs = ["life-chat", "life-recall", "public-chat"]
for a, rows in R.items():
    s = agg(rows)
    s["speedup"] = s["decode_tok_s"] / agg(R[base])["decode_tok_s"]
    ref = {r["id"]: r["gen_ids"] for r in R[base]}
    same = [r["gen_ids"] == ref.get(r["id"]) for r in rows]
    s["identical_first5"] = sum(same[:5]); s["identical_all"] = sum(same); s["n"] = len(rows)
    if a in vram:
        s["peak_vram_mib"], s["wall_s"] = vram[a]
    out["arms"][a] = s
    for src in srcs:
        sub = [r for r in rows if r["src"] == src]
        bs = agg([r for r in R[base] if r["src"] == src])
        x = agg(sub); x["speedup"] = x["decode_tok_s"] / bs["decode_tok_s"] if bs["decode_tok_s"] else None
        out["by_src"].setdefault(src, {})[a] = x
json.dump(out, open(f"{d}/summary.json", "w"), indent=1)
print(f"{'arm':<16}{'acc len':>9}{'accept %':>10}{'decode tok/s':>14}{'speedup':>9}{'same 5/5':>10}{'same 25':>9}{'VRAM MiB':>10}")
for a, s in out["arms"].items():
    al = f"{s['acc_len']:.2f}" if s["acc_len"] else "-"
    ar = f"{100*s['accept_rate']:.1f}" if s["accept_rate"] else "-"
    print(f"{a:<16}{al:>9}{ar:>10}{s['decode_tok_s']:>14.1f}{s['speedup']:>8.2f}x{s['identical_first5']:>8}/5{s['identical_all']:>6}/{s['n']}{s.get('peak_vram_mib', 0):>10}")
for src in srcs:
    print(f"  {src}: " + "  ".join(f"{a} acc {x['acc_len']:.2f} {x['decode_tok_s']:.0f} tok/s ({x['speedup']:.2f}x)" if x["acc_len"] else
                                  f"{a} {x['decode_tok_s']:.0f} tok/s" for a, x in out["by_src"][src].items()))
