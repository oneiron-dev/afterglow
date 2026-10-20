#!/usr/bin/env python3
"""Build G1+G2 prompts by REAL text tokenization on CPU (no model load).

Standard CLI: python3 serve/cuda/make_prompts.py --config $GLIMMER_HOME/config.json --write-prompts
Run with the existing project env (no new downloads, no model):
  $GLIMMER_HOME/venv/bin/python make_prompts.py --config ... --write-prompts
(falls back to system python3 when it already imports tokenizers; otherwise exit 2
with an explicit message. Never pip-installs.)

Method (documented constants below):
- Tokenizer: config prompts.tokenizer (default
  $GLIMMER_MODELS/tokenizer.json), Tokenizer.from_file,
  add_special_tokens=False. Identity recorded (file sha256, lib version, vocab size).
- Parity gate: re-encode the 5 heldout `prompt` texts; ALL must equal stored
  prompt_ids exactly, else TOKENIZER-MISMATCH (exit 3, identity + first-diff index
  recorded; --allow-mismatch writes with parity:false instead).
- Synthetic TEXT: fixed SEED_PARAS (neutral filler, no PII) joined to a block;
  block token length measured once; repeats R = ceil(33280/block_len)+1 (recorded);
  SYNTH_TEXT = block * R (fixed, deterministic per tokenizer).
- Base IDs = encode(SYNTH_TEXT); prompts are HEAD-N truncations (documented):
  synth-4096-v1/snap-8192-v1/prefix-30720-v1/snap-32768-v1 = base[:N] (prefix chain).
- Suffix rows are fixed-offset slices AFTER 32768 (no overlap with any head-N):
  g2-suffix-64-v1=base[32768:32832], prefix-suffix-a=base[32864:32928],
  prefix-suffix-b=base[32960:33024]; prefix-miss-30720-v1 = prefix copy whose first
  16 IDs come from base[33056:33072] (verified different, else +64 shift, logged).
Outputs: prompts_g1.jsonl (7 rows) + prompts_g2.jsonl (7 rows, canonical cache IDs)
  + .sha256 files. --self-test needs no tokenizer (text/arithmetic only).
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

SYNTH_VERSION = "v2-text"
ADD_SPECIAL_TOKENS = False
BASE_MIN_LEN = 33280
G2_SUFFIX_OFF = 32768
SUFA_OFF = 32864
SUFB_OFF = 32960
MISS_OFF = 33056
SUFFIX_LEN = 64
MISS_LEN = 16

SEED_PARAS = [
    "The benchmark preamble states the working agreement for synthetic evaluation text. "
    "All filler paragraphs are fixed, public, and carry no personal data. They exist only "
    "to occupy context positions with natural token statistics during prefill and decode measurement.",
    "A second paragraph continues the fixed sequence with ordinary technical prose. It describes "
    "scheduling, queues, and retries in plain terms: jobs wait their turn, failures are recorded "
    "with their exact identifiers, and no result is claimed without its counters attached.",
    "Numbers appear here in stable form for tokenizer coverage: 0 1 2 3 7 12 42 99 256 1024 4096 "
    "8192 32768 3.14159 2.71828. Punctuation mixes commas, colons: semicolons; dashes - quotes \"hi\" "
    "and (parentheses) [brackets] {braces} across short sentences.",
    "A short list follows. First, determinism: the same text must tokenize to the same identifiers "
    "on every run. Second, exactness: occupied length is counted in tokens, never estimated from "
    "characters. Third, honesty: truncated regions are labeled with their rule, not hidden.",
    "Code-like filler adds lexical variety without meaning: def fill(n): return [tok(i) for i in "
    "range(n)] # fixed seed. Paths such as /data/bench/prompts and keys like prompt_ids_sha appear "
    "as literal text only. No path here is executed and no key is secret.",
    "Multilingual filler mirrors the cohort mix in miniature. Japanese: \u3053\u308c\u306f\u56fa\u5b9a\u306e"
    "\u6e2c\u6e96\u30c6\u30ad\u30b9\u30c8\u3067\u3059\u3002 Ukrainian: \u0446\u0435 \u0444\u0456\u043a\u0441\u043e"
    "\u0432\u0430\u043d\u0438\u0439 \u0442\u0435\u0441\u0442\u043e\u0432\u0438\u0439 \u0442\u0435\u043a\u0441\u0442\u002e "
    "Both lines are fixed and carry no personal data.",
    "The closing paragraph of the block restates the rule: prompts are head-N truncations of one "
    "measured base encoding, suffixes are fixed-offset slices beyond every head length, and the "
    "miss variant differs in exactly its first sixteen identifiers. Nothing is random.",
    "Boundary marker paragraph with a stable nonce line: SYNTH-BLOCK-V2 8f3a1c numeric tail 271828. "
    "Repeating the block preserves determinism because every repetition is literal text: the same "
    "characters in the same order, tokenized once by the recorded tokenizer build.",
]

G1_LENGTHS = {"synth-4096-v1": 4096, "synth-32768-v1": 32768}
G2_HEADS = {"snap-8192-v1": 8192, "snap-32768-v1": 32768, "prefix-30720-v1": 30720}


def build_block_text():
    return "\n\n".join(SEED_PARAS) + "\n\n"


def build_synth_text(block_len_tokens):
    """Repeat block to cover BASE_MIN_LEN. Returns (text, repeats). Deterministic."""
    reps = math.ceil(BASE_MIN_LEN / max(1, block_len_tokens)) + 1
    return build_block_text() * reps, reps


def head_ids(base, n):
    assert len(base) >= n, "base %d shorter than head %d" % (len(base), n)
    return list(base[:n])


def diff_slice(base, off, ln, forbids, label, log):
    """Fixed-offset slice verified different from each forbids entry; +64 shift, logged."""
    for attempt in range(5):
        o = off + attempt * 64
        s = list(base[o:o + ln])
        assert len(s) == ln, "base too short for %s at %d" % (label, o)
        if all(s != list(f) for f in forbids):
            if attempt:
                log("%s shifted +%d to offset %d (collision avoided)" % (label, attempt * 64, o))
            return s, o
    raise RuntimeError("cannot place %s without collision" % label)


def load_heldout(eval_source, heldout_ids):
    rows = {}
    with open(eval_source) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("id") in set(heldout_ids):
                rows[r["id"]] = r
    missing = [i for i in heldout_ids if i not in rows]
    return rows, missing


def verify_parity(tok, rows, heldout_ids):
    """Re-encode heldout prompt texts; return (ok_all, details[])."""
    details = []
    ok_all = True
    for hid in heldout_ids:
        r = rows[hid]
        text = r.get("prompt", "")
        want = r.get("prompt_ids", [])
        try:
            got = tok.encode(text, add_special_tokens=ADD_SPECIAL_TOKENS).ids
        except Exception as e:
            details.append({"id": hid, "ok": False, "error": str(e)[:300]})
            ok_all = False
            continue
        if list(got) == list(want):
            details.append({"id": hid, "ok": True, "n": len(want)})
        else:
            diff = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                        min(len(got), len(want)))
            details.append({"id": hid, "ok": False, "got_n": len(got), "want_n": len(want),
                            "first_diff": diff})
            ok_all = False
    return ok_all, details


def main():
    ap = argparse.ArgumentParser(description="Real text tokenization for G1+G2 prompts (CPU-only).")
    ap.add_argument("--config", default=os.path.join(common.REMOTE_DIR, "config.json"))
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--eval-source", default=None)
    ap.add_argument("--out-g1", default=None)
    ap.add_argument("--out-g2", default=None)
    ap.add_argument("--write-prompts", action="store_true")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="write with parity:false instead of exit 3 on tokenizer mismatch")
    ap.add_argument("--self-test", action="store_true", help="no tokenizer: text+arithmetic only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        t1 = build_block_text()
        assert t1 == build_block_text(), "block text non-deterministic"
        print("SELF-TEST block chars=%d sha=%s" % (len(t1), common.sha256_text(t1)[:16]))
        fake = list(range(50000, 50000 + 34000))
        assert len(head_ids(fake, 4096)) == 4096 and len(head_ids(fake, 32768)) == 32768
        logs = []
        s, o = diff_slice(fake, G2_SUFFIX_OFF, 64, [], "t", logs.append)
        assert o == G2_SUFFIX_OFF and len(s) == 64
        # forced collision: forbid the natural slice -> must shift
        s2, o2 = diff_slice(fake, G2_SUFFIX_OFF, 64, [s], "t2", logs.append)
        assert o2 == G2_SUFFIX_OFF + 64 and s2 != s
        print("SELF-TEST arithmetic+collision-shift PASS", logs)
        print("SELF-TEST version=%s offsets=%s" % (SYNTH_VERSION, [G2_SUFFIX_OFF, SUFA_OFF, SUFB_OFF, MISS_OFF]))
        return 0

    cfg = common.load_config(args.config)
    heldout_ids = list(cfg["prompts"]["heldout_ids"])
    tok_path = args.tokenizer or cfg["prompts"].get("tokenizer", "")
    if not tok_path or not os.path.exists(tok_path):
        # local fallbacks (never download)
        cands = [tok_path,
                 common.MODELS_DIR + "/tokenizer.json",
                 common.MODELS_DIR + "/target-nf4/tokenizer.json"]
        tok_path, _ = common.resolve_first_existing(*[c for c in cands if c])
    eval_source = args.eval_source or cfg["prompts"].get("eval_source", "")
    if not os.path.exists(eval_source):
        alt = common.DEFAULT_CONFIG["prompts"]["eval_source"]
        if os.path.exists(alt):
            eval_source = alt
    out_g1 = args.out_g1 or cfg["prompts"].get("g1_prompts", "")
    out_g2 = args.out_g2 or cfg["prompts"].get("g2_prompts", "")
    print("tokenize config=%s tok=%s eval=%s" % (args.config, tok_path, eval_source), flush=True)
    print("tokenize out_g1=%s out_g2=%s" % (out_g1, out_g2), flush=True)
    if not tok_path or not os.path.exists(tok_path):
        print("tokenize BLOCKED: tokenizer.json missing: %s" % tok_path, flush=True)
        return 2
    if not os.path.exists(eval_source):
        print("tokenize BLOCKED: eval source missing: %s" % eval_source, flush=True)
        return 2
    try:
        from tokenizers import Tokenizer
        import tokenizers as tok_lib
    except Exception as e:
        print("tokenize BLOCKED: no tokenizers lib in %s (%s); run with "
              "a Python that has `tokenizers` installed" % (sys.executable, str(e)[:200]), flush=True)
        return 2
    tok = Tokenizer.from_file(tok_path)
    try:
        vocab_size = tok.get_vocab_size()
    except Exception:
        vocab_size = "unknown"
    tok_identity = {"path": tok_path, "sha256": common.sha256_file(tok_path),
                    "lib": getattr(tok_lib, "__version__", "unknown"),
                    "vocab_size": vocab_size, "python": sys.executable,
                    "add_special_tokens": ADD_SPECIAL_TOKENS}
    print("tokenize identity=%s" % json.dumps(tok_identity)[:400], flush=True)

    rows, missing = load_heldout(eval_source, heldout_ids)
    if missing:
        print("tokenize BLOCKED: heldout IDs missing: %s" % missing, flush=True)
        return 2
    ok_parity, details = verify_parity(tok, rows, heldout_ids)
    for d in details:
        print("tokenize parity %s ok=%s" % (d["id"], d["ok"]), flush=True)
    if not ok_parity and not args.allow_mismatch:
        print("tokenize TOKENIZER-MISMATCH: refusing to write (identity=%s). "
              "Rerun with --allow-mismatch to record identity explicitly." % json.dumps(tok_identity)[:300],
              flush=True)
        print(json.dumps(details), flush=True)
        return 3

    # measure block, build full text, encode once
    block_text = build_block_text()
    block_ids = tok.encode(block_text, add_special_tokens=ADD_SPECIAL_TOKENS).ids
    print("tokenize block tokens=%d" % len(block_ids), flush=True)
    synth_text, reps = build_synth_text(len(block_ids))
    base = tok.encode(synth_text, add_special_tokens=ADD_SPECIAL_TOKENS).ids
    base = [int(x) for x in base]
    print("tokenize reps=%d base_len=%d need>=%d" % (reps, len(base), BASE_MIN_LEN), flush=True)
    if len(base) < BASE_MIN_LEN:
        print("tokenize BLOCKED: base too short", flush=True)
        return 2
    text_sha = common.sha256_text(synth_text)
    log_lines = []
    g2_suffix, g2_off = diff_slice(base, G2_SUFFIX_OFF, SUFFIX_LEN, [], "g2-suffix-64", log_lines.append)
    sufa, sufa_off = diff_slice(base, SUFA_OFF, SUFFIX_LEN, [g2_suffix], "suffix-a", log_lines.append)
    sufb, sufb_off = diff_slice(base, SUFB_OFF, SUFFIX_LEN, [g2_suffix, sufa], "suffix-b", log_lines.append)
    for ln in log_lines:
        print("tokenize " + ln, flush=True)

    meta = {"synth_version": SYNTH_VERSION, "tokenizer": tok_identity, "parity": ok_parity,
            "parity_details": details, "text_sha": text_sha, "block_tokens": len(block_ids),
            "repeats": reps, "base_len": len(base), "truncation": "head-N",
            "offsets": {"g2_suffix": g2_off, "suffix_a": sufa_off, "suffix_b": sufb_off}}

    g1_rows = []
    for hid in heldout_ids:
        r = rows[hid]
        ids = [int(x) for x in r.get("prompt_ids", [])]
        g1_rows.append({"id": hid, "kind": "heldout", "src": r.get("src", "unknown"),
                        "lang": r.get("lang", "unknown"), "n_prompt": len(ids),
                        "prompt_ids": ids, "prompt_ids_sha": common.sha256_ids(ids),
                        "eval_source": eval_source, **{"meta": meta}})
    for sid, n in G1_LENGTHS.items():
        ids = head_ids(base, n)
        g1_rows.append({"id": sid, "kind": "synthetic", "src": "synth-text-v2",
                        "n_prompt": len(ids), "prompt_ids": ids,
                        "prompt_ids_sha": common.sha256_ids(ids), "meta": meta})
    g2_rows = []
    for sid, n in G2_HEADS.items():
        ids = head_ids(base, n)
        g2_rows.append({"id": sid, "kind": "synthetic", "src": "synth-text-v2",
                        "n_prompt": len(ids), "prompt_ids": ids,
                        "prompt_ids_sha": common.sha256_ids(ids), "meta": meta})
    g2_rows.append({"id": "g2-suffix-64-v1", "kind": "suffix", "src": "synth-text-v2",
                    "n_prompt": 64, "prompt_ids": g2_suffix,
                    "prompt_ids_sha": common.sha256_ids(g2_suffix), "meta": meta})
    g2_rows.append({"id": "prefix-suffix-a-64-v1", "kind": "suffix", "src": "synth-text-v2",
                    "n_prompt": 64, "prompt_ids": sufa,
                    "prompt_ids_sha": common.sha256_ids(sufa), "meta": meta})
    g2_rows.append({"id": "prefix-suffix-b-64-v1", "kind": "suffix", "src": "synth-text-v2",
                    "n_prompt": 64, "prompt_ids": sufb,
                    "prompt_ids_sha": common.sha256_ids(sufb), "meta": meta})
    prefix = head_ids(base, 30720)
    miss_head, miss_off = diff_slice(base, MISS_OFF, MISS_LEN, [prefix[:MISS_LEN]], "miss-head", log_lines.append)
    miss = list(miss_head) + prefix[MISS_LEN:]
    assert len(miss) == 30720 and miss[:MISS_LEN] != prefix[:MISS_LEN]
    g2_rows.append({"id": "prefix-miss-30720-v1", "kind": "synthetic-miss", "src": "synth-text-v2",
                    "n_prompt": 30720, "prompt_ids": miss,
                    "prompt_ids_sha": common.sha256_ids(miss),
                    "miss_head_off": miss_off, "meta": meta})
    for r in g1_rows + g2_rows:
        print("tokenize row %s n=%d sha=%s" % (r["id"], r["n_prompt"], r["prompt_ids_sha"][:16]), flush=True)
    if args.dry_run and not args.write_prompts:
        print("tokenize dry-run: 0 files written", flush=True)
        return 0
    if not args.write_prompts:
        print("tokenize: pass --write-prompts to write files", flush=True)
        return 0
    for path, orows in ((out_g1, g1_rows), (out_g2, g2_rows)):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            for r in orows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path + ".sha256", "w") as f:
            f.write(common.sha256_file(path) + "  " + os.path.basename(path) + "\n")
        print("tokenize wrote %s (%d rows) + .sha256" % (path, len(orows)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
