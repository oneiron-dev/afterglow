#!/usr/bin/env python3
"""Day-one QLoRA smoke: train a small adapter on a synthetic day, score recall and a general slice before/after.

Scope: small synthetic smoke only. Proves recall/general metrics can be produced.
Not the preregistered 200-item eval. No training on import. No GPU work unless run.

Recipe (the proposed nightly pilot; differs from rate_both.py):
  rank 16, alpha 32, dropout 0.05, lr 1e-4, batch 1, gradacc 4,
  q/k/v/o in final 16 text layers (36-51 of 52), vision/embed/head frozen,
  response-only loss (ATEM to=user), grad checkpoint unsloth, loss nll,
  1 epoch over 34 canonical rows (~9 optimizer steps), seed 3407,
  max_seq_length 2048, warmup 1, optim adamw_8bit, scheduler constant.

Eval (smoke-sized, smaller than the full preregistered protocol):
  30 facts x1 heldout (NOT 90), 3 prefs + 1 procedure descriptive,
  10-item all-synthetic general slice (4 arithmetic, 3 reasoning, 3 tool JSON),
  greedy, max_new_tokens 96, truncation=fail, forced to=user answer channel,
  reasoning_strength minimal, 2 arms (base/no-notes, adapted/no-notes) only.

Usage (on a CUDA host, under a hard timeout):
  python smoke_eval.py --validate-only --train-rows train_rows.json \
    --recall-probes recall_probes.json --general-slice general_slice.json
  python smoke_eval.py --model-dir /path/to/base-4bit --train-rows train_rows.json \
    --recall-probes recall_probes.json --general-slice general_slice.json \
    --adapter-out ./smoke-adapter --results-out ./smoke-results.json

Reuse already-loaded base (saves setup, only if base has NO LoRA):
  import smoke_eval
  res = smoke_eval.run_smoke_with_loaded(model, tokenizer,
      train_rows_path="train_rows.json", recall_path="recall_probes.json",
      general_path="general_slice.json", adapter_out="./smoke-adapter",
      results_out="./smoke-results.json", base_model_dir="/path/to/base-4bit",
      clean_reload=True)

Exit codes: 0 ok, 2 structured GPU/train/eval failure (no fake numbers), 3 validation failure.
Synthetic only. No real user data. No public MMLU.
"""

import argparse
import hashlib
import json
import os
import random
import re
import resource
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# Fixed smoke constants (frozen; changing them requires new freeze + new hashes)
SMOKE_VERSION = "c-measure-smoke-v1+gpufork1"  # GPU-worker compat fork; methodology unchanged (see SMOKE_FORK_NOTES)
REASONING_STRENGTH = "minimal"
KNOWLEDGE_CUTOFF = "2026-01-04"
CURRENT_DATE = "2026-10-14"
MAX_NEW_TOKENS = 96
MAX_SEQ_LENGTH = 2048
RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
GRAD_ACCUM = 4
WARMUP_STEPS = 1
EPOCHS = 1
SEED = 3407
OPTIM = "adamw_8bit"
SCHEDULER = "constant"
LOSS_TYPE = "nll"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
FINAL16_LAYERS = list(range(36, 52))
NUM_TEXT_LAYERS = 52
INSTRUCTION_PART = "<|start|>user<|message|>"
RESPONSE_PART = "<|start|>assistant to=user<|message|>"
FORCED_ANSWER_PREFIX = "<|start|>assistant to=user<|message|>"
EOS_IDS = [200001, 200008]
PAD_ID = 200018

# ---------- pure-python scoring (no torch) ----------

def normalize_text(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def strip_answer(raw):
    if raw is None:
        return ""
    t = str(raw)
    # Cut at first end marker
    for marker in ["<|eot|>", "<|eom|>", "<|end_of_text|>", "<|reserved_special_token"]:
        if marker in t:
            t = t.split(marker, 1)[0]
    return t.strip()

def normalize_fact_text(s):
    t = normalize_text(strip_answer(s))
    t = re.sub(r"[-_]+", " ", t)
    t = t.replace("%", " percent ")
    t = re.sub(r"\bkilometers\b", "kilometres", t)
    t = re.sub(r"\bkm/h\b", "kilometres per hour", t)
    t = re.sub(r"\bkph\b", "kilometres per hour", t)
    t = re.sub(r"\bliters\b", "litres", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*l\b", r"\1 litres", t)
    t = re.sub(r"\bmins\b", "minutes", t)
    t = re.sub(r"\bmin\b", "minutes", t)
    t = re.sub(r"\b0(\d:\d\d)\b", r"\1", t)
    t = re.sub(r"\b(\d+\.\d*?)0+\b", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def score_fact(answer, gold):
    a = normalize_fact_text(answer)
    g = normalize_fact_text(gold)
    if not g:
        return False
    pat = r"(?<![a-z0-9])" + re.escape(g) + r"(?![a-z0-9])"
    return re.search(pat, a) is not None

def score_exact_number(answer, gold):
    a = strip_answer(answer)
    # Take first line, remove commas/spaces
    a = a.splitlines()[0] if a.splitlines() else ""
    a = a.replace(",", "").replace(" ", "").strip().lower()
    # Remove trailing period
    a = a.rstrip(".")
    g = str(gold).replace(",", "").replace(" ", "").strip().lower()
    return a == g

def score_exact_word(answer, gold):
    a = strip_answer(answer)
    a = a.splitlines()[0] if a.splitlines() else ""
    a = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", a.strip()).lower()
    # Take first word
    a = a.split()[0] if a.split() else ""
    a = re.sub(r"[^a-z]", "", a)
    g = re.sub(r"[^a-z]", "", str(gold).lower())
    return a == g and a != ""

def score_exact_letter(answer, gold):
    a = strip_answer(answer)
    a = a.strip().upper()
    # Find first A-Z letter
    m = re.search(r"[A-Z]", a)
    if not m:
        return False
    # Must be single-letter answer (allow punctuation after)
    # Pass if first letter equals gold and answer is short (<=10 chars stripped)
    first = m.group(0)
    g = str(gold).strip().upper()
    if first != g:
        return False
    # Reject long sentences that happen to start with letter: require answer core <=5 chars alnum
    core = re.sub(r"[^A-Z]", "", a)
    return len(core) == 1

def extract_json_obj(answer):
    t = strip_answer(answer)
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None, f"no braces in {t[:200]!r}"
    blob = t[start:end+1]
    try:
        obj = json.loads(blob)
        return obj, ""
    except Exception as e:
        return None, f"json parse fail: {e}; blob={blob[:300]!r}"

def score_json_exact(answer, gold):
    obj, err = extract_json_obj(answer)
    if obj is None:
        return False
    return obj == gold

def count_bullets(answer):
    t = strip_answer(answer)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    n = 0
    for ln in lines:
        if re.match(r"^([-*\u2022\u2023\u25E6]|\d+[\.)])\s+", ln):
            n += 1
    # If no newline bullets but single line with " - " separators, count conservatively as 1
    if n == 0 and t.strip():
        # Check inline dash bullets
        if re.search(r"(^|\n)\s*[-*]\s+\S+", t):
            n = 1
    return n, len(lines)

def score_pref_p01(answer, rubric=None):
    a = normalize_text(strip_answer(answer))
    # Must contain km/kilometre/kilometer
    has_metric = ("km" in a) or ("kilomet" in a)
    return has_metric

def score_pref_p02(answer, rubric=None):
    n, _ = count_bullets(answer)
    # No bullets detected but answer exists: if answer is short prose without bullets, it technically has 0 bullets (<=3) but rubric expects bullets?
    # For smoke: pass if n<=3. If n==0, check if answer looks like status without bullets: still pass if non-empty and truncated false (lenient).
    # To avoid false pass on empty, require non-empty.
    if not strip_answer(answer):
        return False
    return n <= 3

def score_pref_p03(answer, rubric=None):
    a = normalize_text(strip_answer(answer))
    asks = ("?" in a) or ("approv" in a) or ("confirm" in a) or ("may i" in a) or ("should i" in a) or ("ask" in a and "order" in a)
    claims_scheduled = ("schedul" in a and ("order" in a or "filter" in a)) or ("order placed" in a) or ("ordered" in a) or ("i have scheduled" in a) or ("i scheduled" in a)
    # Must ask and must not claim scheduled/ordered
    if claims_scheduled:
        # If it both asks and claims scheduled, fail (rubric: do not claim order was scheduled)
        return False
    return asks

def score_procedure(answer, rubric_list):
    a = normalize_text(strip_answer(answer))
    if not rubric_list:
        return False
    for phrase in rubric_list:
        p = normalize_text(phrase)
        # For "compare baseline and test": accept "compar" and "baseline" and "test" nearby? Keep simple substring, plus lenient variant
        if p in a:
            continue
        # Lenient: all words of phrase appear in answer
        words = [w for w in re.split(r"[^a-z0-9]+", p) if w]
        if all(w in a for w in words):
            continue
        return False
    return True

def score_item(item, answer_text, truncated):
    if truncated:
        return False, "truncated"
    scorer = item.get("scorer", "")
    gold = item.get("gold", "")
    kind = item.get("kind", "")
    try:
        if scorer == "fact_substring":
            return score_fact(answer_text, gold), ""
        elif scorer == "exact_number":
            return score_exact_number(answer_text, gold), ""
        elif scorer == "exact_word":
            return score_exact_word(answer_text, gold), ""
        elif scorer == "exact_letter":
            return score_exact_letter(answer_text, gold), ""
        elif scorer == "json_exact":
            return score_json_exact(answer_text, gold), ""
        elif scorer == "pref_p01":
            return score_pref_p01(answer_text), ""
        elif scorer == "pref_p02":
            return score_pref_p02(answer_text), ""
        elif scorer == "pref_p03":
            return score_pref_p03(answer_text), ""
        elif scorer == "procedure_all4":
            rubric = item.get("rubric", [])
            return score_procedure(answer_text, rubric), ""
        else:
            return False, f"unknown scorer {scorer}"
    except Exception as e:
        return False, f"scorer exc: {e}"

# ---------- file helpers (no torch) ----------

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def blake3_file(path):
    try:
        import blake3 as _b3
        h = _b3.blake3()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def file_hashes(path):
    p = Path(path)
    if not p.exists():
        return {"exists": False, "path": str(path)}
    return {
        "path": str(path),
        "exists": True,
        "bytes": p.stat().st_size,
        "sha256": sha256_file(path),
        "blake3": blake3_file(path),
    }

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def leak_check(train_rows_path, recall_path):
    train_data = load_json(train_rows_path)
    recall_data = load_json(recall_path)
    # Collect heldout/probe/rubric strings that must NOT appear in train
    forbidden = []
    for it in recall_data.get("items", []):
        for k in ["heldout_question", "probe", "rubric"]:
            v = it.get(k)
            if isinstance(v, str) and v:
                forbidden.append((it["id"], k, v))
            elif isinstance(v, list):
                for j, vv in enumerate(v):
                    if isinstance(vv, str) and vv:
                        forbidden.append((it["id"], f"{k}[{j}]", vv))
    # Collect train texts
    train_texts = []
    for row in train_data.get("rows", []):
        for msg in row.get("messages", []):
            c = msg.get("content", "")
            train_texts.append((row.get("id", "?"), msg.get("role", "?"), c))
    overlaps = []
    for fid, fkind, fstr in forbidden:
        for tid, trole, tstr in train_texts:
            if fstr and fstr in tstr:
                overlaps.append({"forbidden_id": fid, "forbidden_kind": fkind, "train_id": tid, "train_role": trole})
    return {"forbidden_count": len(forbidden), "train_text_count": len(train_texts), "overlap_count": len(overlaps), "overlaps": overlaps[:20]}

def validate_datasets(train_rows_path, recall_path, general_path):
    errors = []
    warnings = []
    # Load
    try:
        train_data = load_json(train_rows_path)
    except Exception as e:
        return {"ok": False, "errors": [f"train load fail: {e}"], "warnings": []}
    try:
        recall_data = load_json(recall_path)
    except Exception as e:
        return {"ok": False, "errors": [f"recall load fail: {e}"], "warnings": []}
    try:
        general_data = load_json(general_path)
    except Exception as e:
        return {"ok": False, "errors": [f"general load fail: {e}"], "warnings": []}
    # Counts
    rows = train_data.get("rows", [])
    if len(rows) != 34:
        errors.append(f"train rows {len(rows)} != 34")
    ritems = recall_data.get("items", [])
    facts = [x for x in ritems if x.get("kind") == "fact"]
    prefs = [x for x in ritems if x.get("kind") == "preference"]
    procs = [x for x in ritems if x.get("kind") == "procedure"]
    if len(facts) != 30:
        errors.append(f"recall facts {len(facts)} != 30")
    if len(prefs) != 3:
        errors.append(f"recall prefs {len(prefs)} != 3")
    if len(procs) != 1:
        errors.append(f"recall procs {len(procs)} != 1")
    if len(ritems) != 34:
        errors.append(f"recall total {len(ritems)} != 34")
    gitems = general_data.get("items", [])
    if not (8 <= len(gitems) <= 12):
        errors.append(f"general {len(gitems)} not in 8-12")
    # Scorers known
    known = {"fact_substring", "exact_number", "exact_word", "exact_letter", "json_exact", "pref_p01", "pref_p02", "pref_p03", "procedure_all4"}
    for it in ritems + gitems:
        if it.get("scorer") not in known:
            errors.append(f"unknown scorer {it.get('id')}: {it.get('scorer')}")
        if not it.get("prompt"):
            errors.append(f"missing prompt {it.get('id')}")
    # Leak
    try:
        leak = leak_check(train_rows_path, recall_path)
        if leak["overlap_count"] != 0:
            errors.append(f"leak overlap {leak['overlap_count']}: {leak['overlaps'][:3]}")
    except Exception as e:
        errors.append(f"leak check fail: {e}")
        leak = {"overlap_count": -1}
    # Scorer self-test: gold answers must pass (except prefs/procedure need crafted gold-like answers)
    # Facts: gold itself must pass
    for it in facts[:3]:
        ok, _ = score_item(it, it["gold"], False)
        if not ok:
            errors.append(f"fact self-test fail {it['id']}")
    # General: gold must pass
    for it in gitems:
        gold = it["gold"]
        if isinstance(gold, dict):
            ans = json.dumps(gold)
        else:
            ans = str(gold)
        ok, _ = score_item(it, ans, False)
        if not ok:
            errors.append(f"general self-test fail {it['id']} ans={ans!r}")
    # Truncation must fail
    for it in (facts[:1] + gitems[:1]):
        ok, _ = score_item(it, str(it.get("gold", "x")), True)
        if ok:
            errors.append(f"truncation self-test fail {it['id']}")
    # Wrong answers must fail (spot check)
    if facts:
        ok, _ = score_item(facts[0], "WRONG-VALUE-XYZ-123", False)
        if ok:
            errors.append("fact wrong-answer self-test fail (passed wrong)")
    if gitems:
        # G01 wrong number
        g01 = [g for g in gitems if g["id"] == "G01"]
        if g01:
            ok, _ = score_item(g01[0], "999", False)
            if ok:
                errors.append("general wrong-answer self-test fail G01")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings, "leak": leak}

# ---------- GPU helpers (torch imported lazily) ----------

def gpu_memory_stats():
    import torch
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else None,
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3) if torch.cuda.is_available() else None,
    }

def host_ram_gib():
    try:
        # ru_maxrss is KB on Linux, bytes on macOS; handle both by magnitude
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return round(v / 1024**3, 3)
        return round(v / 1024**2, 3)
    except Exception:
        return None

class NvmlSampler:
    def __init__(self):
        self.peak_mib = None
        self.stop = threading.Event()
        self.thread = None
    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    def _loop(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
                mib = int(float(out.split()[0]))
                if self.peak_mib is None or mib > self.peak_mib:
                    self.peak_mib = mib
            except Exception:
                pass
            self.stop.wait(1.0)
    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=15)
        if self.peak_mib is None:
            return None
        return round(self.peak_mib / 1024, 3)

def describe_lora_scope(model):
    import re as _re
    trainable = []
    frozen_lora = []
    for n, p in model.named_parameters():
        if "lora_" not in n:
            continue
        m = _re.search(r"layers\.(\d+)\.", n)
        layer = int(m.group(1)) if m else None
        # Extract module kind via regex (q/k/v/o/gate/up/down) before lora_
        kind = ""
        prefix = n.split("lora_")[0]
        for cand in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            if f".{cand}." in f".{prefix}." or prefix.endswith(f".{cand}.") or f".{cand}" in prefix:
                # Prefer exact: check cand appears as module name
                if cand in prefix:
                    kind = cand
                    break
        entry = {"name": n, "layer": layer, "kind": kind, "numel": int(p.numel()), "requires_grad": bool(p.requires_grad)}
        if p.requires_grad:
            trainable.append(entry)
        else:
            frozen_lora.append(entry)
    # Summarize
    train_layers = sorted(set(e["layer"] for e in trainable if e["layer"] is not None))
    train_kinds = sorted(set(e["kind"] for e in trainable if e["kind"]))
    train_params = sum(e["numel"] for e in trainable)
    return {
        "trainable_lora_params": train_params,
        "trainable_layers": train_layers,
        "trainable_kinds": train_kinds,
        "trainable_count": len(trainable),
        "frozen_lora_count": len(frozen_lora),
    }

def has_lora(model):
    try:
        for n, _ in model.named_parameters():
            if "lora_" in n:
                return True
        if hasattr(model, "peft_config") and getattr(model, "peft_config"):
            return True
    except Exception:
        pass
    return False

def load_base_fastmodel(base_model_dir, max_seq_length):
    from unsloth import FastModel
    import torch
    t0 = time.perf_counter()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    model, tokenizer = FastModel.from_pretrained(model_name=base_model_dir, max_seq_length=max_seq_length)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return model, tokenizer, round(dt, 1)

def attach_lora_final16(model):
    from unsloth import FastModel
    # Preferred: q/k/v/o only, r16 a32 dp0.05, vision frozen, grad ckpt unsloth
    # Unsloth API varies by version: try target_modules first, fallback to finetune flags only.
    tried = []
    last_err = None
    # Attempt 1: with target_modules
    try:
        tried.append("target_modules+qkv_o")
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=False,
            r=RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
            target_modules=TARGET_MODULES,
        )
        last_err = None
    except TypeError as e:
        last_err = f"attempt1 TypeError: {e}"
        # Attempt 2: without target_modules (rely on finetune flags, then filter)
        try:
            tried.append("finetune_flags_only")
            model = FastModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=False,
                r=RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=SEED,
            )
            last_err = None
        except Exception as e2:
            last_err = f"attempt2 fail: {e2}"
            raise RuntimeError(f"get_peft_model failed: {tried} last={last_err}")
    except Exception as e:
        last_err = f"attempt1 fail: {e}"
        raise RuntimeError(f"get_peft_model failed: {tried} last={last_err}")
    # Enforce final16 + q/k/v/o by freezing others (works even if Unsloth made all52 or extra gates)
    import re as _re
    before = describe_lora_scope(model)
    frozen_extra = 0
    kept = 0
    for n, p in model.named_parameters():
        if "lora_" not in n:
            continue
        m = _re.search(r"layers\.(\d+)\.", n)
        layer = int(m.group(1)) if m else None
        # Determine module kind: e.g., ...q_proj.lora_A... or ...gate_proj...
        kind = ""
        for cand in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "wqkv", "out_proj"]:
            if f".{cand}." in f".{n}." or f".{cand}.lora" in f".{n}." or cand in n:
                # Prefer exact match: check if cand appears before lora_
                if cand in n.split("lora_")[0]:
                    kind = cand
                    break
        # Keep only q/k/v/o in final16 text layers; freeze vision/embed/head and other modules
        keep = (kind in TARGET_MODULES) and (layer is not None and layer in FINAL16_LAYERS)
        if "vision" in n.lower():
            keep = False
        # If layer parse failed (e.g., non-layer LoRA like embed/head), freeze (we freeze embed/head)
        if layer is None:
            keep = False
        if keep:
            p.requires_grad = True
            kept += 1
        else:
            if p.requires_grad:
                frozen_extra += 1
            p.requires_grad = False
    after = describe_lora_scope(model)
    # Verify vision/embed/head frozen (no trainable outside language layers)
    vision_trainable = 0
    embed_head_trainable = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # LoRA params already counted; check base params that are trainable (should be none except LoRA)
        if "lora_" not in n:
            if "vision" in n:
                vision_trainable += 1
            if "embed_tokens" in n or "lm_head" in n:
                embed_head_trainable += 1
    return model, {"tried": tried, "before": before, "after": after, "frozen_extra": frozen_extra, "kept": kept,
                   "vision_base_trainable": vision_trainable, "embed_head_base_trainable": embed_head_trainable}

def build_train_texts(tokenizer, train_rows_path):
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    data = load_json(train_rows_path)
    rows = data.get("rows", [])
    texts = []
    for row in rows:
        convo = row.get("messages", [])
        # Use pinned template, minimal reasoning, fixed dates
        txt = tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False,
            reasoning_strength=REASONING_STRENGTH,
            knowledge_cutoff=KNOWLEDGE_CUTOFF,
            current_date=CURRENT_DATE,
        )
        # Strip single leading bos if tokenizer will add it (notebook pattern)
        try:
            bos = getattr(tokenizer, "bos_token", None) or getattr(getattr(tokenizer, "tokenizer", None), "bos_token", None)
            if bos and txt.startswith(bos):
                txt = txt[len(bos):]
                # Notebook uses removeprefix(bos_token); keep one bos via tokenizer
                # Actually notebook does .removeprefix(bos_token) to avoid double bos. Do same.
        except Exception:
            pass
        texts.append(txt)
    return texts, {"rows": len(rows)}

def build_eval_prefix(tokenizer, user_prompt):
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    # Pinned template + forced to=user channel for budget (documented deviation)
    messages = [{"role": "user", "content": user_prompt}]
    base = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        reasoning_strength=REASONING_STRENGTH,
        knowledge_cutoff=KNOWLEDGE_CUTOFF,
        current_date=CURRENT_DATE,
    )
    # Append forced answer channel (do not invent other delimiters; this is the template exact assistant prefix)
    return base + FORCED_ANSWER_PREFIX

def generate_one(model, tokenizer, user_prompt, max_new_tokens=MAX_NEW_TOKENS):
    import torch
    model.eval()
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    prefix = build_eval_prefix(tokenizer, user_prompt)
    # Tokenize prefix without adding extra bos (prefix already has it)
    enc = text_tok(prefix, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to("cuda" if torch.cuda.is_available() else "cpu")
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(input_ids.device)
    # Pad/eos handling
    pad_id = getattr(text_tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = PAD_ID
    # Ensure model config use_cache for eval
    try:
        model.config.use_cache = True
    except Exception:
        pass
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=EOS_IDS,
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    # Slice new tokens
    new_ids = out[0][input_ids.shape[1]:].tolist()
    # Decode with special tokens visible for truncation detection
    raw_full = text_tok.decode(new_ids, skip_special_tokens=False)
    # Truncation: hit cap without EOS marker and without eos id
    hit_cap = len(new_ids) >= max_new_tokens
    has_eos_id = any(t in EOS_IDS for t in new_ids)
    has_eot_marker = ("<|eot|>" in raw_full) or ("<|eom|>" in raw_full) or ("<|end_of_text|>" in raw_full)
    truncated = bool(hit_cap and (not has_eos_id) and (not has_eot_marker))
    answer = strip_answer(raw_full)
    return {
        "prefix_chars": len(prefix),
        "raw_output": raw_full[:2000],
        "answer_text": answer[:2000],
        "truncated": truncated,
        "tokens_generated": len(new_ids),
        "latency_s": round(dt, 3),
        "hit_cap": hit_cap,
        "has_eos_id": has_eos_id,
    }

def evaluate_set(model, tokenizer, items, label):
    results = []
    n_pass = 0
    n_trunc = 0
    total_lat = 0.0
    for it in items:
        g = generate_one(model, tokenizer, it["prompt"], max_new_tokens=MAX_NEW_TOKENS)
        ok, reason = score_item(it, g["answer_text"], g["truncated"])
        if ok:
            n_pass += 1
        if g["truncated"]:
            n_trunc += 1
        total_lat += g["latency_s"]
        results.append({
            "id": it["id"],
            "kind": it.get("kind", ""),
            "scorer": it.get("scorer", ""),
            "prompt": it["prompt"],
            "gold": it.get("gold", ""),
            "raw_output": g["raw_output"],
            "answer_text": g["answer_text"],
            "pass": bool(ok),
            "truncated": bool(g["truncated"]),
            "tokens_generated": g["tokens_generated"],
            "latency_s": g["latency_s"],
            "score_note": reason,
        })
    return {
        "label": label,
        "n": len(items),
        "passed": n_pass,
        "rate": round(n_pass / len(items), 4) if items else None,
        "truncated": n_trunc,
        "trunc_rate": round(n_trunc / len(items), 4) if items else None,
        "total_latency_s": round(total_lat, 1),
        "items": results,
    }

def run_smoke(model=None, tokenizer=None, train_rows_path="train_rows.json", recall_path="recall_probes.json",
              general_path="general_slice.json", base_model_dir=None, adapter_out="./smoke-adapter",
              results_out="./smoke-results.json", max_seq_length=MAX_SEQ_LENGTH, clean_reload=True,
              seed=SEED, allow_contaminated_base=False):
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t_all0 = time.perf_counter()
    # Validate first (CPU, no torch needed for this part except version capture later)
    val = validate_datasets(train_rows_path, recall_path, general_path)
    if not val["ok"]:
        out = {"status": "VALIDATION-FAIL", "started_utc": started, "validation": val,
               "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
               "smoke_version": SMOKE_VERSION}
        Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return out
    # Unsloth first to apply patches, before torch/transformers/peft/trl (only on a GPU run)
    import unsloth  # noqa: F401 (patches first)
    import torch
    import transformers
    import peft
    import trl
    import datasets as _ds
    try:
        import unsloth as _u
        unsloth_ver = getattr(_u, "__version__", "unknown")
    except Exception:
        unsloth_ver = "import-fail"
    print(f"torch={torch.__version__} cuda_build={torch.version.cuda} cuda_avail={torch.cuda.is_available()}", flush=True)
    print(f"transformers={transformers.__version__} peft={peft.__version__} trl={trl.__version__} unsloth={unsloth_ver}", flush=True)
    if not torch.cuda.is_available():
        out = {"status": "NO-CUDA", "started_utc": started, "validation": val,
               "error": "torch.cuda.is_available() is False; smoke requires a CUDA GPU",
               "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
               "smoke_version": SMOKE_VERSION}
        Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return out
    # Seeds
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    # GPU info
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    total_gib = round(props.total_memory / 1024**3, 2)
    print(f"gpu={gpu_name} total_gib={total_gib}", flush=True)
    # Base load or reuse
    clean_base_reload = False
    base_contaminated = False
    load_s = None
    if model is None or tokenizer is None:
        if not base_model_dir:
            out = {"status": "NO-BASE", "started_utc": started, "validation": val, "error": "no loaded model and no base_model_dir",
                   "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
                   "smoke_version": SMOKE_VERSION}
            Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(json.dumps(out, indent=2))
            return out
        try:
            model, tokenizer, load_s = load_base_fastmodel(base_model_dir, max_seq_length)
            clean_base_reload = True
            print(f"base load_s={load_s} {gpu_memory_stats()}", flush=True)
        except Exception as e:
            err = f"base load fail: {e}\n" + traceback.format_exc()[-3000:]
            print(err, flush=True)
            out = {"status": "LOAD-FAIL", "started_utc": started, "validation": val, "error": err,
                   "gpu": gpu_name, "total_vram_gib": total_gib,
                   "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
                   "smoke_version": SMOKE_VERSION}
            Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
            return out
    else:
        print("using already-loaded model/tokenizer (saves setup)", flush=True)
        clean_base_reload = False
        if has_lora(model):
            base_contaminated = True
            msg = "provided model already has LoRA; before-score would not be clean base"
            print(f"WARN: {msg}", flush=True)
            if not allow_contaminated_base:
                out = {"status": "CONTAMINATED-BASE", "started_utc": started, "validation": val,
                       "error": msg + " (pass allow_contaminated_base=True to override, but before-score invalid)",
                       "gpu": gpu_name, "total_vram_gib": total_gib,
                       "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
                       "smoke_version": SMOKE_VERSION}
                Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
                return out
    # Pin chat template from the staged base (tokenizer_config.json carries none; unsloth default would be wrong)
    try:
        _text_tok = getattr(tokenizer, "tokenizer", tokenizer)
        if base_model_dir:
            _jt = Path(base_model_dir) / "chat_template.jinja"
            if _jt.exists():
                _pinned = _jt.read_text(encoding="utf-8")
                _text_tok.chat_template = _pinned
                try:
                    tokenizer.chat_template = _pinned
                except Exception:
                    pass
                _probe = _text_tok.apply_chat_template(
                    [{"role": "user", "content": "Probe."}, {"role": "assistant", "content": "Copy."}],
                    tokenize=False, reasoning_strength=REASONING_STRENGTH,
                    knowledge_cutoff=KNOWLEDGE_CUTOFF, current_date=CURRENT_DATE)
                assert RESPONSE_PART in _probe and INSTRUCTION_PART in _probe, "pinned template render missing delimiters"
                print("chat template pinned from chat_template.jinja; delimiters validated", flush=True)
            else:
                print("WARN: chat_template.jinja missing in base_model_dir; using loaded default", flush=True)
    except Exception as e:
        print(f"template pin fail (continuing with loaded default): {e}", flush=True)
    # Template hash
    try:
        chat_tmpl = getattr(tokenizer, "chat_template", None) or getattr(getattr(tokenizer, "tokenizer", None), "chat_template", None) or ""
        tmpl_sha = sha256_bytes(chat_tmpl.encode("utf-8")) if chat_tmpl else None
    except Exception:
        tmpl_sha = None
        chat_tmpl = ""
    # Load eval items
    recall_data = load_json(recall_path)
    general_data = load_json(general_path)
    recall_items = recall_data.get("items", [])
    general_items = general_data.get("items", [])
    recall_facts = [x for x in recall_items if x.get("kind") == "fact"]
    recall_prefs = [x for x in recall_items if x.get("kind") == "preference"]
    recall_proc = [x for x in recall_items if x.get("kind") == "procedure"]
    # BEFORE eval (base)
    print("BEFORE eval starting (base/no-notes equivalent)", flush=True)
    torch.cuda.reset_peak_memory_stats()
    sampler = NvmlSampler()
    sampler.start()
    t_before0 = time.perf_counter()
    try:
        before_facts = evaluate_set(model, tokenizer, recall_facts, "before_facts")
        before_prefs = evaluate_set(model, tokenizer, recall_prefs, "before_prefs")
        before_proc = evaluate_set(model, tokenizer, recall_proc, "before_proc")
        before_general = evaluate_set(model, tokenizer, general_items, "before_general")
    except Exception as e:
        sampler.finish()
        err = f"before-eval fail: {e}\n" + traceback.format_exc()[-3000:]
        print(err, flush=True)
        out = {"status": "EVAL-BEFORE-FAIL", "started_utc": started, "validation": val, "error": err,
               "gpu": gpu_name, "total_vram_gib": total_gib, "clean_base_reload": clean_base_reload,
               "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
               "smoke_version": SMOKE_VERSION}
        Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    torch.cuda.synchronize()
    before_s = round(time.perf_counter() - t_before0, 1)
    before_peak_alloc = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    before_peak_res = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    print(f"BEFORE done in {before_s}s facts {before_facts['passed']}/{before_facts['n']} general {before_general['passed']}/{before_general['n']}", flush=True)
    # Attach LoRA final16
    print("attaching LoRA final16 q/k/v/o r16a32 dp0.05", flush=True)
    try:
        model, lora_info = attach_lora_final16(model)
        print(f"lora after: {json.dumps(lora_info['after'])} frozen_extra={lora_info['frozen_extra']}", flush=True)
    except Exception as e:
        sampler.finish()
        err = f"lora attach fail: {e}\n" + traceback.format_exc()[-3000:]
        print(err, flush=True)
        out = {"status": "LORA-FAIL", "started_utc": started, "validation": val, "error": err,
               "before": {"facts": before_facts, "prefs": before_prefs, "proc": before_proc, "general": before_general},
               "gpu": gpu_name, "total_vram_gib": total_gib, "clean_base_reload": clean_base_reload,
               "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
               "smoke_version": SMOKE_VERSION}
        Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    # Build train texts + dataset
    try:
        train_texts, train_meta = build_train_texts(tokenizer, train_rows_path)
        print(f"train texts {train_meta} sample_len={len(train_texts[0][:200])}", flush=True)
        from datasets import Dataset
        ds = Dataset.from_dict({"text": train_texts})
    except Exception as e:
        sampler.finish()
        err = f"train data build fail: {e}\n" + traceback.format_exc()[-3000:]
        print(err, flush=True)
        out = {"status": "DATA-FAIL", "started_utc": started, "validation": val, "error": err,
               "before": {"facts": before_facts, "prefs": before_prefs, "proc": before_proc, "general": before_general},
               "lora": lora_info, "gpu": gpu_name, "total_vram_gib": total_gib,
               "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
               "smoke_version": SMOKE_VERSION}
        Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    # Trainer (response-only, nll, GA4, LR1e-4, 1 epoch)
    try:
        from trl import SFTTrainer, SFTConfig
        from transformers import TrainerCallback, PreTrainedTokenizerBase, ProcessorMixin
        from unsloth.chat_templates import train_on_responses_only
        _pc = tokenizer if isinstance(tokenizer, (PreTrainedTokenizerBase, ProcessorMixin)) else getattr(tokenizer, "tokenizer", tokenizer)
        print(f"processing_class resolved: {type(_pc).__name__}", flush=True)
        trainer = SFTTrainer(
            model=model,
            processing_class=_pc,
            train_dataset=ds,
            eval_dataset=None,
            args=SFTConfig(
                dataset_text_field="text",
                max_length=max_seq_length,
                packing=False,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM,
                warmup_steps=WARMUP_STEPS,
                num_train_epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                logging_steps=1,
                optim=OPTIM,
                weight_decay=0.0,
                lr_scheduler_type=SCHEDULER,
                seed=seed,
                output_dir=str(Path(adapter_out).parent / "smoke-trainer-out"),
                report_to="none",
            ),
        )
        trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)
        class KeepEmbeddingOffloaded(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                try:
                    et = kwargs["model"].get_input_embeddings()
                    hook = getattr(et, "_hf_hook", None)
                    if getattr(hook, "execution_device", None) is not None:
                        print("EMBED: accelerate-owned, staying on GPU", flush=True)
                        return control
                    if et.weight.device.type != "cpu":
                        et.to("cpu")
                        torch.cuda.empty_cache()
                        print("EMBED: offloaded to CPU", flush=True)
                except Exception as e:
                    print(f"EMBED callback fail: {e}", flush=True)
                return control
        trainer.add_callback(KeepEmbeddingOffloaded())
        # Log masking: input vs loss positions on first row
        try:
            _tok = getattr(tokenizer, "tokenizer", tokenizer)
            _lab = trainer.train_dataset[0]["labels"]
            _inp = trainer.train_dataset[0]["input_ids"]
            _pad = getattr(_tok, "pad_token_id", None)
            n_inp = sum(1 for x in _inp if x != _pad)
            n_loss = sum(1 for x in _lab if x != -100)
            print(f"positions/row0: input(non-pad)={n_inp} loss-bearing={n_loss}", flush=True)
        except Exception as e:
            print(f"mask log fail: {e}", flush=True)
            n_inp = None
            n_loss = None
    except Exception as e:
        sampler.finish()
        err = f"trainer build fail: {e}\n" + traceback.format_exc()[-3000:]
        print(err, flush=True)
        out = {"status": "TRAINER-FAIL", "started_utc": started, "validation": val, "error": err,
               "before": {"facts": before_facts, "prefs": before_prefs, "proc": before_proc, "general": before_general},
               "lora": lora_info, "gpu": gpu_name, "total_vram_gib": total_gib,
               "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
               "smoke_version": SMOKE_VERSION}
        Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    # Train (single short pass)
    print("TRAIN starting (1 epoch, 34 rows, GA4 => ~9 optimizer steps)", flush=True)
    try:
        model.config.use_cache = False
    except Exception:
        pass
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_train0 = time.perf_counter()
    train_err = None
    stats = None
    try:
        stats = trainer.train()
    except torch.cuda.OutOfMemoryError:
        train_err = "OOM:\n" + traceback.format_exc()[-3000:]
    except Exception:
        train_err = "EXC:\n" + traceback.format_exc()[-3000:]
    torch.cuda.synchronize()
    train_s = round(time.perf_counter() - t_train0, 1)
    _train_runtime_s = round(stats.metrics.get("train_runtime", train_s), 1) if stats is not None and hasattr(stats, "metrics") else None
    train_peak_alloc = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    train_peak_res = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    print(f"TRAIN done in {train_s}s err={bool(train_err)} peak_alloc={train_peak_alloc} peak_res={train_peak_res}", flush=True)
    if train_err is not None:
        nvml_peak = sampler.finish()
        out = {"status": "TRAIN-FAIL", "started_utc": started, "validation": val, "error": train_err,
               "before": {"facts": before_facts, "prefs": before_prefs, "proc": before_proc, "general": before_general},
               "lora": lora_info, "gpu": gpu_name, "total_vram_gib": total_gib,
               "train_s": train_s, "train_peak_alloc_gib": train_peak_alloc, "train_peak_reserved_gib": train_peak_res,
               "nvml_peak_gib": nvml_peak, "clean_base_reload": clean_base_reload,
               "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
               "smoke_version": SMOKE_VERSION}
        Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps({k: out[k] for k in ["status", "error"] if k in out}, indent=2))
        return out
    # Save adapter (timed)
    print(f"saving adapter to {adapter_out}", flush=True)
    Path(adapter_out).mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    t_save0 = time.perf_counter()
    try:
        model.save_pretrained(adapter_out)
        getattr(tokenizer, "tokenizer", tokenizer).save_pretrained(adapter_out)
    except Exception as e:
        sampler.finish()
        err = f"save fail: {e}\n" + traceback.format_exc()[-2000:]
        print(err, flush=True)
        out = {"status": "SAVE-FAIL", "started_utc": started, "validation": val, "error": err,
               "before": {"facts": before_facts, "prefs": before_prefs, "proc": before_proc, "general": before_general},
               "lora": lora_info, "gpu": gpu_name, "total_vram_gib": total_gib,
               "train_s": train_s, "clean_base_reload": clean_base_reload,
               "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
               "smoke_version": SMOKE_VERSION}
        Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    torch.cuda.synchronize()
    save_s = round(time.perf_counter() - t_save0, 1)
    # Adapter hashes
    adapter_files = {}
    for name in ["adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"]:
        p = Path(adapter_out) / name
        if p.exists():
            adapter_files[name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    print(f"adapter saved in {save_s}s files={list(adapter_files.keys())}", flush=True)
    # AFTER eval: runtime adapter reload on SAME frozen base (NOT cold base; avoids 2nd base OOM)
    # Parent fix: del model alone does not free GPU (trainer/optimizer hold refs). Same-base reload avoids 2nd base load.
    clean_adapter_reload = False
    reload_s = None
    reload_note = ""
    eval_model = model
    eval_tok = tokenizer
    try:
        del trainer
    except Exception:
        pass
    try:
        del stats
    except Exception:
        pass
    try:
        del ds
    except Exception:
        pass
    try:
        del train_texts
    except Exception:
        pass
    import gc
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    if clean_reload:
        print("runtime adapter reload for adapted score (same frozen base, NOT cold base)", flush=True)
        try:
            t_re0 = time.perf_counter()
            model.load_adapter(adapter_out, adapter_name="smoke_reloaded")
            model.set_adapter("smoke_reloaded")
            model.eval()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            reload_s = round(time.perf_counter() - t_re0, 1)
            eval_model = model
            eval_tok = tokenizer
            clean_adapter_reload = True
            reload_note = "load_adapter+set_adapter on same frozen base (runtime reload, NOT cold base timing)"
            print(f"runtime reload ok in {reload_s}s via {reload_note}", flush=True)
        except Exception as e:
            sampler.finish()
            err = f"runtime adapter reload fail: {e}\n" + traceback.format_exc()[-3000:]
            print(err, flush=True)
            out = {"status": "RELOAD-FAIL", "started_utc": started, "validation": val, "error": err,
                   "before": {"facts": before_facts, "prefs": before_prefs, "proc": before_proc, "general": before_general},
                   "lora": lora_info, "gpu": gpu_name, "total_vram_gib": total_gib,
                   "train_s": train_s, "save_s": save_s, "adapter_files": adapter_files,
                   "clean_base_reload": clean_base_reload, "clean_adapter_reload": False,
                   "reload_note": reload_note,
                   "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
                   "smoke_version": SMOKE_VERSION}
            Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
            return out
    else:
        reload_note = "in-place adapted score (no reload; clean_reload=False, uses in-memory adapter)"
        print(reload_note, flush=True)
        clean_adapter_reload = False
    # AFTER eval
    print("AFTER eval starting (adapted/no-notes)", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t_after0 = time.perf_counter()
    try:
        after_facts = evaluate_set(eval_model, eval_tok, recall_facts, "after_facts")
        after_prefs = evaluate_set(eval_model, eval_tok, recall_prefs, "after_prefs")
        after_proc = evaluate_set(eval_model, eval_tok, recall_proc, "after_proc")
        after_general = evaluate_set(eval_model, eval_tok, general_items, "after_general")
    except Exception as e:
        sampler.finish()
        err = f"after-eval fail: {e}\n" + traceback.format_exc()[-3000:]
        print(err, flush=True)
        out = {"status": "EVAL-AFTER-FAIL", "started_utc": started, "validation": val, "error": err,
               "before": {"facts": before_facts, "prefs": before_prefs, "proc": before_proc, "general": before_general},
               "lora": lora_info, "gpu": gpu_name, "total_vram_gib": total_gib,
               "train_s": train_s, "save_s": save_s, "adapter_files": adapter_files,
               "clean_base_reload": clean_base_reload, "clean_adapter_reload": clean_adapter_reload,
               "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
               "smoke_version": SMOKE_VERSION}
        Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    torch.cuda.synchronize()
    after_s = round(time.perf_counter() - t_after0, 1)
    after_peak_alloc = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    after_peak_res = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    nvml_peak = sampler.finish()
    print(f"AFTER done in {after_s}s facts {after_facts['passed']}/{after_facts['n']} general {after_general['passed']}/{after_general['n']}", flush=True)
    # Paired wins/losses
    def paired(before, after):
        b = {x["id"]: x["pass"] for x in before["items"]}
        a = {x["id"]: x["pass"] for x in after["items"]}
        wins = sum(1 for k in b if (not b[k]) and a.get(k, False))
        losses = sum(1 for k in b if b[k] and (not a.get(k, True)))
        same_pass = sum(1 for k in b if b[k] and a.get(k, False))
        same_fail = sum(1 for k in b if (not b[k]) and (not a.get(k, True)))
        return {"wins": wins, "losses": losses, "same_pass": same_pass, "same_fail": same_fail}
    total_s = round(time.perf_counter() - t_all0, 1)
    out = {
        "status": "OK",
        "started_utc": started,
        "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke_version": SMOKE_VERSION,
        "validation": val,
        "recipe": {
            "rank": RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "lr": LEARNING_RATE,
            "batch": BATCH_SIZE, "gradacc": GRAD_ACCUM, "epochs": EPOCHS, "warmup": WARMUP_STEPS,
            "optim": OPTIM, "scheduler": SCHEDULER, "loss": LOSS_TYPE, "max_seq_length": max_seq_length,
            "targets": TARGET_MODULES, "layers": FINAL16_LAYERS, "num_text_layers": NUM_TEXT_LAYERS,
            "response_only": RESPONSE_PART, "grad_checkpoint": "unsloth", "seed": seed,
            "reasoning_strength": REASONING_STRENGTH, "max_new_tokens": MAX_NEW_TOKENS,
            "greedy": True, "answer_channel": "forced to=user",
        },
        "versions": {"torch": torch.__version__, "cuda_build": torch.version.cuda,
                     "transformers": transformers.__version__, "peft": peft.__version__, "trl": trl.__version__, "unsloth": unsloth_ver},
        "gpu": gpu_name, "total_vram_gib": total_gib,
        "lora": lora_info,
        "template_sha256": tmpl_sha,
        "base_model_dir": base_model_dir,
        "clean_base_reload": clean_base_reload,
        "base_contaminated": base_contaminated,
        "clean_adapter_reload": clean_adapter_reload,
        "reload_note": reload_note,
        "timings": {"load_s": load_s, "before_s": before_s, "train_s": train_s, "save_s": save_s,
                    "reload_s": reload_s, "after_s": after_s, "total_s": total_s,
                    "train_runtime_s": _train_runtime_s},
        "memory": {"before_peak_alloc_gib": before_peak_alloc, "before_peak_reserved_gib": before_peak_res,
                   "train_peak_alloc_gib": train_peak_alloc, "train_peak_reserved_gib": train_peak_res,
                   "after_peak_alloc_gib": after_peak_alloc, "after_peak_reserved_gib": after_peak_res,
                   "nvml_peak_gib": nvml_peak, "host_peak_ram_gib": host_ram_gib()},
        "before": {"facts": before_facts, "prefs": before_prefs, "proc": before_proc, "general": before_general},
        "after": {"facts": after_facts, "prefs": after_prefs, "proc": after_proc, "general": after_general},
        "paired": {"facts": paired(before_facts, after_facts), "general": paired(before_general, after_general)},
        "adapter_files": adapter_files,
        "hashes": {k: file_hashes(p) for k, p in [("train", train_rows_path), ("recall", recall_path), ("general", general_path)]},
        "limitations": [
            "Smoke only: 30x1 recall (not 90), 10-item synthetic general (not 200), 2 arms (no oracle-notes C/D).",
            "Forced to=user answer channel with max 96 tokens; full ATEM reasoning not scored in smoke.",
            "No replay/general anchors in smoke train (34 rows only); REPORT mix is 50/20/30.",
            "Single short pass (~9 optimizer steps); not a dawn-scale run.",
            f"clean_base_reload={clean_base_reload}; clean_adapter_reload={clean_adapter_reload}.",
            "If clean_base_reload is False, before-score is not from fresh base; report as limitation.",
        ],
    }
    Path(results_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["status", "paired", "timings", "memory"] if k in out}, indent=2))
    print("SMOKE-DONE", flush=True)
    return out

def run_smoke_with_loaded(model, tokenizer, **kwargs):
    return run_smoke(model=model, tokenizer=tokenizer, **kwargs)

def main():
    ap = argparse.ArgumentParser(description="QLoRA day-one smoke (synthetic only)")
    ap.add_argument("--train-rows", default="train_rows.json")
    ap.add_argument("--recall-probes", default="recall_probes.json")
    ap.add_argument("--general-slice", default="general_slice.json")
    ap.add_argument("--model-dir", default=None, help="base 4-bit dir for fresh load")
    ap.add_argument("--adapter-out", default="./smoke-adapter")
    ap.add_argument("--results-out", default="./smoke-results.json")
    ap.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--clean-reload", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow-contaminated-base", action="store_true", default=False)
    ap.add_argument("--validate-only", action="store_true", default=False)
    args = ap.parse_args()
    if args.validate_only:
        val = validate_datasets(args.train_rows, args.recall_probes, args.general_slice)
        hashes = {k: file_hashes(p) for k, p in [("train", args.train_rows), ("recall", args.recall_probes), ("general", args.general_slice)]}
        print(json.dumps({"validation": val, "hashes": hashes, "smoke_version": SMOKE_VERSION}, indent=2))
        if val["ok"]:
            print("VALIDATE-OK", flush=True)
            sys.exit(0)
        else:
            print("VALIDATE-FAIL", flush=True)
            sys.exit(3)
    out = run_smoke(
        train_rows_path=args.train_rows,
        recall_path=args.recall_probes,
        general_path=args.general_slice,
        base_model_dir=args.model_dir,
        adapter_out=args.adapter_out,
        results_out=args.results_out,
        max_seq_length=args.max_seq_length,
        clean_reload=args.clean_reload,
        seed=args.seed,
        allow_contaminated_base=args.allow_contaminated_base,
    )
    if out.get("status") == "OK":
        sys.exit(0)
    elif out.get("status") in ("VALIDATION-FAIL",):
        sys.exit(3)
    else:
        sys.exit(2)

if __name__ == "__main__":
    main()

SMOKE_FORK_NOTES = """F1: removed invalid SFTConfig(loss_type=..) kwarg (no such field in installed TRL 0.22.2; proven TypeError otherwise). Loss remains standard causal NLL + response-only masking.
F2: processing_class resolves to wrapper only if HF tokenizer/processor, else unwrapped .tokenizer (TRL TypeError otherwise).
F3: build_train_texts/build_eval_prefix use unwrapped HF tokenizer for apply_chat_template (no-op if already raw).
F4: pin chat_template from base chat_template.jinja + validate delimiters render (tokenizer_config has no template; unsloth default would mismatch RESPONSE_PART masking). Warn-and-continue if missing.
F5: tokenizer.save_pretrained via unwrapped HF tokenizer (no-op if already raw).\nOriginal eval-owned sha256: 31ef859d2b5669290049f144cf2a37dffcda2babaa976063cfa61eb77d902f98"""
