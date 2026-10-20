"""Build the DRAFTER-TUNE prompt sets. Split by life / source BEFORE any generation.
Outputs prompts_train.jsonl, prompts_eval.jsonl and split.json in OUT.
Serving mode: direct answer (prompt ends with '<|start|>assistant to=user<|message|>'), as in the C pilot."""
import glob, json, random, re, sys, os
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = sys.argv[1]
OUT = os.path.join(ROOT, "data")
SEED = 20261014
rng = random.Random(SEED)
tok = Tokenizer.from_file(os.path.join(OUT, "tok/tokenizer.json"))
env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)
def _raise(m): raise ValueError(m)
env.globals["raise_exception"] = _raise
tmpl = env.from_string(open(os.path.join(OUT, "tok/chat_template.jinja")).read())
GEN = " to=user<|message|>"   # force the direct answer after the template's '<|start|>assistant'

def render(messages, current_date):
    s = tmpl.render(messages=messages, bos_token="<|begin_of_text|>", add_generation_prompt=True, current_date=current_date)
    assert s.endswith("<|start|>assistant")
    return s + GEN

lives = [json.loads(l) for f in sorted(glob.glob(os.path.join(OUT, "lives/lives-*.jsonl"))) for l in open(f)]
uc = json.load(open(os.path.join(OUT, "public/ultrachat_slice.json")))

# ---- split by life, stratified by language: 40 held out (20 used for eval prompts, 20 spare) ----
by_lang = {}
for l in lives: by_lang.setdefault(l["lang"], []).append(l["life_id"])
held = {}
for lang, n_eval, n_spare in (("en", 14, 14), ("ja", 3, 3), ("uk", 3, 3)):
    ids = sorted(by_lang[lang]); rng.shuffle(ids)
    held[lang] = (ids[:n_eval], ids[n_eval:n_eval + n_spare])
eval_lives = [i for v in held.values() for i in v[0]]
spare_lives = [i for v in held.values() for i in v[1]]
heldout = set(eval_lives) | set(spare_lives)
train_lives = [l["life_id"] for l in lives if l["life_id"] not in heldout]
L = {l["life_id"]: l for l in lives}

LINE_ID = re.compile(r"^d\d+s\d+t\d+:L\d+\s*")
def fold(turns):
    """turns -> chat messages; tool output is attached to the user message it answers."""
    msgs = []
    for t in turns:
        if t["role"] == "tool":
            body = "\n".join(LINE_ID.sub("", x) for x in t.get("lines", []))
            att = f"\n\n[{t.get('tool','attachment')}]\n{body}"
            if msgs and msgs[-1]["role"] == "user": msgs[-1]["content"] += att
            else: msgs.append({"role": "user", "content": att.strip()})
        elif t["role"] == "user":
            if msgs and msgs[-1]["role"] == "user": msgs[-1]["content"] += "\n\n" + t["text"]
            else: msgs.append({"role": "user", "content": t["text"]})
        else:
            msgs.append({"role": "assistant", "content": t["text"]})
    return msgs

def notes(life, day):
    facts = [f for f in life["ledger"] if f["valid_from"] < day and (f["valid_to"] is None or f["valid_to"] >= day)]
    facts = facts[-40:]
    return "\n".join(f"- {f['subject']} | {f['predicate'].replace('_', ' ')}: {f['value']}" for f in facts)

def system_for(life, day, date, weekday):
    p = life["persona"]; n = notes(life, day)
    s = (f"You are {p['name']}'s personal assistant. You remember what {p['name']} told you in earlier conversations "
         f"and use it when it helps. Reply in the user's language. Today is {weekday}, {date}.")
    if n: s += "\n\nMemory notes from earlier days:\n" + n
    return s

def chat_prompt(life, r):
    days = [d for d in life["days"] if d["day"] > 1] or life["days"]
    d = r.choice(days); s = r.choice(d["sessions"]); turns = s["turns"]
    users = [i for i, t in enumerate(turns) if t["role"] == "user" and i > 0 and (i == 0 or turns[i-1]["role"] != "user")]
    users = [i for i in users if any(t["role"] == "assistant" for t in turns[i+1:])] or [0]
    k = r.choice(users)
    j = k + 1
    while j < len(turns) and turns[j]["role"] in ("tool", "user"): j += 1   # include the user's tool attachments
    msgs = fold(turns[:j])[-17:]
    while msgs and msgs[0]["role"] != "user": msgs = msgs[1:]
    msgs = [{"role": "system", "content": system_for(life, d["day"], d["date"], d["weekday"])}] + msgs
    return msgs, d["date"], f"d{d['day']}{s['sid']}@{turns[k]['id']}"

def recall_prompt(life, r):
    qs = (life.get("questions_ja") or life["questions"]) if life["lang"] == "ja" else life["questions"]
    q = r.choice(qs); last = life["days"][-1]
    msgs = [{"role": "system", "content": system_for(life, last["day"] + 1, last["date"], "the next day after " + last["weekday"])},
            {"role": "user", "content": q["question"]}]
    return msgs, last["date"], q["qid"]

def public_prompt(row):
    return [{"role": "user", "content": row["prompt"]}], "2026-10-14", row["prompt_id"]

def make(pid, split, src, msgs, date, ref, life=None):
    text = render(msgs, date)
    ids = tok.encode(text, add_special_tokens=False).ids
    return {"id": pid, "split": split, "src": src, "life_id": life and life["life_id"], "lang": life["lang"] if life else "en",
            "ref": ref, "messages": msgs, "prompt": text, "prompt_ids": ids, "n_prompt": len(ids)}

train, ev = [], []
for lid in train_lives:
    life = L[lid]; r = random.Random(f"{SEED}-{lid}")
    seen = set()
    for c in range(2):
        for _ in range(5):
            msgs, date, ref = chat_prompt(life, r)
            if ref not in seen: break
        seen.add(ref)
        train.append(make(f"tr-{lid}-c{c}", "train", "life-chat", msgs, date, ref, life))
    msgs, date, ref = recall_prompt(life, r)
    train.append(make(f"tr-{lid}-r0", "train", "life-recall", msgs, date, ref, life))
for i, row in enumerate(uc["train_sft"][:500]):
    msgs, date, ref = public_prompt(row)
    train.append(make(f"tr-uc-{i:03d}", "train", "public-chat", msgs, date, ref))

er = random.Random(f"{SEED}-eval")
lv = list(eval_lives); er.shuffle(lv)
for lid in lv[:15]:
    life = L[lid]; msgs, date, ref = chat_prompt(life, random.Random(f"{SEED}-{lid}-eval"))
    ev.append(make(f"ev-{lid}-c", "eval", "life-chat", msgs, date, ref, life))
for lid in lv[15:20]:
    life = L[lid]; msgs, date, ref = recall_prompt(life, random.Random(f"{SEED}-{lid}-eval"))
    ev.append(make(f"ev-{lid}-r", "eval", "life-recall", msgs, date, ref, life))
for i, row in enumerate(uc["test_sft"][:5]):
    msgs, date, ref = public_prompt(row)
    ev.append(make(f"ev-uc-{i}", "eval", "public-chat", msgs, date, ref))
# interleave eval sources so the 5 losslessness prompts (first 5) cover all three
order = [ev[0], ev[15], ev[20], ev[1], ev[16]] + [e for e in ev if e["id"] not in {ev[0]["id"], ev[15]["id"], ev[20]["id"], ev[1]["id"], ev[16]["id"]}]
ev = order

rng2 = random.Random(SEED + 1); rng2.shuffle(train)   # any prefix of the generation queue is a representative subset
assert not ({e["life_id"] for e in ev if e["life_id"]} & {t["life_id"] for t in train if t["life_id"]})
assert not ({e["ref"] for e in ev if e["src"] == "public-chat"} & {t["ref"] for t in train if t["src"] == "public-chat"})
with open(os.path.join(OUT, "prompts_train.jsonl"), "w") as f:
    for x in train: f.write(json.dumps(x, ensure_ascii=False) + "\n")
with open(os.path.join(OUT, "prompts_eval.jsonl"), "w") as f:
    for x in ev: f.write(json.dumps(x, ensure_ascii=False) + "\n")
json.dump({"seed": SEED, "train_lives": train_lives, "eval_lives": eval_lives, "spare_lives": spare_lives,
           "public_train": "ultrachat_200k train_sft (600 sampled rows, first 500 used)",
           "public_eval": "ultrachat_200k test_sft (100 sampled rows, first 5 used)",
           "counts": {"train": len(train), "eval": len(ev)}}, open(os.path.join(OUT, "split.json"), "w"), indent=1)
import statistics
for name, xs in (("train", train), ("eval", ev)):
    for src in ("life-chat", "life-recall", "public-chat"):
        n = [x["n_prompt"] for x in xs if x["src"] == src]
        if n: print(name, src, len(n), "prompt tok mean", round(statistics.mean(n)), "max", max(n))
