"""Cache the target's hidden taps for DFlash training (offline mode, as in the DFlash paper and SpecForge).

For every (prompt + target greedy output) sequence, run the Glimmer NF4 target once and store the outputs of
decoder layers TAPS (0-based; the drafter's target_layer_ids; z-lab reads hidden_states[id + 1], i.e. the output
of layer id). Layers after the last tap are skipped. Also writes the frozen embed/lm_head the trainer needs and a
check set with the off-by-one taps (layers id - 1) so the trainer can confirm the convention on the stock drafter.

usage: extract_hs.py WORKDIR
"""
import json, os, random, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

W = sys.argv[1]
TAPS = [1, 13, 25, 37, 49]
ALT = [t - 1 for t in TAPS]
MODEL = f"{W}/models/target-nf4"
OUT = f"{W}/hs"
os.makedirs(OUT, exist_ok=True)
t_start = time.time()

# ---------- data: prompt ids + the target's own greedy output; validation split by life BEFORE training ----------
prompts = {json.loads(l)["id"]: json.loads(l) for l in open(f"{W}/data/prompts_train.jsonl")}
gens = [json.loads(l) for l in open(f"{W}/data/gen_train.jsonl")]
rng = random.Random(7)
lives = sorted({prompts[g["id"]]["life_id"] for g in gens if prompts[g["id"]]["life_id"]})
val_lives = set(rng.sample(lives, 40))
pub = sorted(g["id"] for g in gens if prompts[g["id"]]["src"] == "public-chat")
val_pub = set(rng.sample(pub, 40))
seqs = []
for g in gens:
    p = prompts[g["id"]]
    if g["n_gen"] < 2:
        continue
    split = "val" if (p["life_id"] in val_lives or g["id"] in val_pub) else "train"
    seqs.append({"id": g["id"], "src": p["src"], "split": split, "P": p["n_prompt"],
                 "ids": p["prompt_ids"] + g["gen_ids"]})
check_ids = {s["id"] for s in sorted((s for s in seqs if s["split"] == "val"), key=lambda s: s["id"])[:16]}
json.dump([{k: s[k] for k in ("id", "src", "split", "P")} | {"L": len(s["ids"])} for s in seqs],
          open(f"{W}/data/seqs.json", "w"), indent=0)
print(f"{len(seqs)} sequences ({sum(s['split']=='train' for s in seqs)} train, {sum(s['split']=='val' for s in seqs)} val), "
      f"{sum(len(s['ids']) for s in seqs)} tokens, {sum(len(s['ids'])-s['P'] for s in seqs)} generated", flush=True)

# ---------- frozen embed + head for the trainer (bf16, as stored unquantised in the NF4 checkpoint) ----------
eh = f"{W}/models/embed_head.safetensors"
if not os.path.exists(eh):
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    t = {}
    for key, name in (("model.language_model.embed_tokens.weight", "embed"), ("lm_head.weight", "lm_head")):
        with safe_open(f"{MODEL}/{idx[key]}", "pt") as f:
            t[name] = f.get_tensor(key).to(torch.bfloat16).contiguous()
    save_file(t, eh)
    print("saved embed/lm_head", {k: tuple(v.shape) for k, v in t.items()}, flush=True)

# ---------- target ----------
from transformers import AutoConfig, MuseGlimmerForConditionalGeneration
config = AutoConfig.from_pretrained(MODEL)
qc = config.quantization_config
if isinstance(qc, dict):
    qc["llm_int8_enable_fp32_cpu_offload"] = True   # lets the unquantised lm_head / vision tower sit on the CPU (unused)
else:
    qc.llm_int8_enable_fp32_cpu_offload = True
dm = {"model.language_model.layers": 0, "model.language_model.norm": 0, "model.language_model.rotary_emb": 0,
      "model.language_model.embed_tokens": 0, "lm_head": "cpu", "model.vision_tower": "cpu",
      "model.vision_adapter": "cpu", "model.vision_projection": "cpu"}
model = MuseGlimmerForConditionalGeneration.from_pretrained(MODEL, config=config, device_map=dm, dtype=torch.bfloat16)
model.eval()
lm = model.model.language_model
print(f"target loaded in {time.time()-t_start:.0f}s; VRAM {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

cap = {}
class Stop(Exception):
    pass
want = set(TAPS)
def hook(i):
    def f(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if i in want:
            cap[i] = h.detach()
        if i == max(want):
            raise Stop()
    return f
for i, layer in enumerate(lm.layers):
    layer.register_forward_hook(hook(i))

def run(batch, layers):
    global want
    want = set(layers)
    cap.clear()
    L = max(len(s["ids"]) for s in batch)
    ids = torch.full((len(batch), L), 200018, dtype=torch.long)
    am = torch.zeros((len(batch), L), dtype=torch.long)
    for b, s in enumerate(batch):
        ids[b, :len(s["ids"])] = torch.tensor(s["ids"])
        am[b, :len(s["ids"])] = 1
    with torch.no_grad():
        try:
            lm(input_ids=ids.to(0), attention_mask=am.to(0), use_cache=False)
        except Stop:
            pass
    return [torch.cat([cap[i][b, :len(s["ids"])] for i in layers], dim=-1).cpu() for b, s in enumerate(batch)]

order = sorted(seqs, key=lambda s: len(s["ids"]))
batches, cur = [], []
for s in order:
    if cur and (len(cur) + 1) * len(s["ids"]) > 6144:
        batches.append(cur); cur = []
    cur.append(s)
if cur:
    batches.append(cur)
torch.cuda.reset_peak_memory_stats()
t0 = time.time(); ntok = 0
LIMIT = int(os.environ.get("LIMIT_BATCHES", "0"))
for bi, batch in enumerate(batches):
    if LIMIT and bi >= LIMIT:
        break
    todo = [s for s in batch if not os.path.exists(f"{OUT}/{s['id']}.pt")]
    if not todo:
        continue
    hs = run(todo, TAPS)
    alt = run([s for s in todo if s["id"] in check_ids], ALT) if any(s["id"] in check_ids for s in todo) else []
    ai = 0
    for s, h in zip(todo, hs):
        rec = {"ids": torch.tensor(s["ids"], dtype=torch.int32), "P": s["P"], "hs": h.contiguous()}
        if s["id"] in check_ids:
            rec["hs_alt"] = alt[ai].contiguous(); ai += 1
        torch.save(rec, f"{OUT}/{s['id']}.pt.tmp")
        os.replace(f"{OUT}/{s['id']}.pt.tmp", f"{OUT}/{s['id']}.pt")
        ntok += len(s["ids"])
    if bi % 25 == 0:
        el = time.time() - t0
        print(f"batch {bi+1}/{len(batches)}: {ntok} tok, {ntok/max(el,1e-6):.0f} tok/s, peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
el = time.time() - t0
print(f"EXTRACT-DONE {ntok} tokens in {el/60:.1f} min ({ntok/max(el,1):.0f} tok/s); peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB; "
      f"total wall {(time.time()-t_start)/60:.1f} min", flush=True)
