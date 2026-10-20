"""Fine-tune the official Glimmer DFlash drafter on the target's own outputs (DFlash recipe, offline features).

Recipe (DFlash paper sec. 4.2 + appendix A.1, SpecForge dflash objective):
  - start from the official drafter weights; target embed + lm_head frozen and shared; every drafter weight trains
  - random anchor blocks inside the target-generated response: slot 0 = clean anchor token, slots 1..15 = mask token
  - each block sees the target's hidden taps of all tokens before its anchor (sliding window 2048) + its own block
    (bidirectional), never another block
  - CE on slots 1..15 with weight exp(-(k-1)/gamma), gamma 7 for block 16; slot 0 unsupervised
  - AdamW (0.9, 0.95), grad clip 1.0, cosine schedule with warmup; ONE short epoch per learning rate (--lrs)
Memory on one 24 GB 4090: bf16 weights + grads on the GPU, fp32 master weights + AdamW state on the CPU.

usage: train_drafter.py WORKDIR [--lrs 2e-5,6e-5] [--batch 16] [--anchors 256] [--epochs 1] [--eval-only]
"""
import argparse, json, math, os, random, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("workdir")
ap.add_argument("--lrs", default="2e-5,6e-5", help="comma-separated peak learning rates; one run each, best on val kept")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--anchors", type=int, default=256)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--anchors-public", type=int, default=64, help="anchor cap for public-chat sequences (their long replies would otherwise dominate the life turns)")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--gamma", type=float, default=7.0)
ap.add_argument("--warmup", type=float, default=0.1)
ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--out", default="ckpt/drafter-tuned")
ap.add_argument("--seed", type=int, default=20261014)
args = ap.parse_args()
W = args.workdir
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
dev = torch.device("cuda")
T0 = time.time()
def log(*a):
    print(f"[{(time.time()-T0)/60:6.1f} min]", *a, flush=True)

from transformers.models.muse_glimmer_assistant import MuseGlimmerAssistantModel
model, info = MuseGlimmerAssistantModel.from_pretrained(f"{W}/models/drafter-stock", dtype=torch.bfloat16, output_loading_info=True)
assert not any(info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys")), info
model.to(dev)
cfg = model.config
BS, MASK, WIN = cfg.block_size, cfg.mask_token_id, cfg.sliding_window
NH, NKV, HD = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
eh = load_file(f"{W}/models/embed_head.safetensors")
EMBED = eh["embed"]                                  # raw table (DFlash bypasses Glimmer's embed norm), stays on CPU
LM_HEAD = eh["lm_head"].to(dev)                      # frozen, shared with the target
TCFG = json.load(open(f"{W}/models/target-nf4/config.json"))["text_config"]
MULT, CAP = TCFG["output_multiplier"], TCFG["final_logit_softcapping"]
inv_freq = 1.0 / (cfg.rope_parameters["rope_theta"] ** (torch.arange(0, HD, 2, dtype=torch.float32, device=dev) / HD))
log(f"drafter {sum(p.numel() for p in model.parameters())/1e9:.3f}B params; block {BS}, mask {MASK}, window {WIN}")

def rope(pos):
    f = pos.float()[:, None] * inv_freq[None]
    e = torch.cat([f, f], -1)
    return e.cos().to(torch.bfloat16), e.sin().to(torch.bfloat16)
def rot(x, cos, sin):  # x (H, N, D); NEOX rotate_half, as in the HF drafter
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
    return x * cos[None] + torch.cat([-x2, x1], -1) * sin[None]

def forward(rec, anchors, hs_key="hs"):
    """Run the drafter on len(anchors) blocks of one sequence. Returns final hidden (Nb, BS, D) and labels (Nb, BS-1)."""
    ids = rec["ids"].long()
    L = ids.numel()
    nb = len(anchors)
    a = torch.tensor(anchors, dtype=torch.long)
    lo = max(0, int(a.min()) - WIN)                   # oldest context any block can see
    hi = int(a.max())                                 # context positions < max anchor
    ctx = rec[hs_key][lo:hi].to(dev, non_blocking=True)
    T = hi - lo
    noise_ids = torch.full((nb, BS), MASK, dtype=torch.long)
    noise_ids[:, 0] = ids[a]
    x = EMBED[noise_ids.view(-1)].to(dev)             # (Q, D)
    qpos = (a[:, None] + torch.arange(BS)[None]).view(-1).to(dev)
    cpos = torch.arange(lo, hi, device=dev)
    lab_pos = a[:, None] + torch.arange(1, BS)[None]
    labels = torch.where(lab_pos < L, ids[lab_pos.clamp(max=L - 1)], torch.full_like(lab_pos, -100)).to(dev)
    # mask (Q, T + Q): context p allowed iff p < anchor and qpos - p < WIN; own block fully visible
    a_row = a.to(dev).repeat_interleave(BS)
    m_ctx = (cpos[None] < a_row[:, None]) & ((qpos[:, None] - cpos[None]) < WIN)
    blk = torch.arange(nb, device=dev).repeat_interleave(BS)
    m_blk = blk[:, None] == blk[None]
    mask = torch.cat([m_ctx, m_blk], 1)[None, None]
    cos_q, sin_q = rope(qpos)
    cos_k, sin_k = rope(torch.cat([cpos, qpos]))
    c = model.encoder(ctx)                            # fc + norm, shared by every layer
    h = x
    for layer in model.layers:
        att = layer.self_attn
        r = h
        y = layer.input_layernorm(h)
        kv = torch.cat([c, y], 0)
        q = att.q_norm(att.q_proj(y).view(-1, NH, HD)).transpose(0, 1)
        k = att.k_norm(att.k_proj(kv).view(-1, NKV, HD)).transpose(0, 1)
        v = att.v_proj(kv).view(-1, NKV, HD).transpose(0, 1)
        q, k = rot(q, cos_q, sin_q), rot(k, cos_k, sin_k)
        k, v = k.repeat_interleave(NH // NKV, 0), v.repeat_interleave(NH // NKV, 0)   # GQA -> MHA so SDPA can use its memory-efficient kernel
        o = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=mask, scale=HD ** -0.5)
        h = r + att.o_proj(o[0].transpose(0, 1).reshape(-1, NH * HD))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    h = model.norm(h).view(nb, BS, -1)
    return h, labels

WEIGHTS = torch.exp(-torch.arange(0, BS - 1, dtype=torch.float32) / args.gamma).to(dev)   # slot k=1..15 -> exp(-(k-1)/g)

def logits_of(h, scaled):
    z = (h @ LM_HEAD.T).float()
    if scaled:
        z = CAP * torch.tanh(z * MULT / CAP)
    return z

def block_loss(h, labels, scaled, chunk=64):
    """weighted CE summed over slots 1..15; logits built chunk by chunk with checkpointing."""
    tot = torch.zeros((), device=dev)
    for s in range(0, h.shape[0], chunk):
        hh, ll = h[s:s + chunk, 1:], labels[s:s + chunk]
        def f(hh, ll):
            z = logits_of(hh.reshape(-1, hh.shape[-1]), scaled)
            ce = F.cross_entropy(z, ll.reshape(-1), ignore_index=-100, reduction="none").view(ll.shape)
            return (ce * WEIGHTS[None] * (ll >= 0)).sum()
        tot = tot + (torch.utils.checkpoint.checkpoint(f, hh, ll, use_reentrant=False) if torch.is_grad_enabled() else f(hh, ll))
    return tot

def wsum(labels):
    return float((WEIGHTS[None] * (labels >= 0)).sum())

def valid_anchors(rec):
    return list(range(rec["P"], rec["ids"].numel() - 1))

# ---------------- data ----------------
seqs = json.load(open(f"{W}/data/seqs.json"))
train = [s for s in seqs if s["split"] == "train" and os.path.exists(f"{W}/hs/{s['id']}.pt")]
val = [s for s in seqs if s["split"] == "val" and os.path.exists(f"{W}/hs/{s['id']}.pt")]
def load(s):
    return torch.load(f"{W}/hs/{s['id']}.pt", weights_only=True)
log(f"train {len(train)} seqs, val {len(val)} seqs")

@torch.no_grad()
def evaluate(items, scaled_list=(False, True), hs_key="hs", max_anchors=256):
    """teacher-forced greedy acceptance on every valid anchor: tau = 1 + mean leading-correct slots (draft max 15)."""
    model.eval()
    acc_len, n_blocks, pos1 = 0.0, 0, 0.0
    by_src = {}
    ce = {sc: [0.0, 0.0] for sc in scaled_list}
    for s in items:
        rec = load(s)
        if hs_key not in rec:
            continue
        an = valid_anchors(rec)[:max_anchors]
        if not an:
            continue
        for i in range(0, len(an), 128):
            h, lab = forward(rec, an[i:i + 128], hs_key)
            for sc in scaled_list:
                ce[sc][0] += float(block_loss(h, lab, sc)); ce[sc][1] += wsum(lab)
            pred = torch.cat([(h[j:j + 64, 1:] @ LM_HEAD.T).argmax(-1) for j in range(0, h.shape[0], 64)])
            ok = (pred == lab) & (lab >= 0)
            lead = ok.int().cumprod(1).sum(1).float()
            acc_len += float(lead.sum()); n_blocks += lead.numel(); pos1 += float(ok[:, 0].float().sum())
            d = by_src.setdefault(s["src"], [0.0, 0])
            d[0] += float(lead.sum()); d[1] += lead.numel()
    model.train()
    return {"tau": 1 + acc_len / max(n_blocks, 1), "pos1_acc": pos1 / max(n_blocks, 1), "blocks": n_blocks,
            "tau_by_src": {k: round(1 + v[0] / v[1], 3) for k, v in by_src.items()},
            "wce": {("scaled" if k else "raw"): round(v[0] / max(v[1], 1), 4) for k, v in ce.items()}}

# ---------------- step-0 checks on the stock drafter ----------------
check = [s for s in val + train if "hs_alt" in torch.load(f"{W}/hs/{s['id']}.pt", weights_only=True, mmap=True)][:16]
assert check, "no sequences carry the off-by-one taps for the convention check"
res_main = evaluate(check, (False,), "hs")
res_alt = evaluate(check, (False,), "hs_alt")
log(f"tap convention check on {len(check)} val seqs: taps=layer outputs {cfg.target_layer_ids} -> tau {res_main['tau']:.3f}; "
    f"one layer earlier -> tau {res_alt['tau']:.3f}")
assert res_main["tau"] > res_alt["tau"], "tap convention looks wrong"
stock = evaluate(val)
log("STOCK val:", json.dumps(stock))
SCALED = stock["wce"]["scaled"] < stock["wce"]["raw"]
log(f"training CE on {'scaled+softcapped' if SCALED else 'raw lm_head'} logits (the stock drafter is better calibrated there)")
if args.eval_only:
    sys.exit(0)

# ---------------- one short epoch per learning rate; the best on the held-out val lives is kept ----------------
def train_one(lr, out):
    global model
    model = MuseGlimmerAssistantModel.from_pretrained(f"{W}/models/drafter-stock", dtype=torch.bfloat16).to(dev)
    params = [p for p in model.parameters()]
    master = [p.detach().float().cpu() for p in params]   # fp32 master weights + AdamW state live on the CPU
    for m in master:
        m.requires_grad_(True)
        m.grad = torch.zeros_like(m)                       # persistent fp32 grads: no 10 GB allocation per step
    stage = [torch.empty(p.shape, dtype=torch.bfloat16, pin_memory=True) for p in params]   # pinned bf16 staging
    try:
        opt = torch.optim.AdamW(master, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    except Exception:
        opt = torch.optim.AdamW(master, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=True)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = max(1, int(round(steps_per_epoch * args.epochs)))
    warm = max(5, int(round(total * args.warmup)))
    def lr_at(step):
        if step < warm:
            return lr * (step + 1) / warm
        return lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    log(f"[lr {lr:g}] optimizer: {'fused' if opt.defaults.get('fused') else 'foreach'} AdamW on the CPU")
    log(f"[lr {lr:g}] {total} optimizer steps (batch {args.batch} seqs, <= {args.anchors} anchors/seq), warmup {warm}")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    r = random.Random(args.seed)
    order = []
    while len(order) < total * args.batch:
        ep = list(train); r.shuffle(ep); order += ep
    hist = []
    t_train = time.time()
    for step in range(total):
        batch = order[step * args.batch:(step + 1) * args.batch]
        recs = [load(s) for s in batch]
        ans = []
        for s_, rec in zip(batch, recs):
            va = valid_anchors(rec)
            cap = args.anchors_public if s_["src"] == "public-chat" else args.anchors
            ans.append(sorted(r.sample(va, min(cap, len(va)))))
        norm = 0.0
        for rec, an in zip(recs, ans):
            L = rec["ids"].numel()
            for aa in an:
                norm += float(WEIGHTS[: max(0, min(BS - 1, L - 1 - aa))].sum())
        lsum = 0.0
        for rec, an in zip(recs, ans):
            for c0 in range(0, len(an), args.chunk):          # <= chunk blocks per forward/backward keeps activations small
                h, lab = forward(rec, an[c0:c0 + args.chunk])
                loss = block_loss(h, lab, SCALED) / norm
                loss.backward()
                lsum += float(loss.detach())
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        with torch.no_grad():
            for p, st in zip(params, stage):               # GPU bf16 grads -> pinned host
                st.copy_(p.grad, non_blocking=True)
                p.grad = None
            torch.cuda.synchronize()
            for m, st in zip(master, stage):
                m.grad.copy_(st)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.step()                                     # fp32 AdamW on the CPU
            for m, st in zip(master, stage):               # fp32 master -> pinned bf16 -> GPU weights
                st.copy_(m)
            for p, st in zip(params, stage):
                p.copy_(st, non_blocking=True)
            torch.cuda.synchronize()
        hist.append({"step": step + 1, "loss": round(lsum, 4), "grad_norm": round(gn, 3), "lr": lr_at(step)})
        if step % 10 == 0 or step == total - 1:
            log(f"[lr {lr:g}] step {step+1}/{total} loss {lsum:.4f} (mean last 10: {sum(h['loss'] for h in hist[-10:])/len(hist[-10:]):.4f}) "
                f"gnorm {gn:.2f} lr {lr_at(step):.2e} peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    train_min = (time.time() - t_train) / 60
    tuned = evaluate(val)
    log(f"[lr {lr:g}] TUNED val:", json.dumps(tuned))
    os.makedirs(f"{W}/{out}", exist_ok=True)
    model.save_pretrained(f"{W}/{out}", safe_serialization=True)
    rep = {"lr": lr, "args": vars(args), "train_seqs": len(train), "val_seqs": len(val), "steps": total, "warmup_steps": warm,
           "train_minutes": round(train_min, 2), "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "ce_on_scaled_logits": SCALED, "stock_val": stock, "tuned_val": tuned, "tap_check": [res_main["tau"], res_alt["tau"]],
           "history": hist}
    json.dump(rep, open(f"{W}/{out}/train_report.json", "w"), indent=1)
    del opt, master, params, stage
    return rep

reports = []
for lr in [float(x) for x in args.lrs.split(",")]:
    reports.append(train_one(lr, f"{args.out}-lr{lr:g}"))
def score(v):  # equal weight per prompt source, so the long public replies do not decide alone
    return sum(v["tau_by_src"].values()) / len(v["tau_by_src"])
best = max(reports, key=lambda r: score(r["tuned_val"]))
json.dump({"stock_val": stock, "stock_score": score(stock), "runs": [{"lr": r["lr"], "tau": r["tuned_val"]["tau"], "score": score(r["tuned_val"]), "tau_by_src": r["tuned_val"]["tau_by_src"],
           "train_minutes": r["train_minutes"]} for r in reports], "best_lr": best["lr"],
           "best_dir": f"{args.out}-lr{best['lr']:g}"}, open(f"{W}/{args.out}-sweep.json", "w"), indent=1)
log(f"TRAIN-DONE best lr {best['lr']:g}: val tau {stock['tau']:.3f} -> {best['tuned_val']['tau']:.3f}; per-source mean {score(stock):.3f} -> {score(best['tuned_val']):.3f}")
