#!/usr/bin/env python3
"""QLoRA training-rate measurement: QLoRA r16/a32/dp.05, q/k/v/o FINAL16 text layers (36..51 of 52),
LR 1e-4, batch1 GA1 (authorized change from GA4, disclosed), response-only loss,
vision/embed/head frozen, gradient checkpointing. Single session: SEQ 2048 then 4096.
>=3 warmup + >=3 measured full optimizer steps each, CUDA sync.
Sustained tok/s from measured-phase wall (includes data/collation between steps).
Pinned chat template from chat_template.jinja; response header validated against render.
Usage: rate_both.py OUTDIR   (model dir from $GLIMMER_NF4, default ./models)
Exit 0 both lengths ok; exit 2 otherwise (JSONs still written, no fake numbers).
"""
import json, os, re, resource, subprocess, sys, threading, time, traceback
import torch
OUTDIR = sys.argv[1]
MODELS = os.environ.get("GLIMMER_NF4", os.path.abspath("models"))
WARM, MEAS = 3, 3
os.makedirs(OUTDIR, exist_ok=True)
print(f"torch={torch.__version__} cuda_build={torch.version.cuda} cuda_avail={torch.cuda.is_available()}", flush=True)
assert torch.cuda.is_available(), "no CUDA device"
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from datasets import Dataset
import transformers, peft, trl
VERS = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
        "transformers": transformers.__version__, "peft": peft.__version__, "trl": trl.__version__}
print("versions:", json.dumps(VERS), flush=True)
props = torch.cuda.get_device_properties(0)
print(f"gpu={props.name} total_gib={props.total_memory/1024**3:.2f}", flush=True)

def device_census(mod):
    census = {}
    for n, p in mod.named_parameters():
        key = "embed" if "embed_tokens" in n else ("lm_head" if "lm_head" in n else ("vision" if "vision" in n else "language"))
        census.setdefault(key, {}).setdefault(str(p.device), 0)
        census[key][str(p.device)] += p.numel()
    return census

# ---- load (structured capture incl load OOM) ----
load_info = {}
t_load0 = time.time()
try:
    model, tokenizer = FastModel.from_pretrained(model_name=MODELS, max_seq_length=4096)
    # Exact 64-module scope: full HF paths verified against the pinned safetensors
    # index (model.language_model.layers.36..51.self_attn.q/k/v/o_proj). Full paths
    # keep unsloth from regex-routing (which PEFT rejects with layers_to_transform).
    TARGETS = [f"model.language_model.layers.{l}.self_attn.{m}"
               for l in range(36, 52) for m in ("q_proj", "k_proj", "v_proj", "o_proj")]
    assert len(TARGETS) == 64
    model = FastModel.get_peft_model(model,
        finetune_vision_layers=True, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        target_modules=TARGETS,
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407)
except Exception:
    load_info = {"load_error": traceback.format_exc()[-2000:],
                 "load_s": round(time.time() - t_load0, 1)}
    try:
        load_info["cuda_alloc_gib"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        load_info["cuda_reserved_gib"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
    except Exception:
        pass
    print("LOAD-FAILED", flush=True)
    open(os.path.join(OUTDIR, "rate-summary.json"), "w").write(json.dumps(
        {"status": "load-failed", "versions": VERS, "load": load_info,
         "recipe": "r16/a32/dp.05 qkv-o final16 GA1 LR1e-4"}, indent=1))
    print(json.dumps(load_info, indent=1), flush=True)
    sys.exit(2)
load_s = time.time() - t_load0
load_resident = torch.cuda.memory_reserved() / 1024**3
load_alloc = torch.cuda.memory_allocated() / 1024**3
print(f"load+peft s: {load_s:.1f} resident_gib={load_resident:.2f} alloc_gib={load_alloc:.2f}", flush=True)

# ---- trainable inventory + scope assertions ----
train_names = [n for n, p in model.named_parameters() if p.requires_grad]
trainable = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
print(f"trainable params: {trainable} in {len(train_names)} tensors", flush=True)
bad_mod = [n for n in train_names if not any(m in n for m in ("q_proj", "k_proj", "v_proj", "o_proj"))]
layer_ids = sorted(set(int(m.group(1)) for n in train_names for m in [re.search(r"layers\.(\d+)\.", n)] if m))
print("trainable layer ids:", layer_ids, flush=True)
frozen_leak = [n for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ("embed_tokens", "lm_head", "vision", "visual", "projector"))]
scope_err = None
if bad_mod: scope_err = f"non-qkvo trainable: {bad_mod[:6]}"
elif set(layer_ids) - set(range(36, 52)): scope_err = f"layers outside 36..51: {layer_ids}"
elif frozen_leak: scope_err = f"frozen leak: {frozen_leak[:6]}"
elif not train_names: scope_err = "zero trainable params"
print("device_census_post_load:", json.dumps(device_census(model)), flush=True)
if scope_err:
    print("SCOPE-FAIL:", scope_err, flush=True)
    open(os.path.join(OUTDIR, "rate-summary.json"), "w").write(json.dumps(
        {"status": "scope-fail", "scope_error": scope_err, "versions": VERS,
         "trainable_params": trainable, "layer_ids": layer_ids}, indent=1))
    sys.exit(2)

# ---- pinned chat template ----
tok = getattr(tokenizer, "tokenizer", tokenizer)
with open(os.path.join(MODELS, "chat_template.jinja")) as f:
    pinned_template = f.read()
tok.chat_template = pinned_template
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
probe = tok.apply_chat_template(
    [{"role": "user", "content": "Probe."},
     {"role": "assistant", "content": "Copy."}], tokenize=False)
m = re.search(r"<\|start\|>assistant to=user<\|message\|>", probe)
assert m, "response header missing from pinned-template render"
RESPONSE_PART = m.group(0)
assert "<|start|>user<|message|>" in probe, "user header missing"
INSTRUCTION_PART = "<|start|>user<|message|>"
print("template pinned from chat_template.jinja; response header validated", flush=True)

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
         "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu station "
         "checklist pelican harbor beacon ledger compass anchor tide signal relay beacon north").split()

def build_texts(seq, n):
    texts = []
    for r in range(n):
        msgs = [{"role": "user", "content": "Log entry " + " ".join(WORDS[(r * 7 + i) % len(WORDS)] for i in range(40)) + "."}]
        # grow assistant content until tokenized length hits seq
        reps = 1
        while True:
            acontent = ("Copy that, Pelican station. " + " ".join(WORDS) + " ") * reps
            t = tok.apply_chat_template(msgs + [{"role": "assistant", "content": acontent}], tokenize=False)
            L = len(tok(t, add_special_tokens=False).input_ids)
            if L >= seq - 60 or reps > 400:
                break
            reps += 1
        # top up word by word to exact seq
        words_extra = []
        t = tok.apply_chat_template(msgs + [{"role": "assistant", "content": acontent}], tokenize=False)
        L = len(tok(t, add_special_tokens=False).input_ids)
        wi = 0
        while L < seq and wi < 5000:
            words_extra.append(WORDS[wi % len(WORDS)])
            t2 = tok.apply_chat_template(
                msgs + [{"role": "assistant", "content": acontent + " " + " ".join(words_extra)}], tokenize=False)
            L2 = len(tok(t2, add_special_tokens=False).input_ids)
            if L2 > seq:
                words_extra.pop()
                break
            t, L = t2, L2
            wi += 1
        ids = tok(t, add_special_tokens=False).input_ids
        if len(ids) < seq:
            ids = ids + [tok.pad_token_id] * (seq - len(ids))
            t = tok.decode([i for i in ids if i != tok.pad_token_id])
        texts.append(t)
    lens = sorted(set(len(tok(t, add_special_tokens=False).input_ids) for t in texts))
    print(f"seq={seq} retok lens set: {lens}", flush=True)
    assert max(lens) <= seq and min(lens) >= seq - 128, f"shape fail {lens} vs [{seq}]"
    return texts

class NvmlSampler:
    def __init__(self):
        self.peak = None; self.stop = threading.Event(); self.th = None
    def start(self):
        self.th = threading.Thread(target=self.loop, daemon=True); self.th.start()
    def loop(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                      "--format=csv,nounits,noheader"],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
                mib = int(float(out.split()[0]))
                if self.peak is None or mib > self.peak:
                    self.peak = mib
            except Exception:
                pass
            self.stop.wait(1.0)
    def finish(self):
        self.stop.set()
        if self.th: self.th.join(timeout=15)
        return self.peak

def run_length(seq):
    STEPS = WARM + MEAS
    N = STEPS + 2
    texts = build_texts(seq, N)
    ds = Dataset.from_dict({"text": texts})
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, eval_dataset=None,
        args=SFTConfig(dataset_text_field="text", max_length=seq, packing=False,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            warmup_steps=0, max_steps=STEPS, learning_rate=1e-4, logging_steps=1,
            optim="adamw_8bit", weight_decay=0.0, lr_scheduler_type="constant", seed=3407,
            output_dir=os.path.join(OUTDIR, f"rate-out-{seq}"), report_to="none"))
    trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART,
                                      response_part=RESPONSE_PART)
    # exact per-row counts over the STEPS rows actually consumed (batch1, sequential)
    row_counts = []
    for i in range(STEPS):
        rlab = trainer.train_dataset[i]["labels"]
        rinp = trainer.train_dataset[i]["input_ids"]
        row_counts.append((sum(1 for x in rinp if x != tok.pad_token_id),
                           sum(1 for x in rlab if x != -100)))
    n_input = row_counts[0][0]; n_loss = row_counts[0][1]
    meas_in_total = sum(c[0] for c in row_counts[WARM:STEPS])
    meas_loss_total = sum(c[1] for c in row_counts[WARM:STEPS])
    print(f"seq={seq} row0: input={n_input} loss={n_loss}; measured-rows totals: input={meas_in_total} loss={meas_loss_total}", flush=True)
    step_times = {}
    marks = {}
    class Timer(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.started = time.perf_counter()
            self.step = state.global_step
            if self.step == WARM:
                marks["meas_begin"] = time.perf_counter()
            return control
        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            step_times[self.step] = time.perf_counter() - self.started
            if self.step == STEPS - 1:
                marks["meas_end"] = time.perf_counter()
            return control
    trainer.add_callback(Timer())
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    nvml = NvmlSampler(); nvml.start()
    t0 = time.time()
    train_err, stats = None, None
    try:
        stats = trainer.train()
    except Exception:
        train_err = traceback.format_exc()[-2000:]
        stats = None
    torch.cuda.synchronize()
    wall = time.time() - t0
    nvml_peak = nvml.finish()
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_res = torch.cuda.max_memory_reserved() / 1024**3
    host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    meas_keys = sorted(k for k in step_times if k >= WARM)
    meas_t = sum(step_times[k] for k in meas_keys)
    n_done = len(meas_keys)
    meas_wall = (marks["meas_end"] - marks["meas_begin"]) if ("meas_begin" in marks and "meas_end" in marks) else None
    losses = [h.get("loss") for h in (trainer.state.log_history or []) if "loss" in h]
    res = {"seq": seq, "warm_steps": WARM, "meas_steps_planned": MEAS, "meas_steps_done": n_done,
        "rank": 16, "lora_alpha": 32, "lora_dropout": 0.05, "batch": 1, "ga": 1,
        "lr": 1e-4, "optim": "adamw_8bit", "loss": "causal-NLL response-only (ATEM masking)",
        "layer_scope": "final 16 text layers 36..51 of 52", "layer_ids": layer_ids,
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "trainable_params": trainable, "versions": VERS,
        "gpu": props.name, "total_vram_gib": round(props.total_memory / 1024**3, 2),
        "embed_device": str(model.get_input_embeddings().weight.device),
        "device_census_post_train": device_census(model),
        "host_peak_ram_gib": round(host_peak, 2),
        "input_pos_per_row": n_input, "loss_pos_per_row": n_loss,
        "train_wall_s": round(wall, 1),
        "step_times_s": {str(k): round(v, 2) for k, v in sorted(step_times.items())},
        "meas_callback_sum_s": round(meas_t, 2),
        "meas_phase_wall_s": round(meas_wall, 2) if meas_wall else None,
        "meas_input_total": meas_in_total, "meas_loss_total": meas_loss_total,
        "sustained_input_tok_s": round(meas_in_total / meas_wall, 2) if meas_wall and n_done == MEAS else None,
        "sustained_loss_tok_s": round(meas_loss_total / meas_wall, 2) if meas_wall and n_done == MEAS else None,
        "callback_input_tok_s": round(meas_in_total / meas_t, 2) if meas_t and n_done == MEAS else None,
        "callback_loss_tok_s": round(meas_loss_total / meas_t, 2) if meas_t and n_done == MEAS else None,
        "loss_per_step": losses,
        "peak_alloc_gib": round(peak_alloc, 2), "peak_reserved_gib": round(peak_res, 2),
        "nvml_peak_used_gib": round(nvml_peak / 1024, 2) if nvml_peak is not None else None,
        "error": train_err}
    out = os.path.join(OUTDIR, f"rate-{seq}.json")
    open(out, "w").write(json.dumps(res, indent=1))
    print(f"===RATE-{seq}===", flush=True)
    print(json.dumps(res, indent=1), flush=True)
    return res

results = {}
r2k = run_length(2048)
results["2048"] = r2k
ok2 = r2k["error"] is None and r2k["meas_steps_done"] >= 3
if not ok2:
    print("RATE-2048 FAILED; attempting 4096 anyway for OOM/fit evidence", flush=True)
r4k = run_length(4096)
results["4096"] = r4k
ok4 = r4k["error"] is None and r4k["meas_steps_done"] >= 3
summ = {"status": "ok" if (ok2 and ok4) else "partial",
        "load_s": round(load_s, 1), "load_resident_gib": round(load_resident, 2),
        "load_alloc_gib": round(load_alloc, 2), "versions": VERS,
        "recipe": "r16/a32/dp.05 q/k/v/o final16(36..51) GA1 LR1e-4 response-only gradckpt",
        "results": {k: {"meas_done": v["meas_steps_done"], "sustained_input_tok_s": v["sustained_input_tok_s"],
                            "sustained_loss_tok_s": v["sustained_loss_tok_s"],
                            "peak_reserved_gib": v["peak_reserved_gib"],
                            "nvml_peak_used_gib": v["nvml_peak_used_gib"],
                            "error_tail": (v["error"] or "")[-300:] if v["error"] else None}
                    for k, v in results.items()}}
open(os.path.join(OUTDIR, "rate-summary.json"), "w").write(json.dumps(summ, indent=1))
print("===RATE-SUMMARY===", flush=True)
print(json.dumps(summ, indent=1), flush=True)
if not (ok2 and ok4):
    print("RATE-DONE-PARTIAL (real errors logged, no fake numbers)", flush=True)
    sys.exit(2)
print("RATE-DONE-OK", flush=True)
