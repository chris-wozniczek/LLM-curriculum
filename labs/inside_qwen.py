"""Measure what happens inside Qwen3-0.6B: next-token distributions, logit lens,
attention patterns, residual norms, KV cache size and prefill/decode speed.

python labs/inside_qwen.py --model Qwen/Qwen3-0.6B --out inside.json
Runs on CPU in a few minutes. Timings are for whatever machine you run it on.
"""
import argparse, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B")
p.add_argument("--out", default="inside.json")
p.add_argument("--threads", type=int, default=0)
a = p.parse_args()
if a.threads: torch.set_num_threads(a.threads)
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager").eval()
cfg = model.config
R = {"model": a.model, "threads": torch.get_num_threads()}
show = lambda i: tok.decode([i])

@torch.no_grad()
def fwd(text, **kw):
    ids = tok(text, return_tensors="pt").input_ids
    return ids, model(ids, **kw)

# 1. next-token distributions (used by the sampling figure)
R["next"] = []
for text in ["The capital of France is", "My favourite fruit is", "def fibonacci(n):\n    if n <", "Once upon a time, there was a"]:
    ids, o = fwd(text)
    pr = torch.softmax(o.logits[0, -1].float(), -1)
    top = pr.topk(40)
    R["next"].append({"prompt": text, "logits": [round(float(o.logits[0, -1, i]), 3) for i in top.indices],
                      "tokens": [show(int(i)) for i in top.indices], "p": [round(float(x), 5) for x in top.values]})

# 2. tokens of one sentence, attention of every head in a few layers, logit lens, residual norms
sent = "The animal didn't cross the street because it was too tired."
ids, o = fwd(sent, output_attentions=True, output_hidden_states=True)
R["sent"] = {"text": sent, "ids": ids[0].tolist(), "tokens": [show(int(i)) for i in ids[0]]}
layers = [0, 1, 7, 14, 21, 27]
R["attn"] = {str(l): [[[round(float(v), 3) for v in row] for row in o.attentions[l][0, h]] for h in range(cfg.num_attention_heads)] for l in layers}
hs = o.hidden_states  # embeddings + one entry per layer; the last entry is already normed
R["resid_norm"] = [round(float(h[0].norm(dim=-1).mean()), 2) for h in hs[:-1]]

lens = []
lp_ids, lo = fwd("The capital of France is", output_hidden_states=True)
paris = tok(" Paris").input_ids[0]
for i, h in enumerate(lo.hidden_states[1:]):
    x = h[0, -1]
    if i < len(lo.hidden_states) - 2: x = model.model.norm(x)
    pr = torch.softmax(model.lm_head(x).float(), -1)
    t = pr.topk(3)
    lens.append({"layer": i, "top": [show(int(j)) for j in t.indices], "p": [round(float(v), 4) for v in t.values], "paris": round(float(pr[paris]), 5)})
R["lens"] = lens

# 3. KV cache: real bytes in the cache vs the formula
ids = tok("hello " * 600, return_tensors="pt").input_ids[:, :512]
with torch.no_grad(): o = model(ids, use_cache=True)
pkv = o.past_key_values
if hasattr(pkv, "layers"): kv = [(l.keys, l.values) for l in pkv.layers]
else: kv = [(pkv[i][0], pkv[i][1]) for i in range(len(pkv))]
real = sum(k.numel() + v.numel() for k, v in kv)
hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
R["kv"] = {"tokens": 512, "k_shape": list(kv[0][0].shape), "numbers": real,
           "formula": 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * hd * 512,
           "bytes_bf16_per_token": 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * hd * 2}

# 4. prefill vs decode speed, and decode with vs without the KV cache
base = tok("The history of the city " * 300, return_tensors="pt").input_ids
pre = []
for n in [16, 64, 256, 1024]:
    x = base[:, :n]
    with torch.no_grad():
        model(x[:, :8]); t = time.perf_counter(); model(x); dt = time.perf_counter() - t
    pre.append({"tokens": n, "s": round(dt, 3), "tok_s": round(n / dt, 1)})
R["prefill"] = pre

def decode(n_ctx, steps, cache):
    x = base[:, :n_ctx]
    with torch.no_grad():
        o = model(x, use_cache=cache); nxt = o.logits[:, -1:].argmax(-1); t = time.perf_counter()
        for _ in range(steps):
            if cache: o = model(nxt, past_key_values=o.past_key_values, use_cache=True)
            else: x = torch.cat([x, nxt], 1); o = model(x, use_cache=False)
            nxt = o.logits[:, -1:].argmax(-1)
    dt = time.perf_counter() - t
    return {"ctx": n_ctx, "steps": steps, "cache": cache, "s": round(dt, 3), "tok_s": round(steps / dt, 2)}
R["decode"] = [decode(c, 32, True) for c in [64, 512, 1024]] + [decode(c, 32, False) for c in [64, 512, 1024]]

json.dump(R, open(a.out, "w"), indent=1)
print(json.dumps({k: R[k] for k in ["kv", "prefill", "decode", "resid_norm"]}, indent=0)[:3000])
for l in lens: print(l["layer"], l["top"], l["paris"])
