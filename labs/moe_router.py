"""Watch a real MoE router: which experts each token picks, how evenly experts are used,
and how expert choice differs between prose and code.

python labs/moe_router.py --model allenai/OLMoE-1B-7B-0125-Instruct --text wikitext2-test.parquet --code some_file.py
OLMoE: 16 layers, 64 experts, 8 active per token. ~14 GB in BF16, runs on CPU with 32 GB RAM.
"""
import argparse, json
import pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="allenai/OLMoE-1B-7B-0125-Instruct")
p.add_argument("--text", required=True)
p.add_argument("--code", required=True)
p.add_argument("--tokens", type=int, default=2048)
p.add_argument("--out", default="moe.json")
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
c = model.config
E, K = c.num_experts, c.num_experts_per_tok
total = sum(x.numel() for x in model.parameters())
expert = sum(x.numel() for n, x in model.named_parameters() if ".experts." in n)
R = {"model": a.model, "layers": c.num_hidden_layers, "experts": E, "top_k": K, "total_params": total,
     "expert_params": expert, "active_params": total - expert + expert * K // E}

@torch.no_grad()
def route(ids):
    o = model(ids[None], output_router_logits=True)
    return [torch.softmax(r.float(), -1) for r in o.router_logits]  # per layer: [tokens, experts]

prose = "\n".join(pd.read_parquet(a.text)["text"].tolist())
code = open(a.code).read()
use = {}
for name, txt in [("prose", prose), ("code", code)]:
    ids = tok(txt, return_tensors="pt").input_ids[0][: a.tokens]
    probs = []
    for s in range(0, len(ids), 512): probs.append(route(ids[s:s + 512]))
    probs = [torch.cat([p[l] for p in probs]) for l in range(c.num_hidden_layers)]
    counts = [torch.bincount(p.topk(K, -1).indices.flatten(), minlength=E).tolist() for p in probs]
    use[name] = {"tokens": len(ids), "counts": counts,
                 "topk_mass": [round(float(p.topk(K, -1).values.sum(-1).mean()), 4) for p in probs]}
R["use"] = use

sent = "The cat sat on the mat. def add(a, b): return a + b"
ids = tok(sent, return_tensors="pt").input_ids[0]
pr = route(ids)
R["sent"] = {"tokens": [tok.decode([int(i)]) for i in ids],
             "layers": {str(l): [{"e": t.topk(K).indices.tolist(), "p": [round(float(v), 4) for v in t.topk(K).values]} for t in pr[l]] for l in [0, 7, 15]}}
json.dump(R, open(a.out, "w"), indent=1)
print({k: R[k] for k in ["layers", "experts", "top_k", "total_params", "expert_params", "active_params"]})
for n, u in use.items(): print(n, u["tokens"], "max/min uses layer7:", max(u["counts"][7]), min(u["counts"][7]), u["topk_mass"][:4])
