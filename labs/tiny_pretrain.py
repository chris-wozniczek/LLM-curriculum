"""Pretrain a tiny character-level GPT from zero on WikiText-2 and log the loss curve
plus samples, so you can watch a model go from noise to almost-English.

python labs/tiny_pretrain.py --text wikitext2-test.parquet --steps 1500 --out pretrain.json
About 0.9M parameters. Took about 2 minutes on an 8-core CPU. Same architecture ideas as nanoGPT.
"""
import argparse, json, math, time
import pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--text", required=True)
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--out", default="pretrain.json")
a = p.parse_args()
torch.manual_seed(0)

text = "\n".join(t for t in pd.read_parquet(a.text)["text"].tolist() if t.strip())
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text]); n = int(len(data) * 0.9)
train, val = data[:n], data[n:]
V, T, D, L, H, B = len(chars), 128, 128, 4, 4, 32

class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.n1, s.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.qkv, s.o = nn.Linear(D, 3 * D), nn.Linear(D, D)
        s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(s, x):
        q, k, v = s.qkv(s.n1(x)).view(x.size(0), x.size(1), 3, H, D // H).unbind(2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + s.o(y.transpose(1, 2).reshape(x.shape))
        return x + s.mlp(s.n2(x))

class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.tok, s.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        s.blocks = nn.Sequential(*[Block() for _ in range(L)]); s.nf = nn.LayerNorm(D); s.head = nn.Linear(D, V)
    def forward(s, idx):
        x = s.tok(idx) + s.pos(torch.arange(idx.size(1)))
        return s.head(s.nf(s.blocks(x)))

m = GPT(); nparams = sum(x.numel() for x in m.parameters())
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.1)
batch = lambda d: (lambda ix: (torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])))(torch.randint(len(d) - T - 1, (B,)))

@torch.no_grad()
def evaluate():
    m.eval(); out = []
    for d in (train, val):
        out.append(sum(F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item() for x, y in (batch(d) for _ in range(10))) / 10)
    m.train(); return out

@torch.no_grad()
def sample(n=200):
    m.eval(); g = torch.Generator().manual_seed(1); idx = torch.tensor([[stoi[c] for c in " The "]])
    for _ in range(n):
        pr = torch.softmax(m(idx[:, -T:])[0, -1] / 0.8, -1); idx = torch.cat([idx, torch.multinomial(pr, 1, generator=g)[None]], 1)
    m.train(); return "".join(chars[i] for i in idx[0].tolist())

log, samples, t0 = [], {}, time.time()
for step in range(a.steps + 1):
    lr = 3e-3 * min(1, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
    for g in opt.param_groups: g["lr"] = lr
    if step % 50 == 0:
        tr, va = evaluate(); log.append({"step": step, "train": round(tr, 4), "val": round(va, 4), "lr": round(lr, 6), "s": round(time.time() - t0, 1)})
        print(log[-1], flush=True)
    if step in (0, 100, 300, a.steps): samples[str(step)] = sample()
    if step == a.steps: break
    x, y = batch(train); loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()

json.dump({"params": nparams, "vocab": V, "context": T, "layers": L, "width": D, "heads": H, "batch": B,
           "train_chars": len(train), "val_chars": len(val), "tokens_seen": a.steps * B * T,
           "uniform_loss": round(math.log(V), 4), "log": log, "samples": samples}, open(a.out, "w"), indent=1)
