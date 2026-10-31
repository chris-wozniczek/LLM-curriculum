"""Quantize a small model several ways and measure how far each copy drifts from the original.

Every number on the curriculum's quantization pages comes from this script.
Method: "fake quantization" = round each weight to the target format, then turn it back into
a normal float so the unmodified model code can run it. That isolates the rounding error,
which is what top-1 / KLD measure. Speed and memory are not measured here.

usage: python quant_fidelity.py --model Qwen/Qwen3-0.6B --text wikitext2-test.parquet \
         --seqs 16 --len 512 --out quant_fidelity.json
"""
import argparse, hashlib, json, math, os, time
import numpy as np, torch, pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def int_rtn(w, bits, group, sym=True):
    """Round-to-nearest integer quantization with one scale (and zero point) per group."""
    o, i = w.shape
    g = i if group is None else group
    x = w.reshape(o, i // g, g)
    if sym:
        qmax = 2 ** (bits - 1) - 1
        s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / qmax
        q = (x / s).round().clamp(-qmax - 1, qmax)
        return (q * s).reshape(o, i)
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    s = ((hi - lo) / (2 ** bits - 1)).clamp(min=1e-12)
    z = (-lo / s).round()
    q = ((x / s).round() + z).clamp(0, 2 ** bits - 1)
    return ((q - z) * s).reshape(o, i)


def fp8(w, block=None):
    """FP8 E4M3 (max 448). block=None: one scale per tensor; block=128: one per 128x128 tile."""
    if block is None:
        s = w.abs().max() / 448.0
        return (w / s).to(torch.float8_e4m3fn).float() * s
    o, i = w.shape
    x = w.reshape(o // block, block, i // block, block)
    s = x.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / 448.0
    return ((x / s).to(torch.float8_e4m3fn).float() * s).reshape(o, i)


E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])  # every magnitude a 4-bit float can hold


def to_e2m1(x):
    mag = x.abs().unsqueeze(-1)
    idx = (mag - E2M1).abs().argmin(-1)
    return E2M1[idx] * x.sign()


def mxfp4(w):
    """OCP MX FP4: blocks of 32, shared power-of-two (E8M0) scale."""
    o, i = w.shape
    x = w.reshape(o, i // 32, 32)
    amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    s = torch.exp2(torch.floor(torch.log2(amax)) - 2)  # emax of E2M1 is 2 (6 = 1.5 * 2^2)
    return (to_e2m1((x / s).clamp(-6, 6)) * s).reshape(o, i)


def nvfp4(w):
    """NVIDIA NVFP4: blocks of 16, FP8 E4M3 scale per block, plus one FP32 scale per tensor."""
    o, i = w.shape
    x = w.reshape(o, i // 16, 16)
    t = w.abs().max() / (448.0 * 6.0)
    s = (x.abs().amax(-1, keepdim=True) / 6.0 / t).to(torch.float8_e4m3fn).float().clamp(min=1e-12)
    return (to_e2m1((x / (s * t)).clamp(-6, 6)) * s * t).reshape(o, i)


# name -> (function for attention, function for MLP, quantize embeddings?, bits per weight, label)
def cfgs():
    i = lambda b, g, sym=True: (lambda w: int_rtn(w, b, g, sym))
    return {
        "fp8_tensor": (fp8, fp8, False, 8.0, "FP8 E4M3, one scale per tensor"),
        "fp8_block128": (lambda w: fp8(w, 128),) * 2 + (False, 8.0, "FP8 E4M3, 128x128 block scales (DeepSeek-V3 style)"),
        "int8_channel": (i(8, None),) * 2 + (False, 8.0, "INT8 RTN, one scale per output row"),
        "int4_channel": (i(4, None),) * 2 + (False, 4.0, "INT4 RTN, one scale per output row"),
        "int4_g128": (i(4, 128),) * 2 + (False, 4.125, "INT4 RTN, groups of 128 (FP16 scale)"),
        "int4_g32_asym": (i(4, 32, False),) * 2 + (False, 5.0, "INT4 RTN, groups of 32, scale + zero point"),
        "mxfp4": (mxfp4, mxfp4, False, 4.25, "MXFP4 (FP4 E2M1, blocks of 32, power-of-two scale)"),
        "nvfp4": (nvfp4, nvfp4, False, 4.5, "NVFP4 (FP4 E2M1, blocks of 16, FP8 scale)"),
        "int4_g128_embed": (i(4, 128),) * 2 + (True, 4.125, "INT4 g128, embeddings/lm_head quantized too"),
        "int3_g128": (i(3, 128),) * 2 + (False, 3.125, "INT3 RTN, groups of 128"),
        "mixed_attn8_mlp3": (i(8, 128), i(3, 128), False, None, "Mixed: attention INT8, MLP INT3 (g128)"),
        "int2_g64": (i(2, 64),) * 2 + (False, 2.25, "INT2 RTN, groups of 64"),
    }


def apply(model, orig, attn_fn, mlp_fn, embed):
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in LINEARS:
            w = orig[name]
            mod.weight.copy_((attn_fn if leaf in ATTN else mlp_fn)(w))
    emb = model.get_input_embeddings()
    emb.weight.copy_(int_rtn(orig["__embed__"], 4, 128) if embed else orig["__embed__"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--text", required=True)
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--cache", default="ref_cache")
    ap.add_argument("--out", default="quant_fidelity.json")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    orig = {n: m.weight.detach().clone() for n, m in model.named_modules() if n.rsplit(".", 1)[-1] in LINEARS}
    orig["__embed__"] = model.get_input_embeddings().weight.detach().clone()
    n_lin = sum(v.numel() for k, v in orig.items() if k != "__embed__")
    n_attn = sum(v.numel() for k, v in orig.items() if k.rsplit(".", 1)[-1] in ATTN)
    text = "\n".join(pd.read_parquet(a.text)["text"].tolist())
    tok.model_max_length = 10**9
    ids = tok(text, return_tensors="pt").input_ids[0]
    seqs = [ids[k * a.len:(k + 1) * a.len] for k in range(a.seqs)]
    key = hashlib.sha1(f"{a.model}|{a.len}|".encode() + torch.stack(seqs).numpy().tobytes()).hexdigest()[:12]
    cache = os.path.join(a.cache, key)
    os.makedirs(cache, exist_ok=True)
    probes = ["The cat sat on the", "The capital of France is", "def fibonacci(n):\n    if n <"]

    def run(label):
        lp = []
        for k, s in enumerate(seqs):
            f = f"{cache}/ref_{k}.pt"
            if label == "ref" and os.path.exists(f):
                lp.append(None); continue
            out = torch.log_softmax(model(s[None]).logits[0].float(), -1)
            if label == "ref":
                tmp = f"{f}.{os.getpid()}.tmp"; torch.save(out, tmp); os.replace(tmp, f); lp.append(None)
            else:
                lp.append(out)
        return lp

    def probe():
        res = []
        for p in probes:
            x = tok(p, return_tensors="pt").input_ids
            pr = torch.softmax(model(x).logits[0, -1].float(), -1)
            v, ix = pr.topk(5)
            res.append({"prompt": p, "top5": [[tok.decode([int(t)]), round(float(q), 4)] for q, t in zip(v, ix)]})
        return res

    results = {"model": a.model, "tokens": a.seqs * a.len, "seqs": a.seqs, "seq_len": a.len,
               "dataset": "wikitext-2-raw-v1 test", "linear_params": n_lin,
               "embed_params": orig["__embed__"].numel(), "configs": {}}
    t0 = time.time(); run("ref"); results["ref_probe"] = probe()
    ref_nll = []
    for k, s in enumerate(seqs):
        r = torch.load(f"{cache}/ref_{k}.pt")
        ref_nll.append(-r[:-1].gather(-1, s[1:, None]).squeeze(-1))
    results["ref_ppl"] = float(torch.exp(torch.cat(ref_nll).mean()))
    print(f"ref done {time.time()-t0:.0f}s ppl={results['ref_ppl']:.3f}", flush=True)
    for name, (fa, fm, emb, bpw, label) in cfgs().items():
        if a.only and name not in a.only.split(","):
            continue
        t0 = time.time(); apply(model, orig, fa, fm, emb)
        if bpw is None:
            bpw = (8.125 * n_attn + 3.125 * (n_lin - n_attn)) / n_lin
        kl, t1, t5, nll = [], [], [], []
        for k, s in enumerate(seqs):
            r = torch.load(f"{cache}/ref_{k}.pt")
            q = torch.log_softmax(model(s[None]).logits[0].float(), -1)
            kl.append((r.exp() * (r - q)).sum(-1))
            top_q = q.argmax(-1)
            t1.append(top_q == r.argmax(-1))
            t5.append((r.topk(5, -1).indices == top_q[:, None]).any(-1))
            nll.append(-q[:-1].gather(-1, s[1:, None]).squeeze(-1))
        kl = torch.cat(kl).clamp(min=0)
        res = {"label": label, "bpw": round(bpw, 3), "top1": float(torch.cat(t1).float().mean()),
               "top5": float(torch.cat(t5).float().mean()), "kld_mean": float(kl.mean()),
               "kld_median": float(kl.median()), "kld_p95": float(kl.quantile(0.95)),
               "kld_p99": float(kl.quantile(0.99)), "kld_max": float(kl.max()),
               "ppl": float(torch.exp(torch.cat(nll).mean())), "probe": probe(),
               "kld_hist": np.histogram(np.clip(np.log10(kl.numpy() + 1e-8), -8, 1.999), bins=20, range=(-8, 2))[0].tolist()}
        results["configs"][name] = res
        print(f"{name:<18} bpw={res['bpw']:<6} top1={res['top1']:.4f} top5={res['top5']:.4f} "
              f"kld={res['kld_mean']:.5f} med={res['kld_median']:.6f} ppl={res['ppl']:.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
