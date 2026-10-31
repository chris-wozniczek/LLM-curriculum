"""Tokenize the same texts with several real tokenizers and render real chat templates.

python labs/tokenizers_compare.py --dir tok --out tokens.json
--dir holds one folder per model with that repo's tokenizer files
(tokenizer.json, tokenizer_config.json, chat_template.jinja, ...),
without config.json (some repos' configs confuse AutoTokenizer).
"""
import argparse, json, os
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--dir", default="tok")
p.add_argument("--out", default="tokens.json")
a = p.parse_args()

TEXTS = {
    "English": "Quantization makes large language models smaller and faster.",
    "Polish": "Kwantyzacja sprawia, że duże modele językowe są mniejsze i szybsze.",
    "Chinese": "大型语言模型正在改变软件开发。",
    "Python": "def add(a, b):\n    return a + b\n",
    "Numbers": "In 2025, 3.14159 × 1,000,000 = 3141590.",
    "Emoji": "Ship it 🚀🔥",
}
MSGS = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}]

out = {"texts": TEXTS, "models": []}
for name in sorted(os.listdir(a.dir)):
    d = os.path.join(a.dir, name)
    if not os.path.isdir(d): continue
    kw = {"fix_mistral_regex": True} if "mistral" in name.lower() else {}
    try: tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True, **kw)
    except Exception as e: print("skip", name, type(e).__name__, str(e)[:120]); continue
    m = {"name": name, "vocab": len(tok), "special": len(tok.all_special_tokens), "counts": {}, "pieces": {}}
    for k, t in TEXTS.items():
        ids = tok.encode(t, add_special_tokens=False)
        m["counts"][k] = len(ids)
        m["pieces"][k] = [{"id": i, "s": tok.decode([i])} for i in ids]
    try: m["chat"] = tok.apply_chat_template(MSGS, tokenize=False)
    except Exception as e: m["chat"] = None; print("no template", name, str(e)[:100])
    out["models"].append(m)
    print(name, m["vocab"], m["counts"])
json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
