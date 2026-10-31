# LLM Curriculum

An interactive curriculum on how LLMs are built, stored, quantized, post-trained and served, taught by reading real open models (Qwen, Ornith, DeepSeek, Kimi, gpt-oss) from their own files.

Static [Astro](https://astro.build) site, no backend. Search is [Pagefind](https://pagefind.app), built at build time.

```bash
npm install
npm run dev      # http://localhost:4321 (search needs a build first)
npm run build    # astro build + pagefind index -> dist/
npm run preview
npm run check    # astro check (types)
```

## Layout

- `src/data/curriculum.ts` – module list, order, status (`ready` / `planned`)
- `src/content/<slug>.mdx` – lesson text; a module without a file renders as an outline
- `src/data/glossary.ts` – glossary; use `<T k="key">word</T>` in MDX for hover definitions
- `src/components/figs/` – interactive figures
- `src/data/models/*.json` – `config.json` + tensor-name patterns pulled from Hugging Face
- `src/data/quant-fidelity.json` – measured results from `labs/quant_fidelity.py`

## Labs

Python scripts that produce every number marked "measured" on the site.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch transformers safetensors pandas pyarrow numpy
python labs/inspect_safetensors.py path/to/model.safetensors
python labs/quant_fidelity.py --model Qwen/Qwen3-0.6B --text wikitext2-test.parquet --seqs 16 --len 512
```

`quant_fidelity.py` runs on CPU (about 10 minutes for Qwen3-0.6B) and caches the reference log-probs (~5 GB for the default settings) in `ref_cache/`.
