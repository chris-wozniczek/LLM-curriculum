export type Status = "ready" | "planned";
export interface Module {
  n: number;
  slug: string;
  title: string;
  q: string;
  c: string;
  status: Status;
  topics: string[];
  part: string;
}

export const parts = [
  { id: "basics", title: "Basics", blurb: "What a model is made of, in plain numbers." },
  { id: "inside", title: "Inside the model", blurb: "How text becomes probabilities, layer by layer." },
  { id: "files", title: "Files, formats, quantization", blurb: "What you download from Hugging Face and how people shrink it." },
  { id: "arch", title: "Architectures", blurb: "Real open models taken apart through their own configs." },
  { id: "train", title: "Training and changing models", blurb: "Pretraining, post-training and modifying weights." },
  { id: "use", title: "Running and judging", blurb: "Serving engines, evals and labs for your Mac." },
];

export const modules: Module[] = [
  { n: 0, slug: "map", part: "basics", title: "Map and how to use this", c: "var(--led)", status: "ready", q: "Where does each topic sit, and in what order should I read?", topics: ["Reading order", "Glossary and search", "How sources and measured numbers are marked"] },
  { n: 1, slug: "foundations", part: "basics", title: "Foundations", c: "var(--led)", status: "ready", q: "What are vectors, matrices and tensors, and what does training actually change?", topics: ["Numbers, vectors, matrices, tensors", "Matrix multiply as the core operation", "Loss, gradients, backprop, optimizers (SGD, Adam)", "Parameters vs activations", "Why GPUs: FLOPs and memory bandwidth"] },
  { n: 2, slug: "tokens", part: "inside", title: "Tokens and tokenizers", c: "var(--blue)", status: "ready", q: "How does text turn into numbers the model can read?", topics: ["BPE and byte-level BPE", "Vocab size and the embedding table", "Special tokens and chat templates", "Why token counts differ between models", "tokenizer.json vs tokenizer_config.json"] },
  { n: 3, slug: "transformer", part: "inside", title: "The transformer", c: "var(--blue)", status: "ready", q: "What happens inside one layer?", topics: ["Embeddings", "Self-attention: Q, K, V", "MHA, MQA, GQA, MLA", "MLP / FFN and SwiGLU", "RMSNorm, residual stream", "RoPE and position", "Logits, softmax, sampling (temperature, top-p, top-k, min-p)", "Linear attention and hybrid layers (Gated DeltaNet)"] },
  { n: 4, slug: "inference", part: "inside", title: "Inference, KV cache and context", c: "var(--blue)", status: "ready", q: "Why is the first token slow, and what eats memory in long chats?", topics: ["Prefill vs decode", "KV cache: what is stored and how big it gets", "Context length, RoPE scaling (YaRN), sliding windows", "KV cache quantization", "Batching, paged attention", "Speculative decoding, MTP", "Memory bandwidth as the speed limit"] },
  { n: 5, slug: "model-files", part: "files", title: "Anatomy of a model repo", c: "var(--amber)", status: "ready", q: "What are all those files on a Hugging Face model page?", topics: ["config.json line by line", "Tokenizer files and chat template", "safetensors: header, tensors, shards, index", "Counting parameters from the tensor list", "GGUF, MLX and EXL3 repos", "Model cards and licenses"] },
  { n: 6, slug: "number-formats", part: "files", title: "Number formats", c: "var(--amber)", status: "ready", q: "What is FP32, FP16, BF16, FP8, MXFP4, NVFP4, and why does it matter?", topics: ["Sign, exponent, mantissa", "FP32 vs FP16 vs BF16", "FP8 E4M3 and E5M2", "INT8, INT4", "Block formats: MXFP4, NVFP4", "Bits per weight and file size", "Mixed precision in training vs in quantization"] },
  { n: 7, slug: "quantization", part: "files", title: "Quantization", c: "var(--amber)", status: "ready", q: "How do people shrink a model, and how do they prove it still works?", topics: ["Round-to-nearest, scales, zero points", "Group size and block scales", "Outliers and sensitive tensors", "Mixed precision: what stays high", "GPTQ, AWQ, imatrix", "GGUF k-quants and i-quants", "EXL2 vs EXL3 (QTIP trellis)", "MLX quants", "Measuring: perplexity, KLD, top-1, top-5", "Our measured results on Qwen3-0.6B"] },
  { n: 8, slug: "moe", part: "arch", title: "Mixture of Experts", c: "var(--pink)", status: "ready", q: "What are experts, and why does a 30B-A3B run like a 3B?", topics: ["Router and top-k", "Shared experts", "Total vs active parameters", "Load balancing", "Expert offloading", "Expert pruning (REAP-style)"] },
  { n: 9, slug: "architectures", part: "arch", title: "Architectures compared", c: "var(--pink)", status: "ready", q: "How is DeepSeek built differently from Qwen, Kimi or gpt-oss?", topics: ["Llama / Qwen3 dense", "Qwen3 MoE and Qwen3-Next", "Qwen3.5 and Ornith (hybrid linear attention)", "DeepSeek-V3 (MLA, shared expert, MTP)", "Kimi K2", "gpt-oss (MXFP4, sliding window)", "Gemma", "Muse Spark Open"] },
  { n: 10, slug: "pretraining", part: "train", title: "Pretraining", c: "var(--violet)", status: "ready", q: "How is a model trained from zero, and what does it cost?", topics: ["Data: crawl, filtering, dedup, mixes", "Training a tokenizer", "Compute: 6ND rule", "Scaling laws (Chinchilla)", "Mixed-precision training (BF16, FP8)", "Parallelism: data, tensor, pipeline, expert", "nanoGPT / nanochat"] },
  { n: 11, slug: "post-training", part: "train", title: "Post-training", c: "var(--violet)", status: "ready", q: "How does a base model become a chat or coding agent, in theory and in practice?", topics: ["SFT and chat templates", "Full fine-tune vs LoRA vs QLoRA", "DPO and preference data", "RL: PPO, GRPO, verifiable rewards", "Distillation from a teacher", "Reasoning traces", "Otter as a case study", "Tooling: TRL, Unsloth, axolotl, MLX-LM"] },
  { n: 12, slug: "modifying", part: "train", title: "Modifying models", c: "var(--violet)", status: "ready", q: "How do merges, abliteration and steering change a model without training?", topics: ["Merges: linear, SLERP, TIES, DARE (mergekit)", "Abliteration: finding and removing the refusal direction", "Activation steering", "Expert pruning and layer pruning", "Upscaling (frankenmerges)"] },
  { n: 13, slug: "recipes", part: "use", title: "Recipes", c: "var(--gold)", status: "ready", q: "What is a recipe on Hugging Face, and how do I read one?", topics: ["Quant recipes (llm-compressor, EXL3, GGUF)", "Merge recipes", "Training recipes", "Serving recipes (vLLM, llama.cpp)"] },
  { n: 14, slug: "evaluation", part: "use", title: "Evaluation", c: "var(--gold)", status: "ready", q: "How do I tell if a model is actually good at my work?", topics: ["Benchmarks and what they measure", "Harnesses: Terminal-Bench, SWE-bench, BFCL", "Contamination", "Practical task evals (pagoda, todo app)", "Agent reliability: tool-call failures, retries", "Statistics: variance, pass@k"] },
  { n: 15, slug: "serving", part: "use", title: "Serving engines", c: "var(--gold)", status: "ready", q: "vLLM, SGLang, llama.cpp, MLX, ExLlamaV3: which one when?", topics: ["vLLM and SGLang", "llama.cpp and GGUF", "MLX / oMLX on Apple Silicon", "ExLlamaV3 and TabbyAPI", "OpenAI-compatible APIs", "Throughput vs latency"] },
  { n: 16, slug: "labs", part: "use", title: "Labs for your Mac", c: "var(--gold)", status: "ready", q: "Which scripts can I run myself to see all of this?", topics: ["Read a safetensors header", "Measure top-1 / KLD of a quant", "Quantize with llama.cpp and MLX", "LoRA fine-tune with MLX-LM", "Abliterate a small model", "Build a nanoGPT"] },
];

export const bySlug = Object.fromEntries(modules.map((m) => [m.slug, m]));
export const ready = modules.filter((m) => m.status === "ready");
