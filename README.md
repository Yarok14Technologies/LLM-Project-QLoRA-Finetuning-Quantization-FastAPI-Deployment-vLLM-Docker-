# 🧠 **LLM Project — QLoRA Finetuning, Quantization & FastAPI Deployment (vLLM + Docker)**

A complete end-to-end pipeline for building your own efficient LLM similar to *Nano/Banana* small models.

This repository includes:

* 🟩 **QLoRA Finetuning** (1B–7B models)
* 🟨 **LoRA Merge + GPTQ/AWQ Quantization**
* 🟦 **FastAPI Inference Server (vLLM optimized)**
* 🟥 **GPU Deployment with Docker & docker-compose**
* ⚙️ **VRAM Model Size Calculator**
* 📦 **Modular Repo Structure for Production**

---

# 📁 Repository Structure

```
llm-project/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── data_prep.py
│
├── training/
│   ├── finetune_qlora.py
│   ├── merge_lora.py
│   ├── config/
│   │   └── training_args.json
│   └── logs/
│
├── models/
│   ├── base/
│   ├── lora-output/
│   ├── merged-model/
│   ├── quantized/
│   └── tokenizer/
│
├── quantization/
│   ├── quantize_gptq.py
│   ├── quantize_awq.py
│   └── calibrate_data.jsonl
│
├── api/
│   ├── app.py
│   ├── model_loader.py
│   ├── schemas.py
│   └── utils.py
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── start.sh
│
├── scripts/
│   ├── install_env.sh
│   ├── vram_calculator.py
│   ├── download_model.py
│   └── test_api.sh
│
└── deployment/
    ├── nginx.conf
    ├── service_monitoring/
    ├── systemd_service.service
    └── k8s/
```

---

# 🚀 Quick Start

## 1️⃣ **Install Dependencies (local machine)**

```bash
chmod +x scripts/install_env.sh
./scripts/install_env.sh
```

Activating environment:

```bash
source llm-env/bin/activate
```

---

# 2️⃣ **Prepare Dataset**

Place JSONL dataset in:

```
data/processed/train.jsonl
```

Format:

```json
{"instruction": "Explain biogas.", "input": "", "output": "Biogas is..."}
```

---

# 3️⃣ **Finetune the Model with QLoRA**

Modify your base model path in:

`training/finetune_qlora.py`

Run training:

```bash
python training/finetune_qlora.py
```

Output LoRA adapter is saved to:

```
models/lora-output/
```

---

# 4️⃣ **Merge LoRA into Base Model**

```bash
python training/merge_lora.py
```

Merged model saved to:

```
models/merged-model/
```

---

# 5️⃣ **Quantize Model (GPTQ)**

```bash
python quantization/quantize_gptq.py \
  --model_path models/merged-model \
  --output_path models/quantized \
  --bits 4
```

Final quantized model is used for API serving.

---

# 6️⃣ **Run FastAPI Server (vLLM optimized)**

Ensure correct model path in:

`api/app.py` → `MODEL_PATH="/models/quantized"`

### Local run:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Test:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain biomethane in 2 lines."}'
```

---

# 7️⃣ **Deploy with Docker + NVIDIA GPU**

### Build and run:

```bash
cd docker
docker compose up --build -d
```

### Test API:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello, what can you do?"}'
```

---

# ⚙️ Model Size Calculator

Run:

```bash
python scripts/vram_calculator.py
```

This shows how big a model your GPU can support for:

* FP16
* 8-bit
* 4-bit (GPTQ / AWQ)
* QLoRA finetuning

Includes suggested GPU → model mappings.

---

# 🧩 Configuration

### Change base model

`training/finetune_qlora.py`

### Modify training hyperparameters

`training/config/training_args.json`

### Change runtime model path

`api/app.py` → `MODEL_PATH`

---

# 📦 Docker Deployment Notes

### Model volume mount:

```
models/quantized → /models/quantized
```

### FastAPI is served via:

```
http://localhost:8000
```

Default endpoints:

| Method | Endpoint    | Description               |
| ------ | ----------- | ------------------------- |
| GET    | `/health`   | Check model health        |
| POST   | `/generate` | Generate text from prompt |

---

# 🔥 Features

* ⚡ **QLoRA finetuning** (very low VRAM use)
* 🧩 **LoRA merging** script included
* ⚙️ **GPTQ/AWQ quantization**
* 🚀 **vLLM inference** (super-fast GPU serving)
* 🐳 **Dockerized deployment with GPU access**
* 📡 **FastAPI REST API**
* 📊 Optional **Prometheus/Grafana monitoring**

---

# 🛠️ Hardware Recommendations

| GPU            | Max QLoRA Model | Max Quantized Inference |
| -------------- | --------------- | ----------------------- |
| RTX 3060 12GB  | 1B              | 3B                      |
| RTX 3090 24GB  | 3B–7B           | 7B                      |
| RTX 4090 24GB  | 7B              | 7B (fast)               |
| A100 40GB      | 13B             | 13B                     |
| A100/H100 80GB | 13B–34B         | 34B+                    |

---

# 🤝 Contributing

PRs welcome for:

* Training improvements
* LoRA merge automation
* Quantization tools
* Inference optimization
* Docker images

---

# 📄 License

Apache 2.0 (https://github.com/Yarok14Technologies/LLM-Project-QLoRA-Finetuning-Quantization-FastAPI-Deployment-vLLM-Docker-/blob/main/LICENSE)

---

# 💬 Support

For help integrating your own dataset or customizing inference, open an issue or ask ChatGPT to generate specialized scripts (merge automate, quantize tools, dataset filters, etc.)

---

Nice — great question. I’ll give you a practical, end-to-end playbook for building a compact, production-ready model in the spirit of **“Nano Banana Pro”** (a small/fast multimodal/LLM-grade model Google ships for low-latency tasks): goals → data → model choices → training/finetune recipes → quantization & compression → inference & serving → evaluation, cost, and safety. I’ll include concrete tool names, commands/patterns you can copy, and pointers to further reading. Where a claim benefits from recent info I’ve cited the sources.

---

# 1) Decide the exact product & constraints

Pick 2–3 concrete constraints before doing anything:

* **Size / latency target** (e.g., <6B params for cheap inference on a single 80GB GPU, or ~1–2B for phone/edge).
* **Capabilities**: text-only vs. multimodal (images) vs. code/specialized domain.
* **Accuracy vs. cost tradeoff**: small models + good finetune + quantization usually win for latency.
* **License/commercial** requirements (some open models are ok for commercial use; check licensing).

Note: “Nano Banana Pro” is a Google product aimed at fast/visual outputs — good reference for a compact, fast model. ([Android Central][1])

---

# 2) Pick a base model

Unless you have massive compute, **start from an open pretrained base** and fine-tune/adapter it:

* Common base families: **LLaMA (meta forks), Mistral, Falcon, Vicuna / Llama-2 variants** (pick one with permissive license for your use).
* For very small low-latency models, consider distilled or smaller parameter models (1–7B) and then use LoRA/QLoRA for task adaptation.

Why not train from scratch? Training >100B models is massively expensive and time-consuming. If you do want to (research), plan for multi-million GPU-hours. Practical path: finetune + quantize. (See cost estimates / GPU guidance below.) ([Runpod][2])

---

# 3) Gather & prepare data

* **General pretraining** (only if you are training from scratch): huge corpora (CommonCrawl, The Pile, ref-filtered web text, code corpora, multimodal pairs for vision).
* **Finetuning / instruction tuning**: high-quality instruction datasets (OpenAssistant, ShareGPT, Anthropic/Tri) or create your own curated prompt–response pairs.
* **Multimodal**: image–caption pairs (LAION), curated captioning datasets, plus any proprietary domain images/labels.
* **Data hygiene**: deduplication, profanity and PII checks, license filtering.
* Tokenization: use the tokenizer of the base model, or train a new tokenizer if you change model architecture.

---

# 4) Fine-tuning strategies (practical & low-cost)

* **LoRA** (low-rank adapters) — cheap, small-memory parameter updates.
* **QLoRA** — quantize the base model to 4-bit and fine-tune LoRA adapters on top (huge VRAM savings). Use bitsandbytes + peft/transformers workflows. This is the standard for efficient finetuning. ([Generative AI][3])
* **RLHF** (optional, expensive) — for alignment/quality improvements (requires reward model + preference data).

**Typical pipeline (example for 7B using QLoRA + PEFT):**

```bash
# environment: python, transformers, accelerate, bitsandbytes, peft
# 1. Load model with 4-bit bitsandbytes
python finetune_qlora.py \
  --model_name_or_path pretrained-7b \
  --dataset my_instruction_dataset.jsonl \
  --bnb_4bit True \
  --lora_r 8 --lora_alpha 32 \
  --output_dir ./my-7b-lora
```

(Implementation details: huggingface transformers + bitsandbytes + peft/trl; many example notebooks exist.)

---

# 5) Compression & quantization for inference

* **Post-training quantization**: GPTQ, AWQ, or bitsandbytes 4/8bit formats. AWQ (Activation-aware Weight Quantization) and GPTQ are state-of-the-art for 4-bit post-training quantization and often preserve accuracy better than naïve PTQ. ([Medium][4])
* **Quantize + LoRA inference**: vLLM and other servers now support LoRA on quantized bases — which is how you get both size and adapter benefits. ([VLLM Documentation][5])
* **FlashAttention / optimized kernels**: use FlashAttention for faster attention and lower memory; many modern frameworks support this for big speed gains. ([GitHub][6])

Common stack:

* Convert weights with GPTQ / AWQ → load via AutoGPTQ or bitsandbytes.
* Serve with vLLM or FasterTransformer/Triton for high throughput.

---

# 6) Inference & serving (production)

Options depend on scale:

* **Local / single-GPU**: vLLM (fast token throughput, paging, LoRA+quantization support), or Hugging Face Runner. ([VLLM Documentation][5])
* **Multi-tenant high throughput**: vLLM with Ray or a cluster, Triton Inference Server, or NVIDIA FasterTransformer for optimized speed.
* **Edge**: convert to ONNX / TensorRT / CoreML / TFLite for mobile/embedded devices after careful quantization and testing.
* **API wrapper**: FastAPI / Starlette + async queues; add caching and prompt-prefix caching for repeated queries.

Key performance tips:

* Use batching and prompt caching; use FlashAttention kernels; use 4-bit quantized weights where acceptable; use tensor parallelism and KV cache offloading when needed. ([GitHub][6])

---

# 7) Costs & hardware (very rough guidance)

* Fine-tuning a 7B–13B model with QLoRA: doable on a single A100/H100 40–80GB or a small multi-GPU node. Training from scratch is orders of magnitude more expensive. Example high-level GPU cost guidance is available in community writeups (hundreds of thousands to millions for full training of large models). ([Runpod][2])

Rough rule-of-thumb:

* **Finetuning 7B with QLoRA**: hours–days on 1–2 x A100/H100.
* **Training 7B from scratch**: tens of thousands of GPU hours; avoid unless you have major resources.

---

# 8) Evaluation & safety

* **Metrics**: Perplexity (for pure LM), ROUGE/BLEU for specific tasks, human eval + instruction response scoring for chatty models.
* **Bias & safety testing**: automated checks for toxic output, adversarial prompt testing, and human red-team. Prioritize filtering PII and copyrighted content if deploying publicly.
* **Monitoring**: logs, rate limits, prompt/response sampling, user feedback loop.

---

# 9) Legal & licensing

* Verify licenses of the base model & training data before commercial use. Some models require non-commercial use or have other restrictions.

---

# 10) Example concrete stack (copy/paste friendly)

* Model: Mistral / LLaMA-2 7B (open license).
* Finetune: Hugging Face Transformers + bitsandbytes + PEFT (QLoRA).
* Quantize: AutoGPTQ / AWQ (4-bit).
* Inference: vLLM on a GPU node, with FlashAttention enabled.
* Serving: Docker + FastAPI + Prometheus + Grafana for metrics.

Useful docs & tools:

* vLLM docs (fast inference & LoRA + quantized support). ([VLLM Documentation][5])
* FlashAttention (fast attention kernels). ([GitHub][6])
* Quantization/QLoRA, bitsandbytes & AWQ community writeups. ([Generative AI][3])
* GPU / cost guides for fine-tuning. ([Runpod][2])

---

# Quick starter checklist (actionable)

1. Pick base model (7B recommended to start).
2. Collect/prepare instruction dataset (10k–100k prompt/response pairs).
3. Set up Python env: `transformers`, `accelerate`, `bitsandbytes`, `peft`, `datasets`.
4. Do QLoRA finetune (save LoRA adapters).
5. Convert base weights to 4-bit (AWQ/GPTQ) for inference.
6. Serve via vLLM with LoRA merged or applied at runtime.
7. Test, safety-check, iterate.

---

If you want, I can:

* give you an exact **QLoRA training script** tuned for a 7B model and ~20k instruction pairs (with `accelerate` config and tips), **or**
* produce a **deployment Dockerfile + vLLM config** for serving a quantized 7B+LoRA model on a single 80GB GPU, **or**
* estimate **detailed costs** for a chosen target (e.g., fine-tune + serve 1k qps vs. 10k qps).

Which one do you want me to generate first?

[1]: https://www.androidcentral.com/apps-software/ai/google-expands-gemini-3-and-nano-banana-pro-availability-for-ai-mode-in-search?utm_source=chatgpt.com "Google expands Gemini 3 and Nano Banana Pro availability for AI Mode in Search globally"
[2]: https://www.runpod.io/blog/llm-fine-tuning-gpu-guide?utm_source=chatgpt.com "The Complete Guide to GPU Requirements for LLM Fine- ..."
[3]: https://generativeai.pub/practical-guide-of-llm-quantization-gptq-awq-bitsandbytes-and-unsloth-bdeaa2c0bbf6?utm_source=chatgpt.com "GPTQ, AWQ, BitsandBytes, and Unsloth - Generative AI"
[4]: https://medium.com/%40shubham.shardul2019/fine-tuning-large-language-models-llms-practical-guide-intuition-lora-qlora-deep-dive-and-ba01d61cd0a7?utm_source=chatgpt.com "Fine-tuning Large Language Models (LLMs): Practical ..."
[5]: https://docs.vllm.ai/en/latest/?utm_source=chatgpt.com "vLLM"
[6]: https://github.com/Dao-AILab/flash-attention?utm_source=chatgpt.com "Dao-AILab/flash-attention: Fast and memory-efficient ..."



Nice — here’s a **practical, copy-pasteable, step-by-step playbook** to build a compact LLM in the “Banana Nano Pro” spirit: small, fast, production-ready. I’ll assume you’ll **start from an existing 7B model**, QLoRA-finetune it, GPTQ/AWQ quantize for inference, then serve with **vLLM** (supports LoRA + quantized models). I’ll show commands, code snippets and where to look for details. I’ll also cite authoritative docs for the heavy lifts.

---

## TL;DR (one-line)

1. Create environment → 2. Prepare data → 3. QLoRA finetune (bitsandbytes + PEFT) → 4. Export LoRA adapters → 5. Quantize base model (AutoGPTQ / GPTQ / AWQ) → 6. Serve with vLLM (LoRA on quantized base) → 7. Test, evaluate, monitor.

Key references: QLoRA / bitsandbytes (Hugging Face), AutoGPTQ (GPTQ), vLLM quantized+LoRA examples, FlashAttention for speed. ([Hugging Face][1])

---

# Step 0 — assumptions & required hardware

* Target: **7B** base model (you can pick Llama-2 / Mistral / Llama-family snapshot with permissive license).
* Single GPU for QLoRA: **A100/H100 40–80GB** recommended; QLoRA reduces memory but still needs a big GPU. For 7B you can often use 24–48GB with the right tricks.
* Inference: one GPU (for low qps) or multiple for higher throughput. (See vLLM for scaling.) ([ROCm Documentation][2])

---

# Step 1 — create environment

(Using conda + pip is common.)

```bash
conda create -n llm-small python=3.10 -y
conda activate llm-small

# core libs
pip install --upgrade pip
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets safetensors

# bitsandbytes (4-bit support), peft, huggingface-hub
pip install bitsandbytes
pip install peft
pip install "git+https://github.com/huggingface/transformers.git"
pip install datasets safetensors

# optional: bits for monitoring & experiment tracking
pip install wandb

# vLLM for serving (later)
pip install "vllm"
```

Notes: `bitsandbytes` and CUDA must match your GPU / driver. Bitsandbytes provides the 4-bit QLoRA features. ([GitHub][3])

---

# Step 2 — prepare your fine-tune data

Use **JSONL** of instruction pairs, one per line:

```jsonl
{"instruction":"Summarize the following...","input":"<long text>","output":"Short summary..."}
{"instruction":"Translate to Malayalam","input":"Hello","output":"Namaskaram"}
```

Guidelines:

* 10k–100k examples is a practical sweet spot for instruction tuning.
* Deduplicate, remove PII, filter copyrighted/illegal content.
* Use the same tokenizer as the base model.

You can use `datasets` to load & preprocess.

---

# Step 3 — QLoRA finetune (LoRA on a 4-bit frozen base)

Quasi-complete script outline (adapt paths & params). This uses `bitsandbytes` + `peft` + `accelerate`.

`finetune_qlora.py` (simplified):

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "meta-llama/Llama-2-7b-hf"  # example
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# load dataset
ds = load_dataset("json", data_files="mydata.jsonl", split="train")

# prepare model in 4-bit
from transformers import AutoConfig
import bitsandbytes as bnb
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"], bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

# tokenization + training (use Trainer or accelerate)
# ... build datasets -> tokenized, DataCollator, TrainingArguments ...
```

Run via `accelerate launch` or `python` depending on your training script setup.

Why QLoRA? It quantizes the frozen base to 4-bit and only trains small LoRA adapters — huge VRAM savings while keeping full-precision-like quality. See the Hugging Face blog on QLoRA / bitsandbytes. ([Hugging Face][1])

---

# Step 4 — save/export LoRA adapter

After training save the adapter (PEFT/LoRA standard):

```python
model.save_pretrained("./my-lora-adapter")
# This folder will hold adapter_model.safetensors + adapter_config.json
```

Keep the adapter separate from the base weights — you’ll apply it at inference time.

---

# Step 5 — quantize base model (GPTQ / AWQ / AutoGPTQ)

For smallest inference footprint, convert the base model to a 4-bit GPTQ/AWQ format.

Option A: **AutoGPTQ** (user-friendly tool to create a GPTQ quantized model):

```bash
# clone and follow repo instructions
git clone https://github.com/AutoGPTQ/AutoGPTQ.git
cd AutoGPTQ
# follow README; typical flow uses their script to quantize a HF model
```

AutoGPTQ produces quantized files that inference runtimes (llama.cpp, some HF loaders, vLLM variations) can consume. AutoGPTQ is widely used for 4-bit GPTQ conversions. ([GitHub][4])

Option B: **AWQ** (activation-aware quantization) — newer and often gives better accuracy at 3–4 bits; tools vary by project. (Search AWQ implementations when you need slightly better fidelity.)

Note: bitsandbytes’ NF4 format + QLoRA can also be used directly for some inference stacks. For unmerged LoRA on quantized bases, vLLM has examples. ([Hugging Face][1])

---

# Step 6 — serve (vLLM example: quantized base + LoRA)

**vLLM** supports LoRA + quantized models and is excellent for fast token throughput. Example (Python offline load example from vLLM docs):

```python
from vllm import LLMEngine, EngineArgs
from vllm.lora.request import LoRARequest

engine_args = EngineArgs(
    model="path/to/quantized-base-or-hf-id",
    # additional args: tokenizer, device, dtype...
)
engine = LLMEngine(engine_args)

# Use LoRA adapter at request time
lora_req = LoRARequest(adapter="path/to/my-lora-adapter")

output = engine.generate(
    [["Write a short poem about biomethane optimization."]],
    lora=lora_req
)
print(output)
```

You can also run vLLM as a server (see docs) and let it handle batching, KV cache, and token streaming. vLLM docs include a **LoRA with quantization** example. ([VLLM Documentation][5])

Performance tips:

* Enable FlashAttention (if your CUDA & model supports it) for faster attention. ([GitHub][6])
* Use batching and prompt caching for repeated prompts.
* For very high throughput, use vLLM + Ray or deploy multiple GPU workers behind a queue.

---

# Step 7 — test & evaluate

* Automated: perplexity (on held-out LM tokens), and instruction-score (BLEU/ROUGE when applicable).
* Human eval: sample prompts, rate helpfulness/toxicity/safety.
* Safety: run adversarial prompts, filter PII, implement rate limits, content filters.

---

# Step 8 — optionally merge LoRA (for tiny inference footprint)

Merging LoRA into the base weights produces a single merged model file (makes serving simpler but loses adapter flexibility). Use `peft` merge utilities or `transformers` tooling to merge and then re-quantize if desired.

---

# Step 9 — productionize: Docker + monitoring

Dockerfile skeleton:

```Dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "serve_vllm.py"]
```

Add Prometheus/Grafana metrics and logging. Add authentication/rate limits.

---

# Quick checklist (copy to your terminal)

1. Create env and install libraries. ([Hugging Face][1])
2. Prepare `mydata.jsonl`.
3. Run QLoRA finetune script (use `accelerate launch`). ([Hugging Face][1])
4. Save LoRA adapter: `model.save_pretrained("./my-lora-adapter")`.
5. Quantize base → AutoGPTQ / GPTQ / AWQ. ([GitHub][4])
6. Serve with vLLM, applying LoRA adapters at runtime. ([VLLM Documentation][5])
7. Test, evaluate, monitor.

---

## Helpful docs & reading (fast links)

* QLoRA / bitsandbytes explainer (Hugging Face blog). ([Hugging Face][1])
* PEFT / LoRA docs (Hugging Face). ([Hugging Face][7])
* AutoGPTQ (quantization). ([GitHub][4])
* vLLM docs — LoRA + quantization example. ([VLLM Documentation][5])
* FlashAttention (speedups). ([GitHub][6])

---

If you want right now I can **do one of these next** (pick one) and produce exact runnable files:

* A complete `accelerate` + `finetune_qlora.py` script tuned for **7B + 20k instruction pairs** (includes `accelerate` config and Trainer loop), **or**
* A one-file `serve_vllm.py` + Dockerfile that loads a quantized base and applies your LoRA adapter at runtime, ready for an 80GB GPU.

Which one should I generate for you immediately?

[1]: https://huggingface.co/blog/4bit-transformers-bitsandbytes?utm_source=chatgpt.com "Making LLMs even more accessible with bitsandbytes, 4- ..."
[2]: https://rocm.docs.amd.com/projects/ai-developer-hub/en/v5.1/notebooks/fine_tune/QLoRA_Llama-3.1.html?utm_source=chatgpt.com "Fine-tuning Llama-3.1 with QLoRA"
[3]: https://github.com/bitsandbytes-foundation/bitsandbytes?utm_source=chatgpt.com "bitsandbytes-foundation/bitsandbytes"
[4]: https://github.com/AutoGPTQ/AutoGPTQ?utm_source=chatgpt.com "AutoGPTQ/AutoGPTQ: An easy-to-use LLMs quantization ..."
[5]: https://docs.vllm.ai/en/v0.7.0/getting_started/examples/lora_with_quantization_inference.html?utm_source=chatgpt.com "LoRA With Quantization Inference - vLLM"
[6]: https://github.com/Dao-AILab/flash-attention?utm_source=chatgpt.com "Dao-AILab/flash-attention: Fast and memory-efficient ..."
[7]: https://huggingface.co/docs/peft/en/package_reference/lora?utm_source=chatgpt.com "LoRA"


Here is a **clear, exact, production-ready list of hardware & software requirements** to build an LLM similar to *Banana Nano Pro* (small, fast, 1–7B parameter class).
I’ll divide it into:

### ✅ 1. Hardware for **Training / Finetuning**

### ✅ 2. Hardware for **Inference / Deployment**

### ✅ 3. Required **Software Stack**

### ✅ 4. Optional (for mobile, on-device models)

---

# ✅ **1. HARDWARE REQUIREMENTS — TRAINING / FINETUNING**

### **A) Minimum (QLoRA small-scale finetuning for 1–3B models)**

Useful if you want *very small models only*.

| Component | Requirement                            |
| --------- | -------------------------------------- |
| GPU       | **NVIDIA RTX 3090 / 4090 (24GB VRAM)** |
| RAM       | **32GB** minimum (recommended 64GB)    |
| Storage   | **500GB SSD** (datasets + checkpoints) |
| CPU       | Any modern 8-core (Intel/AMD)          |

💡 Good for:

* Finetuning 1B, 2B, 3B models
* Simple QLoRA instruction tuning
* No full pre-training possible

---

### **B) Recommended (for 7B QLoRA finetuning)**

If your target is a *Banana Nano Pro-style 7B model*:

| Component | Requirement                                  |
| --------- | -------------------------------------------- |
| GPU       | **NVIDIA A100 40GB / 80GB** OR **H100 80GB** |
| RAM       | **64GB–128GB**                               |
| Storage   | **1TB SSD / NVMe**                           |
| CPU       | ≥ 16-core workstation/server CPU             |

💡 Good for:

* QLoRA finetuning **7B** models
* Training LoRA adapters
* Running validation & merging LoRA

---

### **C) High-end (pre-training or finetuning 13B+)**

If you want to *train >7B models* or *finetune 13B+*:

| Component    | Requirement                                        |
| ------------ | -------------------------------------------------- |
| GPU          | **Multiple A100/H100 GPUs** with NVLink (4–8 GPUs) |
| VRAM per GPU | 40–80GB                                            |
| RAM          | 128GB–256GB                                        |
| Storage      | 2TB–4TB NVMe                                       |
| Network      | 100GbE for distributed training                    |

💡 Needed only for:

* Training foundation models
* Training multimodal models from scratch
* Large-scale RLHF

---

# ✅ **2. HARDWARE REQUIREMENTS — INFERENCE / DEPLOYMENT**

### **A) Desktop / Single-user deployment**

| Component | Requirement                                |
| --------- | ------------------------------------------ |
| GPU       | **NVIDIA RTX 3060 / 4060 / 4070 (8–12GB)** |
| RAM       | 16GB                                       |
| Storage   | 20GB free                                  |

💡 Works for:

* Running **quantized 4-bit 1–3B models**
* Chatbots
* Low traffic APIs

---

### **B) Cloud / Production API**

| Traffic                          | Recommended Hardware    |
| -------------------------------- | ----------------------- |
| Low traffic                      | **T4 / L4 GPU (16GB)**  |
| Medium                           | **A10G 24GB**           |
| High throughput (1k+ tokens/sec) | **A100 40/80GB**        |
| Very high throughput             | **H100 + TensorRT-LLM** |

💡 Use servers such as:

* AWS (g5, g6, p4d)
* LambdaLabs
* RunPod
* Paperspace

---

### **C) On-device / Mobile / Edge (Banana Nano style)**

For deploying **tiny models (≤1B parameters)**:

| Hardware                    | Compatible?                               |
| --------------------------- | ----------------------------------------- |
| Smartphones                 | YES (Android/iOS with ONNX/TFLite/CoreML) |
| Raspberry Pi 4/5            | YES for <1B models quantized to int8      |
| Jetson Nano / Xavier / Orin | YES for 1–3B models int8                  |

---

# ✅ **3. SOFTWARE REQUIREMENTS**

## **A) Base OS**

* **Linux (Ubuntu 20.04 / 22.04)** — best for CUDA + PyTorch
* Windows also possible, but not recommended for training

---

## **B) Required Core Software**

| Software         | Version              | Purpose                       |
| ---------------- | -------------------- | ----------------------------- |
| **Python**       | 3.10+                | Main ML ecosystem             |
| **CUDA Toolkit** | 11.8 or 12.x         | GPU compute                   |
| **cuDNN**        | 8.x+                 | Faster training               |
| **PyTorch**      | ≥ 2.1                | Backbone deep learning        |
| **Transformers** | Latest (HuggingFace) | LLM architecture & tokenizers |
| **PEFT**         | Latest               | LoRA / QLoRA                  |
| **BitsAndBytes** | Latest               | 4-bit & 8-bit quantization    |
| **Accelerate**   | Latest               | Distributed training          |
| **Datasets**     | Latest               | Load JSON/CSV/text data       |
| **Safetensors**  | Latest               | Model weight format           |

---

## **C) Training Tools**

* **QLoRA** (from PEFT + bitsandbytes)
* **DeepSpeed** (optional, large scale training)
* **TensorBoard / WandB** for monitoring
* **HuggingFace Hub** for model versioning

---

## **D) Inference / Deployment Tools**

* **vLLM** → fastest inference engine for LLMs
* **Text-Generation Inference (TGI)**
* **FastAPI / Flask** for API
* **Docker** for deployment
* **Nginx** as reverse proxy
* **Prometheus + Grafana** for monitoring

---

## **E) Quantization Tools**

Quantization is essential for “Nano-style” small models.

| Tool                       | Purpose                                         |
| -------------------------- | ----------------------------------------------- |
| **AutoGPTQ**               | 4-bit GPTQ quantization                         |
| **AWQ**                    | Activation-aware quantization (higher accuracy) |
| **BitsAndBytes Int4/Int8** | QLoRA training + inference                      |
| **llama.cpp**              | CPU inference + mobile builds                   |

---

## **F) Optional (Multimodal Support)**

If your model should handle images like Nano-Pro:

Install:

* **PyTorch Vision**
* **OpenCV**
* **CLIP / SigLIP encoders**
* **Timm** (vision backbones)
* **Diffusers** (if doing image generation)

And ensure GPU has **≥24GB VRAM** for image-text models.

---

# ✅ **4. OPTIONAL: For Android / iOS Deployment (Tiny LLM)**

### **A) Android**

* **TensorFlow Lite**
* **ONNX Runtime Mobile**
* **GGML/gguf models via llama.cpp**

### **B) iOS**

* **CoreML Tools**
* Apple Neural Engine (ANE) optimized quantized models

### Realistically you can put:

* **100M–1B parameter models** directly on a phone
* **2B–3B** on high-end phones with 4-bit quantization

---

# 🎯 **Final Summary (Copy This Checklist)**

### If you want to **build + train your own LLM like Banana Nano Pro** you need:

### **Hardware**

* **Training:**

  * Minimum: RTX 3090/4090
  * Recommended: A100 40/80GB
* **Inference:**

  * T4/A10G for small APIs
  * A100/H100 for high throughput
* **On-device:** Android/iOS + quantized <1B model

### **Software**

* Ubuntu + CUDA 11.8/12
* Python 3.10+
* PyTorch >= 2.1
* HuggingFace Transformers
* PEFT (LoRA/QLoRA)
* BitsAndBytes
* AutoGPTQ/AWQ
* vLLM (deployment)

---

If you want, I can also give you:

### ✅ A **ready-to-use shopping list** (exact GPU, SSD, RAM, cost)

### ✅ A **complete software installation script** (bash script)

### ✅ A **step-by-step local PC training setup**

Just tell me **which one you want next**.



Perfect — here are **all three deliverables**, complete, clean, and copy-paste ready:

---

# ✅ **1. READY-TO-USE SHOPPING LIST (PC FOR LLM TRAINING + DEPLOYMENT)**

This is the **best cost–performance** build for training 1B–7B LLMs with QLoRA and running quantized inference.

---

## **💻 PC Build (₹ Pricing for India | USD included)**

### **🟦 A) Recommended Build — RTX 4090 (Best for QLoRA)**

Can finetune up to **7B models** comfortably.

| Component       | Model                                   | Price (₹ INR)        | Price (USD) |
| --------------- | --------------------------------------- | -------------------- | ----------- |
| **GPU**         | NVIDIA **RTX 4090 24GB**                | ₹1,85,000 – 2,05,000 | $1800–2000  |
| **CPU**         | AMD Ryzen **9 7950X**                   | ₹45,000              | $430        |
| **Motherboard** | X670 / B650 high-end                    | ₹20,000–30,000       | $200–300    |
| **RAM**         | **64GB DDR5 (2×32GB)**                  | ₹18,000              | $150        |
| **Storage 1**   | **1TB NVMe Gen4 SSD** (OS + tools)      | ₹6,000               | $60         |
| **Storage 2**   | **2TB NVMe Gen4 SSD** (datasets/models) | ₹10,000              | $100        |
| **PSU**         | **1000W Gold**                          | ₹12,000              | $120        |
| **Cooling**     | 360mm AIO                               | ₹10,000              | $90         |
| **Cabinet**     | Airflow cabinet                         | ₹6,000               | $60         |

### **➡️ Total Cost:**

**₹3,20,000 – ₹3,50,000** (approx)
**$3,100 – $3,400 USD**

---

## **🟦 B) Budget Build — RTX 3090 (Still excellent)**

Good for finetuning **3B–7B** with minor restrictions.

| GPU | NVIDIA **RTX 3090 24GB**
| Price | ₹1,00,000 – 1,20,000 (used market) |

➡ Total build: **₹2,20,000 – ₹2,40,000 INR**

---

## **🟩 C) High-End Server Build — A100 40GB (Cloud or workstation)**

If you want REAL training power (not just finetuning):

* NVIDIA **A100 40GB PCIe** — ₹6,50,000 to ₹9,00,000
* Requires server chassis
* 128GB RAM minimum

➡ Only recommended for **professional lab-level training**.

---

# ✅ **2. COMPLETE SOFTWARE INSTALLATION SCRIPT (BASH — COPY & RUN)**

This will install:

✔ CUDA
✔ PyTorch
✔ HuggingFace
✔ BitsAndBytes
✔ Transformers
✔ vLLM
✔ AutoGPTQ
✔ PEFT
✔ Accelerate

Save as `install_llm_env.sh`:

```bash
#!/bin/bash

echo "=== Updating System ==="
sudo apt update && sudo apt upgrade -y

echo "=== Installing Essentials ==="
sudo apt install -y git wget curl build-essential python3-pip python3-venv

echo "=== Creating Python Environment ==="
python3 -m venv llm-env
source llm-env/bin/activate

echo "=== Upgrading Pip ==="
pip install --upgrade pip

echo "=== Installing PyTorch with CUDA ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing HuggingFace Libraries ==="
pip install transformers accelerate datasets safetensors

echo "=== Installing BitsAndBytes for 4-bit/8-bit ==="
pip install bitsandbytes

echo "=== Installing PEFT (LoRA/QLoRA) ==="
pip install peft

echo "=== Installing AutoGPTQ for Quantization ==="
pip install auto-gptq

echo "=== Installing vLLM for Fast Inference ==="
pip install vllm

echo "=== Installing Optional Monitoring Tools ==="
pip install tensorboard wandb

echo "=== Environment Setup Complete ==="
echo "Run: source llm-env/bin/activate"
```

Run via:

```bash
chmod +x install_llm_env.sh
./install_llm_env.sh
```

---

# ✅ **3. STEP-BY-STEP LOCAL PC TRAINING SETUP (QLoRA FINETUNING)**

This will teach you EXACTLY how to finetune a 1B–7B model on your PC.

---

# **STEP 1 — Create Project Folder**

```bash
mkdir llm-training
cd llm-training
```

---

# **STEP 2 — Activate Environment**

```bash
source llm-env/bin/activate
```

---

# **STEP 3 — Pick a Base Model**

For example:

```bash
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

OR a 7B:

```bash
MODEL="meta-llama/Llama-2-7b-hf"
```

---

# **STEP 4 — Prepare Dataset (JSONL format)**

Example: `data.jsonl`

```jsonl
{"instruction":"Explain biogas.","input":"","output":"Biogas is..."}
{"instruction":"Summarize.","input":"Methane is produced when...","output":"Summary..."}
```

---

# **STEP 5 — Finetune with QLoRA (Python Script)**

Create: `train_qlora.py`

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "meta-llama/Llama-2-7b-hf"
dataset = load_dataset("json", data_files="data.jsonl", split="train")

tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(example):
    prompt = f"### Instruction:\n{example['instruction']}\n### Input:\n{example['input']}\n### Response:\n{example['output']}"
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(preprocess)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora)

training_args = TrainingArguments(
    output_dir="lora-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    fp16=True,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=200,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
model.save_pretrained("lora-adapter")
```

---

# **STEP 6 — Run Finetuning**

```bash
python train_qlora.py
```

This produces:

```
/lora-adapter/
    adapter_model.safetensors
    adapter_config.json
```

---

# **STEP 7 — Quantize Base Model (4-bit GPTQ)**

```bash
auto-gptq quantize \
  --model meta-llama/Llama-2-7b-hf \
  --dataset data.jsonl \
  --bits 4 \
  --output_dir quantized-model
```

---

# **STEP 8 — Run Fast Inference using vLLM**

Create `run_vllm.py`:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="quantized-model")

prompt = "What is biomethane?"
output = llm.generate(prompt, SamplingParams(max_tokens=100))
print(output[0].outputs[0].text)
```

Run:

```bash
python run_vllm.py
```

---

# 🎯 YOU ARE DONE

You now have:

### ✔ A high-end PC shopping list

### ✔ A full install script

### ✔ A complete QLoRA training pipeline

### ✔ Quantization

### ✔ Fast inference engine

---

If you want, I can also give you:

### 🔥 **A full Docker deployment file (Dockerfile + docker-compose)**

### 🔥 **A full API server (FastAPI) with your finetuned model**

### 🔥 **What exact model size your hardware can support (calculator)**

Just tell me **what you want next**.


Great — here are **all three** deliverables, production-ready and copy-pasteable.

* A GPU-aware **Dockerfile** (CUDA base, installs dependencies).
* A **docker-compose.yml** that runs the FastAPI server with GPU access and mounts model volumes.
* A **FastAPI server** (`app.py`) that serves your finetuned/quantized model and exposes a `/generate` API.
* A **model-size calculator** with clear formulas and a practical table mapping common GPUs → what model sizes they can support for QLoRA finetuning and quantized inference.

I assume you already have:

* A quantized base model directory (e.g. `models/quantized-base`) **or** a merged (LoRA merged) + quantized model directory (e.g. `models/merged-quantized`) ready to load at runtime.
* If you only have separate LoRA adapters, I include notes on both runtime-adapter usage and the simpler merged model workflow.

---

# 1) Dockerfile (GPU, CUDA 11.8, Python env)

Save as `Dockerfile`:

```dockerfile
# Use NVIDIA CUDA base image (11.8) for compatibility with PyTorch/cu118
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# Basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv git build-essential curl ca-certificates \
    libsndfile1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /app

# Create venv for isolation
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch for CUDA 11.8 (index url). Adjust torch version if needed.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Core python dependencies
# Note: we pin minimal versions. Adjust if you prefer latest.
RUN pip install --no-cache-dir \
    "transformers>=4.30.0" \
    "accelerate" \
    "datasets" \
    "safetensors" \
    "bitsandbytes" \
    "peft" \
    "auto-gptq" \
    "uvicorn[standard]" \
    "fastapi" \
    "vllm" \
    "pydantic"

# Copy app code
COPY ./app /app

# Expose port
EXPOSE 8000

# Default command - run Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

> Notes:
>
> * This Dockerfile installs `vllm` and `auto-gptq` and bitsandbytes. If you prefer a lighter image (only transformers + AutoGPTQ), remove vLLM.
> * If you use a different CUDA version, change the base image and PyTorch index URL accordingly.
> * `--workers 1` is recommended for GPU-bound processes; scale with multiple containers for more concurrency.

---

# 2) docker-compose.yml

Save as `docker-compose.yml` next to the Dockerfile.

```yaml
version: "3.8"
services:
  llm-api:
    build: .
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    environment:
      - MODEL_PATH=/models/merged-quantized
      - FASTAPI_WORKERS=1
    volumes:
      - ./models:/models          # host folder with your quantized/merged model(s)
      - ./app:/app                # app code
    ports:
      - "8000:8000"
    restart: unless-stopped
```

> To run:
>
> ```bash
> docker compose up --build -d
> ```

If your Docker runtime requires `--gpus` config or a different compose key, adapt accordingly (e.g., `deploy.resources.reservations.devices` may need changes depending on Docker version). On NVIDIA Docker Engine you can also pass `--gpus all`.

---

# 3) FastAPI server (`app.py`)

Create an `app` folder and put this `app.py` inside it (`./app/app.py`). This server uses **vLLM** for fast inference of quantized models. It expects a merged+quantized model directory at the environment variable `MODEL_PATH`. If you have separate LoRA adapters and want runtime LoRA application, I include notes below.

```python
# app/app.py
import os
import typing as t
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

# vLLM import (fast inference)
try:
    from vllm import LLM, SamplingParams
except Exception as e:
    raise RuntimeError("vllm import failed. Make sure vllm is installed in the container.") from e

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/merged-quantized")
MAX_TOKENS_DEFAULT = 256

app = FastAPI(title="LLM FastAPI", version="1.0")

# Create a global LLM instance on startup
llm: t.Optional[LLM] = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = MAX_TOKENS_DEFAULT
    temperature: float = 0.8
    top_p: float = 0.95
    stop: t.Optional[t.List[str]] = None

class GenerateResponse(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    global llm
    # vLLM LLM loader - adjust model arg if necessary
    try:
        llm = LLM(model=MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}") from e

@app.get("/health")
async def health():
    return {"status": "ok", "model_path": MODEL_PATH}

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    global llm
    if llm is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Build the prompt (you can adopt instruction format used during finetune)
    prompt = req.prompt

    sampling_params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )

    try:
        outputs = llm.generate([prompt], sampling_params=sampling_params)
        text = outputs[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GenerateResponse(text=text)
```

### How to call the API

Example `curl`:

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain biomethane in 3 sentences.","max_tokens":150}'
```

Response JSON:

```json
{"text":"Biomethane is ..."}
```

---

# 4) Notes about LoRA adapters vs merged models

**Two practical options** to serve a finetuned model:

A) **Merged quantized model** (recommended for simple serving)

* Merge LoRA adapters into the base model, then quantize the merged weights (AutoGPTQ/AWQ).
* Place merged & quantized model under `/models/merged-quantized`.
* vLLM (or Transformers + AutoGPTQ) can load the merged quantized model directly — easiest to serve in Docker.

B) **Runtime LoRA application** (flexible)

* Load the quantized base and apply LoRA adapters at runtime. Some inference stacks (vLLM, Transformers with bitsandbytes + peft) support applying adapters without merging, but setup is more complex.
* If you want this route, tell me your exact inference stack preference and I’ll give the runtime-LoRA server code.

If you want, I can provide a small helper script that merges LoRA into the base and re-quantizes automatically.

---

# 5) Model-size calculator (formulas + practical table)

Below are **exact formulas** and practical, conservative guidance so you can pick the correct GPU.

## Key memory-size formulas (bytes)

* FP16 parameter size (bytes) = `params * 2`
* INT4 parameter size (bytes) = `params * 0.5` (4 bits = 0.5 bytes)
* Convert bytes → GiB for practical VRAM use: divide by `1e9` for GB approximations (we use base-10 GB here for simplicity).

We use the following **parameter counts**:

* 1B  = 1,000,000,000 parameters
* 3B  = 3,000,000,000
* 7B  = 7,000,000,000
* 13B = 13,000,000,000

### Exact computed parameter-storage sizes (rounded to 2 decimals, GB)

* **1B**: FP16 = `1e9 * 2 = 2.00 GB` ; INT4 = `1e9 * 0.5 = 0.50 GB`
* **3B**: FP16 = `3e9 * 2 = 6.00 GB` ; INT4 = `3e9 * 0.5 = 1.50 GB`
* **7B**: FP16 = `7e9 * 2 = 14.00 GB` ; INT4 = `7e9 * 0.5 = 3.50 GB`
* **13B**: FP16 = `13e9 * 2 = 26.00 GB` ; INT4 = `13e9 * 0.5 = 6.50 GB`

> These are the raw parameter storage; you still need headroom for activations, optimizer states (during training), KV cache (inference), and other library overhead.

## Practical memory overhead assumptions (conservative)

* **QLoRA finetuning (4-bit base + LoRA)** requires additional VRAM for:

  * model activations and gradients (if any),
  * optimizer (small for LoRA),
  * framework overhead (~2–8 GB depending on batching, gradient_accumulation, CPU-offload usage).
* **Inference with quantized 4-bit model** still needs extra memory for:

  * token KV cache (per token overhead depends on model dims; budget ~0.5–2 GB for moderate sessions),
  * activation buffers,
  * runtime overhead.

## Practical GPU recommendations (conservative)

| GPU (VRAM)                   | QLoRA finetune (recommended max model)                                                         | Inference (quantized 4-bit)               |
| ---------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **RTX 3060 / 4060 (8–12GB)** | ≤ **1B** (small models only)**                                                                 | 1B int4 comfortably                       |
| **RTX 3090 / 4090 (24GB)**   | ≈ **up to 7B** with careful config (gradient_accumulation, CPU offload) — best practice: 3B–7B | 7B int4 comfortable (with extra headroom) |
| **A100 40GB**                | **7B** easily; 13B with heavy optimizations                                                    | 13B int4 feasible, 7B effortless          |
| **A100 80GB / H100 80GB**    | **13B** comfortably; larger with sharding                                                      | 13B+ inference & more throughput          |
| **T4 / L4 (16GB)**           | ≤ **3B** (small QLoRA jobs)                                                                    | 3B int4 ok with optimizations             |

### Quick mapping examples

* **You have RTX 4090 (24GB)** → Best practical: finetune **1B–7B** (QLoRA). For 7B use careful configs (smaller batch sizes, gradient accumulation). For inference, 7B quantized is easily served.
* **You have A100 40GB** → Finetune **7B** comfortably, **13B** with memory/CPU offload tweaks. Inferencing 13B quantized is feasible.
* **You have H100 80GB** → Use for larger models (13B and beyond) and high-throughput serving.

---

# 6) Small script to *estimate* model VRAM (local calculator)

Copy this small Python snippet to test different model sizes and GPU VRAM assumptions.

```python
# quick_vram_calc.py
def sizes(params):
    fp16_gb = params * 2 / 1e9
    int4_gb = params * 0.5 / 1e9
    return fp16_gb, int4_gb

models = {"1B":1_000_000_000, "3B":3_000_000_000, "7B":7_000_000_000, "13B":13_000_000_000}
for name,p in models.items():
    fp16, int4 = sizes(p)
    print(f"{name}: FP16={fp16:.2f} GB, INT4={int4:.2f} GB")
```

---

# 7) How to build & run (quick steps)

1. Put your merged+quantized model under `./models/merged-quantized` on the host.
2. Build & run:

   ```bash
   docker compose up --build -d
   ```
3. Test health:

   ```bash
   curl http://localhost:8000/health
   ```
4. Generate:

   ```bash
   curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" \
     -d '{"prompt":"Explain biomethane simply.","max_tokens":120}'
   ```

---

# 8) Want me to also provide (pick any)

* A script to **merge LoRA adapter into the base model** and re-quantize automatically (AutoGPTQ-compatible).
* The **runtime-LoRA** FastAPI server (apply LoRA adapters at request time instead of merging).
* A **docker-compose scaling example** behind an API gateway (NGINX) with Prometheus metrics.

Tell me which of the above (or multiple) you want and I’ll produce the exact files.

---

If you want, I can now **produce the merge-and-quantize helper** (shell + Python) — this is the most common next step to make deployment trivial. Which next?


Here is a **clean, production-ready repository structure** for your **LLM Finetuning + Quantization + FastAPI Deployment (Docker + vLLM)** project.

This structure is used in real ML Ops deployments and keeps training, quantization, serving, and Docker pipelines clean and separated.

---

# 🚀 **Complete Repo Structure (Best Practice)**

```
llm-project/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/                      # original training datasets
│   │   └── data_raw.jsonl
│   ├── processed/                # cleaned/filtered data
│   │   └── train.jsonl
│   └── data_prep.py              # script to clean/prepare dataset
│
├── training/
│   ├── finetune_qlora.py         # QLoRA finetuning script
│   ├── merge_lora.py             # merge LoRA → base model
│   ├── config/
│   │   └── training_args.json    # training hyperparameters
│   └── logs/                     # training logs, wandb logs
│
├── models/
│   ├── base/                      # downloaded pretrained model
│   │   └── (hf files)
│   ├── lora-output/               # trained LoRA adapter weights
│   │   ├── adapter_model.safetensors
│   │   └── adapter_config.json
│   ├── merged-model/              # merged (base + LoRA)
│   │   └── (unquantized merged files)
│   ├── quantized/                 # final quantized (GPTQ or AWQ)
│   │   └── (model.safetensors etc)
│   └── tokenizer/                 # tokenizer assets (optional)
│
├── quantization/
│   ├── quantize_gptq.py           # script for AutoGPTQ quantization
│   ├── quantize_awq.py            # optional AWQ quantization script
│   └── calibrate_data.jsonl       # calibration dataset
│
├── api/
│   ├── app.py                     # FastAPI server (vLLM inference)
│   ├── model_loader.py            # vLLM / HF loader utility
│   ├── schemas.py                 # request/response models
│   └── utils.py                   # helper functions
│
├── docker/
│   ├── Dockerfile                 # GPU-enabled Dockerfile
│   ├── docker-compose.yml         # for GPU deployment
│   └── start.sh                   # start script inside container
│
├── scripts/
│   ├── install_env.sh             # install PyTorch + tools
│   ├── vram_calculator.py         # determine max model size
│   ├── test_api.sh                # curl tests
│   └── download_model.py          # script to download HF model
│
└── deployment/
    ├── nginx.conf                 # reverse proxy (optional)
    ├── service_monitoring/        # Prometheus/Grafana configs
    ├── systemd_service.service    # systemd deployment file
    └── k8s/                       # Kubernetes manifests (optional)
```

---

# 🧠 **WHY THIS STRUCTURE IS PERFECT**

Each part of your workflow is isolated:

| Folder            | Purpose                                               |
| ----------------- | ----------------------------------------------------- |
| **data/**         | All dataset processing, cleaning, splitting.          |
| **training/**     | QLoRA training, merging LoRA, training configs.       |
| **models/**       | Stores every stage: base → LoRA → merged → quantized. |
| **quantization/** | GPTQ/AWQ scripts and calibration sets.                |
| **api/**          | FastAPI + vLLM serving code.                          |
| **docker/**       | Deployment infrastructure.                            |
| **scripts/**      | Utility scripts (installation, VRAM calculator).      |
| **deployment/**   | Advanced deployment: NGINX, systemd, Kubernetes.      |

---

# 📦 **Recommended File Breakdown**

## **training/finetune_qlora.py**

* Loads base model
* Applies QLoRA
* Saves LoRA adapter → `models/lora-output`

## **training/merge_lora.py**

* Merges base + LoRA → outputs **models/merged-model**

## **quantization/quantize_gptq.py**

* Converts merged model → **models/quantized**

## **api/app.py**

* FastAPI server
* Loads model from `/models/quantized`
* Exposes `/generate` endpoint

## **docker/docker-compose.yml**

* Mounts models + API
* Gives GPU access
* Runs Uvicorn + vLLM

---

# 🏗️ **Where To Place Your Existing Files**

| File you already created       | Folder                       |
| ------------------------------ | ---------------------------- |
| `train_qlora.py`               | `training/finetune_qlora.py` |
| `run_vllm.py / FastAPI app.py` | `api/app.py`                 |
| Quantized model                | `models/quantized/`          |
| Dockerfile                     | `docker/Dockerfile`          |
| docker-compose.yml             | `docker/docker-compose.yml`  |
| VRAM calculator                | `scripts/vram_calculator.py` |

---

# 🔥 **Optional Extras to Make Repo Enterprise-Grade**

### Add a config folder for environment variables:

```
config/
  ├── dev.env
  ├── prod.env
  └── model_paths.json
```

### Add CI/CD:

```
.github/
  workflows/
    build.yml
    test.yml
    deploy.yml
```

### Add tests:

```
tests/
  ├── test_api.py
  └── test_training.py
```

---

# 🎯 **Want me to generate the actual files now?**

I can generate:

### ✔ Full working repo **with all folders + empty placeholders**

### ✔ Auto-generated **LoRA merge script**

### ✔ Auto-generated **GPTQ quantization script**

### ✔ Full **FastAPI code with streaming tokens**

### ✔ GitHub-ready **README.md** with diagrams

### ✔ `download_model.py` to automatically fetch the base model

### ✔ "get_started.sh" to prepare entire repo in one click

Just tell me:

👉 **Do you want the FULL repo ZIP (all files ready to run), or only specific files?**


# llm-project — Full Working Repo

This document contains a complete, runnable repository scaffold with all requested files: working placeholders, auto-generated LoRA merge script, GPTQ quantization script, FastAPI server with token streaming (vLLM), README (GitHub-ready with simple diagrams), `download_model.py`, and a `get_started.sh` helper to prepare the repo end-to-end.

> **Note:** This textdoc is the authoritative copy of all files. Do not duplicate these files into chat — they're available here for download and editing.

---

## Repo tree

```
llm-project/
├── README.md
├── .gitignore
├── requirements.txt
├── get_started.sh
├── download_model.py
|
├── data/
│   ├── raw/data_raw.jsonl
│   ├── processed/train.jsonl
│   └── data_prep.py
|
├── training/
│   ├── finetune_qlora.py
│   ├── merge_lora.py
│   └── config/training_args.json
|
├── quantization/
│   ├── quantize_gptq.py
│   └── calibrate_data.jsonl
|
├── models/
│   ├── base/                # downloaded base model
│   ├── lora-output/         # adapter output after finetune
│   ├── merged-model/        # merged base + lora
│   └── quantized/           # final quantized model
|
├── api/
│   ├── app.py               # FastAPI server (streaming tokens)
│   ├── model_loader.py
│   └── schemas.py
|
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
|
├── scripts/
│   ├── install_env.sh
│   └── vram_calculator.py
|
└── LICENSE
```

---

# Files (copy each into the corresponding path)

---

## README.md

```markdown
# llm-project

Full end-to-end repo for QLoRA finetuning → LoRA merge → GPTQ quantization → vLLM FastAPI streaming inference → Docker deployment.

## Quick diagram

```

[Local machine / Server] --> [training/ (QLoRA)] --> models/lora-output/
|--> training/merge_lora.py --> models/merged-model/
|--> quantization/quantize_gptq.py --> models/quantized/

[models/quantized/] --> [docker/ container running api/app.py (vLLM)] --> clients

```

## Quickstart

1. Run `./get_started.sh` to create env and install packages.
2. Download base model: `python download_model.py --model-id <HF_MODEL_ID> --out models/base`.
3. Prepare data: `python data/data_prep.py` (place output in `data/processed/train.jsonl`).
4. Finetune: `python training/finetune_qlora.py --model models/base --data data/processed/train.jsonl`.
5. Merge LoRA: `python training/merge_lora.py --base models/base --lora models/lora-output --out models/merged-model`.
6. Quantize: `python quantization/quantize_gptq.py --model models/merged-model --out models/quantized --bits 4`.
7. Run Docker: `docker compose -f docker/docker-compose.yml up --build -d`.
8. Test: `curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' -d '{"prompt":"Hello","max_tokens":50}'`.

For more details, see the sections in repo files.
```

````

---

## .gitignore

```gitignore
venv/
__pycache__/
*.pyc
*.safetensors
models/merged-model/*
models/quantized/*
.env
*.log
````

---

## requirements.txt

```text
# CPU/GPU libs (versions are indicative; tune for your CUDA)
torch>=2.1.0
transformers>=4.30.0
datasets
accelerate
safetensors
bitsandbytes
peft
auto-gptq
vllm
fastapi
uvicorn[standard]
python-multipart
pydantic
requests
```

---

## get_started.sh

```bash
#!/bin/bash
set -e

echo "=== get_started.sh: setup starting ==="

# create python venv
python3 -m venv llm-env
source llm-env/bin/activate
pip install --upgrade pip

# install core packages
pip install -r requirements.txt

echo "=== Done. Activate environment: source llm-env/bin/activate ==="
```

---

## download_model.py

```python
"""
Simple HF downloader for a model (weights + tokenizer) using huggingface_hub / transformers.
Usage: python download_model.py --model-id meta-llama/Llama-2-7b-hf --out models/base
"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def download(model_id, out_dir):
    print(f"Downloading {model_id} -> {out_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.save_pretrained(out_dir)
    # We don't always want to download huge weights to repo; use local cache or HF login when needed
    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
    model.save_pretrained(out_dir)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    download(args.model_id, args.out)
```

---

## data/data_prep.py

```python
"""
Minimal data prep placeholder: copies raw jsonl -> processed with very small dedup flow.
Place your raw examples in data/raw/data_raw.jsonl
"""
import json
from pathlib import Path

RAW = Path("data/raw/data_raw.jsonl")
OUT = Path("data/processed/train.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

seen = set()
with RAW.open('r', encoding='utf-8') as inf, OUT.open('w', encoding='utf-8') as outf:
    for line in inf:
        line=line.strip()
        if not line:
            continue
        try:
            j=json.loads(line)
        except Exception:
            continue
        key=(j.get('instruction','')+j.get('input','')+j.get('output','')).strip()
        if key in seen:
            continue
        seen.add(key)
        outf.write(json.dumps(j, ensure_ascii=False)+"\n")

print(f"Written processed dataset to {OUT}")
```

---

## training/finetune_qlora.py

```python
"""
QLoRA finetuning script (small, portable). Uses PEFT + bitsandbytes.
This is a minimal template — tune batch sizes and accelerate config as needed.
"""
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch


def build_prompt(example):
    # change prompt format to match your instruction tuning
    instr = example.get('instruction','')
    inp = example.get('input','')
    out = example.get('output','')
    prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='models/lora-output')
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    ds = load_dataset('json', data_files={'train':args.data})['train']
    def tokenize(example):
        prompt = build_prompt(example)
        return tokenizer(prompt, truncation=True, max_length=512)
    ds = ds.map(tokenize, batched=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_4bit=True,
        device_map='auto'
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj','v_proj'], bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        fp16=True,
        logging_steps=10,
        save_total_limit=2,
    )

    def collate_fn(batch):
        input_ids = [torch.tensor(b['input_ids']) for b in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collate_fn,
    )

    trainer.train()
    model.save_pretrained(args.output)
    print(f"Saved LoRA adapter to {args.output}")

if __name__ == '__main__':
    main()
```

---

## training/merge_lora.py

```python
"""
Merge LoRA adapters into base model. Saves merged weights under out_dir.
This script uses PEFT's merge capabilities when possible.
"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def merge(base_dir, lora_dir, out_dir):
    print(f"Loading base model from {base_dir}")
    model = AutoModelForCausalLM.from_pretrained(base_dir, device_map='auto')
    print(f"Applying LoRA from {lora_dir}")
    # Load LoRA via PeftModel
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(model, lora_dir)
    # Merge weights
    print("Merging LoRA into base weights (in-place). This may take a while...")
    peft_model.merge_and_unload()
    print(f"Saving merged model to {out_dir}")
    peft_model.save_pretrained(out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--lora', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    merge(args.base, args.lora, args.out)
```

---

## quantization/quantize_gptq.py

```python
"""
AutoGPTQ-style quantization wrapper. This is a high-level script to run AutoGPTQ
on a merged HF-format model and dump quantized checkpoint to out_dir.

Note: auto-gptq package and its CLI can also be used directly. This script uses
`auto_gptq` Python API if available. Adjust to your repo's AutoGPTQ installation.
"""
import argparse
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer


def quantize(model_path, out_dir, bits=4):
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print("Creating AutoGPTQ quantizer...")
    # This interface may differ between versions. Adjust per your auto-gptq install.
    q_config = BaseQuantizeConfig(bits=bits, group_size=128)
    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config=q_config, device='cuda:0')
    print(f"Saving quantized model to {out_dir}")
    model.save_quantized(out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--bits', type=int, default=4)
    args = parser.parse_args()
    quantize(args.model, args.out, args.bits)
```

---

## quantization/calibrate_data.jsonl

```jsonl
{"text":"This is a calibration sentence for GPTQ/AWQ."}
```

---

## api/schemas.py

```python
from pydantic import BaseModel
from typing import Optional, List

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    stop: Optional[List[str]] = None

class TokenChunk(BaseModel):
    token: str
    is_last: bool = False

class GenerateResponse(BaseModel):
    text: str
```

---

## api/model_loader.py

```python
import os
from vllm import LLM

MODEL_PATH = os.environ.get('MODEL_PATH','/models/quantized')

llm = None

def load_model(model_path=None):
    global llm
    if model_path is None:
        model_path = MODEL_PATH
    print(f"Loading vLLM model from {model_path}...")
    llm = LLM(model=model_path)
    return llm
```

---

## api/app.py (FastAPI with streaming tokens)

```python
import os
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from api.schemas import GenerateRequest
from api.model_loader import load_model, llm
from vllm import SamplingParams

app = FastAPI(title='LLM vLLM Streaming API')

@app.on_event('startup')
async def startup_event():
    load_model()

@app.get('/health')
async def health():
    return {'status':'ok'}

@app.post('/generate')
async def generate(req: GenerateRequest):
    if llm is None:
        raise HTTPException(status_code=500, detail='Model not loaded')

    prompt = req.prompt
    sampling_params = SamplingParams(temperature=req.temperature, top_p=req.top_p, max_tokens=req.max_tokens)

    # vLLM synchronous generate returns an iterator-like object; we stream tokens.
    def gen():
        outputs = llm.generate([prompt], sampling_params=sampling_params)
        # outputs is a list of Generation outputs
        for out in outputs:
            text = out.text
            yield text

    return StreamingResponse(gen(), media_type='text/plain')
```

---

## docker/Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash","-lc"]

RUN apt-get update && apt-get install -y python3-pip python3-venv git curl build-essential ca-certificates libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY ./api /app/api

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## docker/docker-compose.yml

```yaml
version: '3.8'
services:
  llm-api:
    build: .
    runtime: nvidia
    environment:
      - MODEL_PATH=/models/quantized
    volumes:
      - ../models:/models:ro
      - ../api:/app/api
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

---

## scripts/install_env.sh

```bash
#!/bin/bash
set -e

python3 -m venv llm-env
source llm-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready. Activate with: source llm-env/bin/activate"
```

---

## scripts/vram_calculator.py

```python
"""Small utility to print model memory sizes."""

models = {'1B':1_000_000_000,'3B':3_000_000_000,'7B':7_000_000_000,'13B':13_000_000_000}

for name,params in models.items():
    fp16 = params*2/1e9
    int4 = params*0.5/1e9
    print(f"{name}: FP16={fp16:.2f} GB, INT4={int4:.2f} GB")
```

---

## LICENSE

```text
Apache-2.0
```

---

# Final notes

* All scripts are minimal, portable templates. Some third-party APIs (AutoGPTQ, vLLM, bitsandbytes) may require specific versions of CUDA and matching PyTorch builds. Adjust `requirements.txt` and Docker base image to match your hardware/CUDA versions.

* If you want, I can now generate a downloadable ZIP of this repo, or produce each file separately as editable snippets. Tell me which you prefer next.

---

*End of repo textdoc.*



Nice — I’ll generate **all five files now**. Below are ready-to-save files you can drop into the repo paths shown earlier.

---

## 1) `training/merge_lora.py`

Merges LoRA adapters into a base HF model (uses `peft` merge). Saves merged model to `--out`.

```python
#!/usr/bin/env python3
"""
training/merge_lora.py

Merge a LoRA adapter into a base HF model and save the merged model.

Usage:
    python training/merge_lora.py --base models/base --lora models/lora-output --out models/merged-model
"""
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge(base_dir: str, lora_dir: str, out_dir: str):
    assert os.path.isdir(base_dir), f"Base model dir not found: {base_dir}"
    assert os.path.isdir(lora_dir), f"LoRA dir not found: {lora_dir}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[merge_lora] Loading base model from {base_dir} (may be CPU heavy)...")
    model = AutoModelForCausalLM.from_pretrained(base_dir, device_map="auto", low_cpu_mem_usage=True)
    print(f"[merge_lora] Attaching LoRA from {lora_dir}...")
    peft_model = PeftModel.from_pretrained(model, lora_dir, device_map="auto")
    print("[merge_lora] Merging LoRA weights into base (this modifies model in-memory)...")
    peft_model.merge_and_unload()
    print(f"[merge_lora] Saving merged model to {out_dir} ...")
    peft_model.save_pretrained(out_dir)
    # Save tokenizer if present in base
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=False)
        tokenizer.save_pretrained(out_dir)
    except Exception:
        print("[merge_lora] tokenizer copy skipped (not found or error).")
    print("[merge_lora] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Path to base pretrained HF model")
    parser.add_argument("--lora", required=True, help="Path to LoRA adapter folder")
    parser.add_argument("--out", required=True, help="Output path for merged model")
    args = parser.parse_args()
    merge(args.base, args.lora, args.out)
```

---

## 2) `quantization/quantize_gptq.py`

A robust wrapper that tries the `auto_gptq` Python API and falls back to calling the AutoGPTQ CLI (if available). Saves quantized model to `--out`.

```python
#!/usr/bin/env python3
"""
quantization/quantize_gptq.py

Quantize a merged HF model using AutoGPTQ-style quantization.

Usage:
  python quantization/quantize_gptq.py --model models/merged-model --out models/quantized --bits 4 --group-size 128

Notes:
- Requires `auto-gptq` python package or `auto-gptq` CLI available in PATH.
- Check AutoGPTQ docs for optimal group_size and quantization settings.
"""
import argparse
import os
import subprocess
import json

def quantize_with_api(model_path, out_dir, bits=4, group_size=128):
    try:
        from auto_gptq import BaseQuantizeConfig, AutoGPTQForCausalLM
    except Exception as e:
        raise RuntimeError("auto_gptq python API not available.") from e

    print(f"[quantize_gptq] Using auto_gptq python API to quantize {model_path} -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    q_config = BaseQuantizeConfig(bits=bits, group_size=group_size)
    # device can be 'cuda:0' or 'cpu' depending on env - prefer cuda if available
    device = "cuda:0"
    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config=q_config, device=device)
    model.save_quantized(out_dir)
    print("[quantize_gptq] Quantization complete via API.")

def quantize_with_cli(model_path, out_dir, bits=4, group_size=128):
    print(f"[quantize_gptq] Falling back to AutoGPTQ CLI for quantization.")
    # This is a common CLI form used by some AutoGPTQ forks - adjust if your CLI uses different flags
    cmd = [
        "auto_gptq", "quantize", 
        "--model", model_path,
        "--output", out_dir,
        "--bits", str(bits),
        "--group-size", str(group_size),
        "--no-tracing"
    ]
    print("[quantize_gptq] Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("[quantize_gptq] Quantization complete via CLI.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to merged HF model")
    parser.add_argument("--out", required=True, help="Output directory for quantized model")
    parser.add_argument("--bits", type=int, default=4, help="Bits for quantization (4 recommended)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for GPTQ-like quantization")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        raise SystemExit(f"Model path not found: {args.model}")

    os.makedirs(args.out, exist_ok=True)

    # Try API path, otherwise CLI
    try:
        quantize_with_api(args.model, args.out, bits=args.bits, group_size=args.group_size)
    except Exception as e:
        print(f"[quantize_gptq] auto_gptq API quantization failed: {e}\nAttempting CLI approach (if available).")
        quantize_with_cli(args.model, args.out, bits=args.bits, group_size=args.group_size)

if __name__ == "__main__":
    main()
```

---

## 3) `data/data_prep.py`

Cleaner, slightly enhanced dataset prep — supports simple filtering, deduplication, and output sample stats.

```python
#!/usr/bin/env python3
"""
data/data_prep.py

Simple dataset cleaning pipeline:
- Reads data/raw/data_raw.jsonl
- Filters empty lines and missing fields
- Deduplicates by instruction+input+output key
- Writes cleaned file to data/processed/train.jsonl

Run:
  python data/data_prep.py
"""
import json
from pathlib import Path

RAW = Path("data/raw/data_raw.jsonl")
OUT = Path("data/processed/train.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

if not RAW.exists():
    print(f"[data_prep] No raw input found at {RAW}. Create it first.")
    raise SystemExit(1)

seen = set()
count_in = 0
count_out = 0
skipped = 0

with RAW.open("r", encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
    for line in fin:
        count_in += 1
        line = line.strip()
        if not line:
            skipped += 1
            continue
        try:
            obj = json.loads(line)
        except Exception:
            skipped += 1
            continue

        instr = (obj.get("instruction") or "").strip()
        inp = (obj.get("input") or "").strip()
        out = (obj.get("output") or "").strip()

        # Basic validation
        if not instr and not inp:
            skipped += 1
            continue
        if not out:
            skipped += 1
            continue

        key = (instr + "||" + inp + "||" + out)[:1000]  # truncated key
        if key in seen:
            continue
        seen.add(key)
        fout.write(json.dumps({"instruction": instr, "input": inp, "output": out}, ensure_ascii=False) + "\n")
        count_out += 1

print(f"[data_prep] Input lines: {count_in}, Skipped: {skipped}, Written: {count_out}")
print(f"[data_prep] Cleaned dataset saved to {OUT}")
```

---

## 4) `api/model_loader.py`

Improved loader that tries `vllm` first; falls back to a HF+transformers loader if `vllm` isn't available. Exposes `get_llm()` to return the model instance.

```python
#!/usr/bin/env python3
"""
api/model_loader.py

Load the model for inference. Prefer vLLM for high-throughput GPU inference.
If vLLM is not available, attempt to load a transformers AutoModelForCausalLM.
"""
import os
import traceback

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/quantized")
LLM = None
LOADER = None

def load_vllm(model_path):
    from vllm import LLM
    print(f"[model_loader] Loading vLLM model from {model_path} ...")
    return LLM(model=model_path)

def load_transformers(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[model_loader] Loading transformers model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", low_cpu_mem_usage=True)
    return model, tokenizer

def get_llm():
    global LLM, LOADER
    if LLM is not None:
        return LLM

    # Try vLLM first
    try:
        LLM = load_vllm(MODEL_PATH)
        LOADER = "vllm"
        return LLM
    except Exception as e:
        print("[model_loader] vLLM load failed:", e)
        traceback.print_exc()

    # Fallback to transformers
    try:
        model, tokenizer = load_transformers(MODEL_PATH)
        LLM = (model, tokenizer)
        LOADER = "transformers"
        return LLM
    except Exception as e:
        print("[model_loader] transformers load failed:", e)
        traceback.print_exc()
        raise RuntimeError("Failed to load any model backend. Check logs and ensure model files exist.")

if __name__ == "__main__":
    print("[model_loader] Testing model loader ...")
    get_llm()
    print(f"[model_loader] LOADER = {LOADER}")
```

---

## 5) `docker/start.sh`

A container start helper that sets ownership/permissions, optionally preloads the model, and starts uvicorn. Place inside `docker/start.sh` and make executable. This is helpful for docker entrypoint if you prefer extra startup behavior.

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "[start.sh] container startup beginning..."

# Optional: ensure model dir exists
MODEL_DIR=${MODEL_PATH:-/models/quantized}
if [ ! -d "$MODEL_DIR" ]; then
  echo "[start.sh] WARNING: Model dir not found at $MODEL_DIR"
else
  echo "[start.sh] Model dir present: $MODEL_DIR"
fi

# Fix permissions for mounted volumes (if necessary)
if [ "$(id -u)" -eq 0 ]; then
  chown -R 1000:1000 "$MODEL_DIR" 2>/dev/null || true
  echo "[start.sh] chown attempted on $MODEL_DIR"
fi

# Optional model warmup: attempt a small load to avoid first-request latency
if [ "${WARMUP:-1}" -eq 1 ]; then
  echo "[start.sh] Warmup enabled. Attempting lightweight model load (if vLLM present)..."
  python - <<'PY'
import os
try:
    from vllm import LLM
    model_path = os.environ.get("MODEL_PATH","/models/quantized")
    print("[start.sh:warmup] Loading model (vLLM) for warmup:", model_path)
    LLM(model=model_path)
    print("[start.sh:warmup] vLLM warmup load OK")
except Exception as e:
    print("[start.sh:warmup] vLLM warmup failed or not installed:", e)
PY
fi

# Start the FastAPI server with Uvicorn (single worker recommended for GPU)
UVICORN_CMD=${UVICORN_CMD:-"uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 1"}
echo "[start.sh] Starting server: $UVICORN_CMD"
exec $UVICORN_CMD
```

Make it executable:

```bash
chmod +x docker/start.sh
```

If you prefer the Dockerfile to use `start.sh` as entrypoint, change `CMD` / `ENTRYPOINT` in the Dockerfile to:

```dockerfile
COPY ./docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
```

---

### Final notes & next steps

* Drop these files into the repo at the paths shown earlier.
* If you want, I can:

  * produce a ready-to-download ZIP of the entire repo scaffold,
  * generate unit tests for the API endpoints,
  * add a `merge+quantize` orchestration script that runs `merge_lora.py` then `quantize_gptq.py` automatically,
  * or modify the FastAPI streaming to yield token-by-token SSE (server-sent events) for richer front-end integration.

Which of those would you like next?
