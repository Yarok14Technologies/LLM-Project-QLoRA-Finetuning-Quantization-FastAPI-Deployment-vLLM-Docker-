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

Apache 2.0 (or your preferred license)

---

# 💬 Support

For help integrating your own dataset or customizing inference, open an issue or ask ChatGPT to generate specialized scripts (merge automate, quantize tools, dataset filters, etc.)

---

