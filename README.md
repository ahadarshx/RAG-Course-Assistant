# course / assistant

<p align="center">
  <strong>A local-first, citation-first RAG assistant for university course material.</strong><br>
  Upload notes. Ask questions. Study from evidence.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Gradio-5-FF7C00?style=flat-square&logo=gradio&logoColor=white" alt="Gradio 5">
  <img src="https://img.shields.io/badge/RAG-FAISS%20%2B%20BM25-7B61FF?style=flat-square" alt="RAG">
  <img src="https://img.shields.io/badge/Model-Qwen2.5%201.5B-111111?style=flat-square" alt="Qwen">
</p>

![Course Assistant preview](docs/preview.png)

> Add a screenshot at `docs/preview.png` after your first run.

## Why it exists

Course material is dense. This project turns PDFs, slides, Word documents, Markdown, and text files into a searchable study companion that answers from uploaded notes only and shows the supporting sources.

```text
course files → semantic chunks → FAISS + BM25 → reranking → cited answer
```

## Highlights

- Grounded answers with strict out-of-scope refusal
- Page, module, chapter, topic, and source citations
- Separate course-content and official-question-bank retrieval
- Hybrid retrieval: dense vectors, BM25, and cross-encoder reranking
- Bounded per-chat conversation memory
- Qwen2.5-1.5B 4-bit GPU inference for consumer hardware
- Modern Gradio study workspace for local use

## Quick start

```powershell
git clone https://github.com/YOUR_USERNAME/RAG-Based-LLM.git
cd RAG-Based-LLM

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Start the API:

```powershell
uvicorn backend.api:app --reload --port 8000
```

Start the local interface in a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python frontend/gradio_app.py
```

Open `http://127.0.0.1:7860`, upload your notes, and select **Upload & index**.

## Configure

```env
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
DEVICE=auto
STRICT_EXTRACTIVE=false
CONFIDENCE_THRESHOLD=0.35
HISTORY_TURNS=6
```

Set `STRICT_EXTRACTIVE=true` for verbatim evidence passages instead of generated summaries.

## Project map

```text
backend/       FastAPI, ingestion, retrieval, generation
frontend/      Local Gradio study interface
evaluation/    Golden-set evaluation script
data/          Your uploaded course material
vectorstore/   Local persisted FAISS index
```

## Evaluation

```bash
python evaluation/evaluate.py evaluation/golden.example.json
```
