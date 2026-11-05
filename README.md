# Multimodal AI Assistant for Intelligent Document & Visual Understanding

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-orange.svg)](https://www.trychroma.com/)
[![RapidOCR](https://img.shields.io/badge/OCR-RapidOCR-brightgreen.svg)](https://github.com/RapidAI/RapidOCR)
[![Google GenAI](https://img.shields.io/badge/LLM-Gemini_3.5_Flash-4285F4.svg)](https://ai.google.dev/)

A unified multimodal conversational AI assistant combining **Retrieval-Augmented Generation (RAG)** over user documents (PDF, DOCX, TXT, MD) with **Computer Vision** and **Deep Learning OCR** over images (PNG, JPG, WEBP) and scanned pages.

Developed as a **B.Tech Capstone Project** emphasizing clean, explainable software design, factual grounding with verifiable citations, zero-system-dependency OCR, and offline vector representations.

---

## Architecture Pipeline

```text
User Query / Uploads
       │
       ▼
 [ Input Router (router.py) ]
   ├── Document File ──► [ PyMuPDF / RapidOCR ] ──► [ Recursive Chunker ] ──► [ BGE-small Embeddings ] ──► [ ChromaDB ]
   ├── Text Query ─────► [ Embed Query (Prefix) ] ─► [ ChromaDB Similarity ] ──► [ Top-K Grounded Chunks ] ──► [ Gemini LLM ]
   ├── Image Query ────► [ Gemini Vision Engine ] ──────────────────────────────────────────────────────────► [ Visual Analysis ]
   └── Mixed Query ────► [ Top-K Chunks + Attached Image ] ────────────────────────────────────────────────► [ Unified Multimodal Call ]
```

---

## Key Features

1. **Grounded Document RAG with Exact Citations**: Indexes documents with SHA-256 deduplication. Generates answers strictly from retrieved context chunks with clickable citations (`[Doc: filename, Page: X]`) and an expandable source inspection panel.
2. **Visual Understanding & Diagram Reasoning**: Reads charts, graphs, screenshots, and visual schematics using Google's Gemini Vision models.
3. **Mixed Multimodal Queries**: Attach an image and query the indexed document library simultaneously in a single prompt call, synthesizing visual observations with document specifications.
4. **Scanned PDF Support (RapidOCR)**: Automatically detects scanned or sparse pages (< 50 characters) and executes OCR via `rapidocr-onnxruntime` (pure Python/ONNX, zero host binary dependencies).
5. **PDF Figure Searchability**: Rasterizes and captions figures and charts inside PDF documents using the vision model during ingestion, storing them as searchable `figure_caption` chunks.
6. **Strict Grounding & Anti-Prompt-Injection**: System prompts isolate retrieved excerpts inside XML tags and politely refuse out-of-domain queries without hallucinating facts or citations.

---

## Project Structure

```text
mm-assistant/
├── app/
│   └── streamlit_app.py        # Streamlit web application with chat & document manager
├── src/mmassistant/
│   ├── config.py               # Application configuration dataclass reading .env
│   ├── router.py               # Input classifier (document, text, image, mixed)
│   ├── pipeline.py             # End-to-end MultimodalPipeline coordinator
│   ├── prompts.py              # Strict grounding rules, XML context quarantine, refusal policies
│   ├── cli.py                  # Full CLI runner: ingest, ask, ask-image, ask-mixed, list-docs
│   ├── embeddings.py           # BGE-small-en-v1.5 sentence-transformers embedder
│   ├── vectorstore.py          # Persistent ChromaDB vector store with cosine distance
│   ├── retrieval.py            # Similarity search, top-k ranking, threshold filtering
│   ├── vision.py               # Computer vision reasoning and image validation
│   ├── ingestion/
│   │   ├── loaders.py          # PDF (PyMuPDF), Word (python-docx), TXT, MD loaders
│   │   ├── chunking.py         # Recursive splitter with table preservation
│   │   ├── ocr.py              # RapidOCR offline engine with Gemini Vision fallback
│   │   └── figures.py          # PDF diagram rasterizer & semantic captioning engine
│   └── llm/
│       ├── base.py             # Abstract LLMProvider interface & LLMResponse
│       └── gemini.py           # Google GenAI SDK client with exponential backoff & model fallback
├── tests/                      # Automated test suite (34 offline pytest tests)
├── eval/
│   ├── test_dataset.json       # 15 curated benchmark queries
│   ├── evaluate.py             # Benchmark runner for Hit@K, Citations, and Faithfulness
│   └── results.md              # Recorded evaluation metrics and analysis
├── docs/
│   ├── ARCHITECTURE.md         # Detailed Mermaid diagrams and component contracts
│   └── REPORT_NOTES.md         # Comprehensive B.Tech report notes & viva cheat sheet
├── storage/                    # Persistent vector database and cache (Git-ignored)
└── data/samples/               # Synthetic test files (markdown, scanned PDF, chart image)
```

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.11 or higher
- Git

### 2. Setup Virtual Environment

#### On Windows (PowerShell):
```powershell
# Clone or open the project folder
cd mm-assistant

# Create virtual environment with Python 3.11
py -3.11 -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

#### On macOS / Linux (Bash):
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -e ".[dev]"
```

### 4. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and set your Google Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_ID=gemini-3.5-flash
```

---

## Running the Web Application

Launch the Streamlit web interface:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at **`http://localhost:8501`**.

---

## Command Line Interface (CLI)

The CLI allows full headless execution and testing:

```bash
# 1. Inspect configuration
python -m mmassistant.cli config

# 2. Test Gemini connection
python -m mmassistant.cli test-llm

# 3. Ingest documents (PDF, DOCX, TXT, MD)
python -m mmassistant.cli ingest data/samples/sample_document.md
python -m mmassistant.cli ingest data/samples/scanned_report.pdf --figures

# 4. List indexed documents
python -m mmassistant.cli list-docs

# 5. Ask document RAG question
python -m mmassistant.cli ask "What is the continuous electrical power delivered by the MMRTG?"

# 6. Ask image question (Computer Vision)
python -m mmassistant.cli ask-image --image data/samples/sample_chart.png "What are the subsystem masses?"

# 7. Ask mixed query (Image + Vector Index)
python -m mmassistant.cli ask-mixed --image data/samples/sample_chart.png "Cross-reference the chart with the report"
```

---

## Automated Testing & Benchmarking

### Unit & Integration Tests (Offline)
```bash
pytest -v tests/
```
All 34 tests run offline using synthetic documents, PIL images, and mocked LLM providers.

### Quantitative Evaluation Benchmark
```bash
python eval/evaluate.py
```
Evaluates **Hit@1**, **Hit@4**, **Citation Precision**, and **Faithfulness** across 15 curated queries, writing results to `eval/results.md`.

---

## Viva Cheat Sheet & Design Decisions

- **Why RAG instead of Fine-Tuning?** Instant updates without GPU retraining, verifiable document citations, and zero risk of hallucinating company/academic data.
- **Why BGE-small-en-v1.5?** 384 dimensions, runs locally on CPU in <20ms, outperforms larger models on MTEB, and enforces query-passage asymmetry via instruction prefixes.
- **Why RapidOCR?** Powered by ONNX Runtime; requires zero host-level C++ binaries (unlike Tesseract), eliminating cross-platform setup headaches.
- **Why ChromaDB?** Serverless in-process vector store with HNSW indexing ($O(\log N)$ search time) and persistent disk storage in `storage/chroma/`.
