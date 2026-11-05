# Multimodal AI Assistant for Intelligent Document and Visual Understanding

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-orange.svg)](https://www.trychroma.com/)

A multimodal AI assistant capable of answering natural language queries across documents (PDF, DOCX, TXT, MD) via Retrieval-Augmented Generation (RAG) and images (PNG, JPG, WEBP) via computer vision and multimodal LLMs.

Built from scratch for a **B.Tech Capstone Project**, emphasizing clean architecture, offline local embeddings, zero-system-dependency OCR, and factual grounding with verifiable source citations.

---

## Key Features

1. **Document RAG with Exact Citations**: Indexes multiple documents with SHA-256 deduplication. Answers strictly from retrieved chunks with clickable citations (`[Doc: report.pdf, Page: 4]`) and an expandable source inspection panel.
2. **Visual Understanding & Diagram Reasoning**: Reads charts, graphs, screenshots, and visual documents using Google's Gemini Vision models.
3. **Mixed Multimodal Queries**: Attach an image and ask a question that compares it against indexed documents in a single unified prompt.
4. **Scanned PDF Support (RapidOCR)**: Automatically detects scanned or sparse pages (< 50 characters) and executes OCR via `rapidocr-onnxruntime` (pure Python/ONNX, zero host dependencies).
5. **PDF Figure Searchability**: Rasterizes and captions figures and charts inside PDF documents using the vision model during ingestion, storing them as searchable `figure_caption` chunks.
6. **Robust Grounding & Anti-Prompt-Injection**: Strict system prompts instruct the model to treat retrieved document text strictly as passive data and politely refuse out-of-domain queries.

---

## Architecture Pipeline

```text
User Query / Document / Image
       │
       ▼
 [ Input Router ]
   ├── Document Upload ──► [ PyMuPDF / RapidOCR ] ──► [ Recursive Chunker ] ──► [ BGE-small Embeddings ] ──► [ ChromaDB ]
   ├── Text Query ───────► [ Embed Query ] ───────► [ ChromaDB Similarity ] ──► [ Top-K Grounding Chunks ] ──► [ Gemini LLM ]
   ├── Image Query ──────► [ Gemini Vision ] ─────────────────────────────────────────────────────────────► [ Answer ]
   └── Mixed Query ──────► [ Top-K Chunks + Image ] ──────────────────────────────────────────────────────► [ Unified Multimodal Call ]
```

---

## Project Structure

```text
mm-assistant/
├── .env.example                # Template configuration
├── .gitignore                  # Git ignore rules
├── pyproject.toml              # Build & dependency metadata
├── requirements.txt            # Pinned requirements
├── README.md                   # Project overview & instructions
├── AGENTS.md                   # Developer rules & phase tracking
├── app/
│   └── streamlit_app.py        # Streamlit web UI
├── src/mmassistant/
│   ├── config.py               # Application configuration
│   ├── router.py               # Query & input classifier
│   ├── pipeline.py             # RAG and Multimodal coordinator
│   ├── prompts.py              # System prompts & grounding rules
│   ├── cli.py                  # CLI tool for headless testing
│   ├── embeddings.py           # BGE-small local sentence-transformers
│   ├── vectorstore.py          # ChromaDB persistence & metadata filtering
│   ├── retrieval.py            # Similarity search & chunk ranking
│   ├── vision.py               # Image processing & visual analysis
│   ├── ingestion/              # Document parsing, chunking, OCR & figure extractor
│   └── llm/                    # Abstract LLMProvider & Gemini implementation
├── tests/                      # Pytest automated test suite (mocked APIs)
├── eval/                       # Evaluation dataset & Hit@K metrics script
├── docs/                       # Architecture diagrams & B.Tech report notes
└── storage/                    # Persistent vector database and cache
```

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.11 or higher
- Git

### 2. Setup Virtual Environment

#### On Windows (PowerShell):
```powershell
# Clone or navigate to the repository directory
cd mm-assistant

# Create virtual environment with Python 3.11
py -3.11 -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

#### On macOS / Linux:
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
Open `.env` and add your Google Gemini API key:
```env
GOOGLE_API_KEY=AIzaSy...your_actual_key_here
GEMINI_MODEL_ID=gemini-2.0-flash
```

### 5. Verify Installation
```bash
# Verify active configuration and paths
python -m mmassistant.cli config

# Test connection to Gemini API
python -m mmassistant.cli test-llm

# Run automated unit tests (runs offline with mocked LLMs)
pytest -v tests/
```

### 6. Run Web Application
```bash
streamlit run app/streamlit_app.py
```

---

## Running Automated Tests
```bash
pytest -v tests/
```
All tests use synthetic document fixtures and mocked LLM providers, ensuring they run quickly and offline without consuming API quota.
