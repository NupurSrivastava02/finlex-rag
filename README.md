<img width="771" height="689" alt="Screen Shot 2026-02-23 at 1 10 30 AM" src="https://github.com/user-attachments/assets/8ccfca7c-d22b-4133-87a7-13e9fd40955b" />
# FinLex RAG — Financial & Legal Document Intelligence

A production-grade Retrieval-Augmented Generation (RAG) system for analyzing financial filings, legal contracts, and SEC reports.

## Architecture

```
User Query
    │
    ▼
FastAPI Backend
    │
    ├── Document Ingestion (LlamaIndex)
    │       └── PDF/DOCX → Chunks → ChromaDB
    │
    ├── RAG Pipeline (LangChain)
    │       └── Query → MMR Retrieval → GPT/Mistral → Cited Answer
    │
    └── RAGAS Evaluation
            └── Faithfulness · Answer Relevancy · Context Precision
```

## Tech Stack

| Layer | Tools |
|---|---|
| Document Parsing | LlamaIndex, PyPDF, python-docx |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (persistent) |
| LLM | OpenAI GPT-3.5/4 or Mistral (via Ollama) |
| Orchestration | LangChain RetrievalQA with MMR |
| Evaluation | RAGAS (faithfulness, relevancy, precision) |
| Backend | FastAPI |
| Frontend | Vanilla JS + CSS (dark finance theme) |

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo>
cd finlex-rag
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. (Optional) Use local Mistral instead of OpenAI

```bash
# Install Ollama: https://ollama.ai
ollama pull mistral

# In .env:
USE_LOCAL_LLM=true
```

### 4. Run the server

```bash
uvicorn main:app --reload --port 8000
```

### 5. Open the UI

Visit `http://localhost:8000`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/documents/upload` | Upload PDF/DOCX |
| GET | `/api/documents/list` | List ingested docs |
| DELETE | `/api/documents/{filename}` | Remove document |
| POST | `/api/query` | Ask a question |
| POST | `/api/query/evaluate` | Run RAGAS eval |

## Usage

1. Upload a 10-K filing, contract, or SEC report
2. Ask questions in plain English
3. Get cited answers with source page references
4. Run RAGAS evaluation to score pipeline quality

## Demo
Upload any SEC filing or legal contract and ask questions in plain English. 
FinLex retrieves relevant passages and returns cited answers with page references.
