# RAG demo — ask your documents, get answers with sources

Small, production-style **Retrieval-Augmented Generation** service: Markdown docs → chunks → local embeddings → **Chroma** vector store → **LangChain** prompt → any **OpenAI-compatible LLM** (Ollama, vLLM, OpenAI). Exposed as a **FastAPI** API and a CLI. Answers cite their sources and say "I don't know" when the context doesn't cover the question.

The sample docs are a **fictional** coffee company handbook (refunds, shipping, warranty) so the demo runs anywhere.

```
docs/*.md ──► split (500 chars, 60 overlap) ──► MiniLM embeddings (ONNX, local, no API key)
                                                     │
question ──► similarity search (k=3) ◄── Chroma ◄────┘
     │                 │
     └──► prompt: "answer ONLY with the context, cite [source]" ──► LLM ──► answer + sources
```

## Run it

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows (use bin/activate on Linux/Mac)
pip install -r requirements.txt
python ingest.py                                  # builds .chroma/ from docs/*.md
pytest -q                                         # retrieval tests, no LLM needed

# LLM: local Ollama by default
ollama pull qwen2.5:3b
python ask.py "Can I return an opened bag of coffee?"
uvicorn rag.api:app --reload                      # POST /ask  ·  GET /search?q=
```

Use another backend with environment variables:

| Variable | Default | Example |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | a vLLM server, `https://api.openai.com/v1` |
| `LLM_MODEL` | `qwen2.5:3b` | `gpt-4o-mini`, a fine-tuned model |
| `LLM_API_KEY` | `ollama` | your key (never commit it) |
| `RAG_DB_DIR` | `./.chroma` | any folder |

## API

```http
POST /ask        {"question": "How much is express shipping?"}
→ {"answer": "Express shipping costs US$ 15 and takes 1–2 business days [shipping.md].", "sources": ["shipping.md"]}

GET /search?q=warranty&k=2      → retrieved chunks (debug retrieval without an LLM)
```

## Design choices

- **Local embeddings** (all-MiniLM-L6-v2 via ONNX): no API key, no cost, works offline.
- **Grounded prompt**: the model must answer only from the retrieved context and cite `[source]`; otherwise it says it doesn't know.
- **Retrieval is tested separately** from generation (`tests/test_retrieval.py`), so quality regressions are caught without calling an LLM.
- **Backend-agnostic**: same code against a laptop (Ollama), a GPU server (vLLM) or a hosted API.

## Ideas to extend

Hybrid search (BM25 + vectors), re-ranking, PDF ingestion, per-user collections, evaluation set with answer-faithfulness scoring.

---

Built by **Yordy Rojas** · [GitHub](https://github.com/Psico777) · [LinkedIn](https://linkedin.com/in/yordyrojasb/) · related: [THORP-IA case study](https://github.com/Psico777/thorp-ia-case-study)
