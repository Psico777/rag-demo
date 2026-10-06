"""FastAPI: POST /ask {"question": "..."} -> {"answer", "sources"}; GET /search?q= -> retrieved chunks (no LLM)."""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .pipeline import answer, retrieve

app = FastAPI(title="RAG demo")


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.post("/ask")
def ask(q: Ask):
    return answer(q.question)


@app.get("/search")
def search(q: str, k: int = 3):
    return [{"source": d.metadata["source"], "text": d.page_content} for d in retrieve(q, k)]
