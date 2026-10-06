"""Ingest Markdown docs into Chroma and answer questions with citations (LangChain LCEL)."""
from __future__ import annotations

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import LocalMiniLM

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = os.getenv("RAG_DB_DIR", str(ROOT / ".chroma"))
COLLECTION = "docs"

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY with the context below. If the answer is not there, say you don't know. "
               "Cite sources as [source] after each claim. Be brief.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def load_docs(folder: str | Path = ROOT / "docs") -> list[Document]:
    docs = [Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name}) for p in sorted(Path(folder).glob("*.md"))]
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60).split_documents(docs)


def vectorstore(reset: bool = False, folder: str | Path = ROOT / "docs") -> Chroma:
    vs = Chroma(collection_name=COLLECTION, embedding_function=LocalMiniLM(), persist_directory=DB_DIR)
    if reset:
        vs.delete_collection()
        vs = Chroma(collection_name=COLLECTION, embedding_function=LocalMiniLM(), persist_directory=DB_DIR)
    if reset or not vs.get(limit=1)["ids"]:
        vs.add_documents(load_docs(folder))
    return vs


def retrieve(question: str, k: int = 3, vs: Chroma | None = None) -> list[Document]:
    return (vs or vectorstore()).similarity_search(question, k=k)


def llm():
    """Any OpenAI-compatible server: Ollama (default), vLLM, OpenAI…"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"), api_key=os.getenv("LLM_API_KEY", "ollama"),
                      model=os.getenv("LLM_MODEL", "qwen2.5:3b"), temperature=0)


def answer(question: str, k: int = 3) -> dict:
    hits = retrieve(question, k)
    context = "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in hits)
    text = (PROMPT | llm() | StrOutputParser()).invoke({"context": context, "question": question})
    return {"answer": text, "sources": sorted({d.metadata["source"] for d in hits})}
