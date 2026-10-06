"""Rebuild the vector store from docs/*.md:  python ingest.py"""
from rag.pipeline import vectorstore

vs = vectorstore(reset=True)
print("chunks indexed:", len(vs.get()["ids"]))
