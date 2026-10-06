"""Retrieval quality without an LLM: the right document must be the top hit."""
import pytest

from rag.pipeline import retrieve, vectorstore

CASES = [
    ("Can I get my money back for an unopened bag?", "refunds.md"),
    ("How much does express delivery cost?", "shipping.md"),
    ("Is motor damage covered and for how long?", "grinders.md"),
    ("Do you ship to Peru?", "shipping.md"),
]


@pytest.fixture(scope="module")
def vs(tmp_path_factory):
    import os
    os.environ["RAG_DB_DIR"] = str(tmp_path_factory.mktemp("chroma"))
    import importlib
    import rag.pipeline as p
    importlib.reload(p)
    return p.vectorstore(reset=True)


@pytest.mark.parametrize("question,expected", CASES)
def test_top_hit(vs, question, expected):
    assert retrieve(question, k=1, vs=vs)[0].metadata["source"] == expected


def test_api_search(vs):
    from fastapi.testclient import TestClient

    from rag.api import app
    r = TestClient(app).get("/search", params={"q": "express shipping price", "k": 1})
    assert r.status_code == 200 and r.json()[0]["source"] == "shipping.md"
