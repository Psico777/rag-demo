"""CLI:  python ask.py "How long do I have to return unopened beans?" """
import sys

from rag.pipeline import answer

r = answer(" ".join(sys.argv[1:]) or "What does the grinder warranty cover?")
print(r["answer"], "\n\nSources:", ", ".join(r["sources"]))
