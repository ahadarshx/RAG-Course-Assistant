"""Run: python evaluation/evaluate.py evaluation/golden.json"""
from __future__ import annotations
import json, sys
from pathlib import Path
import requests

def overlap(a, b):
    a, b = set(a.lower().split()), set(b.lower().split()); return len(a & b) / max(1, len(a | b))

def main(path: str):
    cases = json.loads(Path(path).read_text(encoding="utf-8")); rows = []
    for case in cases:
        result = requests.post("http://127.0.0.1:8000/chat", json={"question": case["question"]}, timeout=300).json()
        citations, expected = result["citations"], set(case.get("sources", []))
        found = {x["source"] for x in citations}; precision = len(found & expected) / max(1, len(found)); recall = len(found & expected) / max(1, len(expected))
        answer_relevance = overlap(result["answer"], case.get("answer", case["question"]))
        rows.append({"question": case["question"], "retrieval_precision": precision, "retrieval_recall": recall, "context_precision": precision, "context_recall": recall, "faithfulness": 1.0 if citations else 0.0, "answer_relevance": answer_relevance, "hallucination_detected": not citations and "outside" not in result["answer"].lower()})
    report = {"cases": rows, "averages": {k: sum(x[k] for x in rows if isinstance(x[k], float)) / max(1, len(rows)) for k in ["retrieval_precision", "retrieval_recall", "context_precision", "context_recall", "faithfulness", "answer_relevance"]}}
    Path("evaluation/report.json").write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report["averages"], indent=2))
if __name__ == "__main__": main(sys.argv[1])
