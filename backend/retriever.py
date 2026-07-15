from __future__ import annotations

import re
from collections import defaultdict
from .embeddings import Embedder
from .vectorstore import VectorStore


def tokens(text: str) -> list[str]: return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, reranker_model: str):
        self.store, self.embedder, self.reranker_model = store, embedder, reranker_model; self._bm25 = self._reranker = None

    def _bm(self):
        if self._bm25 is None and self.store.metadata:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tokens(x["text"]) for x in self.store.metadata])
        return self._bm25

    def search(self, question: str, top_k: int, rerank_k: int, corpus: str = "all") -> list[dict]:
        dense = self.store.search(self.embedder.encode([question])[0], top_k * 3)
        scores = defaultdict(float)
        dense_scores = {}
        for rank, (score, item) in enumerate(dense):
            scores[item["chunk_id"]] += 1 / (rank + 1)
            dense_scores[item["chunk_id"]] = score
        bm = self._bm()
        if bm:
            bm_scores = bm.get_scores(tokens(question))
            for rank, i in enumerate(sorted(range(len(self.store.metadata)), key=lambda x: bm_scores[x], reverse=True)[:top_k * 3]): scores[self.store.metadata[i]["chunk_id"]] += 1 / (rank + 1)
        result = [{**next(x for x in self.store.metadata if x["chunk_id"] == key), "score": score, "dense_score": dense_scores.get(key, 0.0)} for key, score in scores.items()]
        if corpus != "all": result = [x for x in result if x["corpus"] == corpus]
        result = sorted(result, key=lambda x: x["score"], reverse=True)[:top_k * 2]
        if result:
            try:
                if self._reranker is None:
                    from sentence_transformers import CrossEncoder
                    self._reranker = CrossEncoder(self.reranker_model)
                for item, score in zip(result, self._reranker.predict([(question, x["text"]) for x in result])): item["rerank_score"] = float(score)
                result.sort(key=lambda x: x["rerank_score"], reverse=True)
            except Exception: pass  # Retrieval remains useful offline without the optional reranker weights.
        return result[:rerank_k]
