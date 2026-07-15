from __future__ import annotations

import json
from pathlib import Path
import numpy as np


class VectorStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.index = None; self.metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        import faiss
        if self.index is None: self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings); self.metadata.extend(metadata); self.persist()

    def search(self, embedding: np.ndarray, k: int) -> list[tuple[float, dict]]:
        if self.index is None: return []
        scores, ids = self.index.search(embedding.reshape(1, -1), min(k, len(self.metadata)))
        return [(float(score), self.metadata[i]) for score, i in zip(scores[0], ids[0]) if i >= 0]

    def persist(self) -> None:
        import faiss
        faiss.write_index(self.index, str(self.directory / "index.faiss"))
        (self.directory / "metadata.json").write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        import faiss
        index, meta = self.directory / "index.faiss", self.directory / "metadata.json"
        if index.exists() and meta.exists():
            self.index = faiss.read_index(str(index)); self.metadata = json.loads(meta.read_text(encoding="utf-8"))
