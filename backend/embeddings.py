from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str, device: str = "auto"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=None if device == "auto" else device)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")
