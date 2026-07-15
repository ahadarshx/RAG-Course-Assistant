from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    vectorstore_dir: Path = Path(os.getenv("VECTORSTORE_DIR", "vectorstore"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "8"))
    rerank_k: int = int(os.getenv("RERANK_K", "4"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    device: str = os.getenv("DEVICE", "auto")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "400"))
    strict_extractive: bool = os.getenv("STRICT_EXTRACTIVE", "true").lower() == "true"
    history_turns: int = int(os.getenv("HISTORY_TURNS", "6"))


settings = Settings()
