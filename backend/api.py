from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from .config import settings
from .embeddings import Embedder
from .generator import Generator
from .ingest import ingest_file
from .retriever import Retriever
from .vectorstore import VectorStore

app = FastAPI(title="Course RAG Assistant")
store = VectorStore(settings.vectorstore_dir); store.load()
embedder = None; generator = None

def services():
    global embedder, generator
    if embedder is None: embedder = Embedder(settings.embedding_model, settings.device)
    if generator is None: generator = Generator(settings.model_name, settings.device, settings.max_new_tokens, settings.strict_extractive)
    return Retriever(store, embedder, settings.reranker_model), generator

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    corpus: str = "all"
    history: list[ChatTurn] = []

@app.get("/health")
def health(): return {"status": "ok", "chunks": len(store.metadata), "strict_extractive": settings.strict_extractive}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() not in {".pdf", ".docx", ".pptx", ".md", ".txt"}: raise HTTPException(400, "Unsupported file type")
    settings.data_dir.mkdir(parents=True, exist_ok=True); target = settings.data_dir / Path(file.filename).name
    with target.open("wb") as out: shutil.copyfileobj(file.file, out)
    return {"file": target.name}

@app.post("/ingest")
def ingest():
    files = [p for p in settings.data_dir.iterdir() if p.suffix.lower() in {".pdf", ".docx", ".pptx", ".md", ".txt"}]
    chunks = [chunk for file in files for chunk in ingest_file(file, settings.chunk_size, settings.chunk_overlap)]
    if not chunks: raise HTTPException(400, "No supported files in data directory")
    retriever, _ = services(); store.index = None; store.metadata = []; store.add(retriever.embedder.encode([x.text for x in chunks]), [x.metadata() for x in chunks]); retriever._bm25 = None
    return {"files": len(files), "chunks": len(chunks)}

@app.post("/chat")
def chat(request: ChatRequest):
    retriever, gen = services(); matches = retriever.search(request.question, settings.top_k, settings.rerank_k, request.corpus)
    history = [turn.model_dump() for turn in request.history if turn.content][-settings.history_turns:]
    return {"answer": gen.answer(request.question, matches, settings.confidence_threshold, history), "citations": matches, "diagnostics": {"matches": len(matches), "best_score": matches[0]["score"] if matches else 0}}
