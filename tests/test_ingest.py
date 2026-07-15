from pathlib import Path
from backend.ingest import ingest_file


def test_markdown_is_chunked_with_metadata(tmp_path: Path):
    note = tmp_path / "course.md"
    note.write_text("Module 1: Basics\n\nData mining finds patterns.\n\nModel Questions\n\n1. Define data mining.")
    chunks = ingest_file(note, chunk_size=80, chunk_overlap=10)
    assert chunks and chunks[0].source == "course.md" and chunks[0].module.startswith("Module")
    assert any(chunk.corpus == "questions" for chunk in chunks)
