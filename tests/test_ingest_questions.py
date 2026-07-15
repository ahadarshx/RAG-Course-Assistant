from pathlib import Path
from backend.ingest import ingest_file


def test_question_bank_is_separate(tmp_path: Path):
    note = tmp_path / "course.md"
    note.write_text("Module 1: Basics\n\nDefinitions belong here.\n\nModel Questions\n\n1. Define data mining.\n2. Explain KDD.")
    chunks = ingest_file(note)
    questions = [chunk for chunk in chunks if chunk.corpus == "questions"]
    assert len(questions) == 2
    assert questions[0].text == "1. Define data mining."
