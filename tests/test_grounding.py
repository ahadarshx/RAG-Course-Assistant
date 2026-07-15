from backend.generator import Generator


def test_rejects_unrelated_question():
    result = Generator("unused", "cpu", 1).answer("Who is the president of France?", [{"text": "Data mining discovers patterns in data.", "dense_score": 0.12}], 0.35)
    assert "not available" in result


def test_rejects_generic_word_overlap():
    result = Generator("unused", "cpu", 1).answer("Who won the FIFA World Cup in 2022?", [{"text": "Data mining has real world applications.", "dense_score": 0.70}], 0.35)
    assert "not available" in result


def test_accepts_relevant_question_without_loading_model():
    generator = Generator("unused", "cpu", 1, strict_extractive=False)
    generator._load = lambda: setattr(generator, "pipe", lambda *_args, **_kwargs: [{"generated_text": "Grounded answer"}])
    result = generator.answer("What is data mining?", [{"text": "Data mining discovers patterns in data.", "dense_score": 0.72, "source": "notes.pdf", "page_number": 1}], 0.35)
    assert result == "Grounded answer"


def test_strict_mode_returns_note_text_only():
    result = Generator("unused", "cpu", 1).answer("What is data mining?", [{"text": "Data mining discovers hidden patterns in data. It supports analysis.", "dense_score": 0.72}], 0.35)
    assert "Data mining discovers hidden patterns in data." in result


def test_strict_mode_rejoins_pdf_line_breaks():
    result = Generator("unused", "cpu", 1).answer("What is a data warehouse?", [{"text": "A Data Warehouse is a centralized repository designed to store historical\n data from heterogeneous sources.", "dense_score": 0.72}], 0.35)
    assert "historical data" in result


def test_warehousing_matches_warehouse_evidence():
    result = Generator("unused", "cpu", 1).answer("What is data warehousing?", [{"text": "A Data Warehouse is a centralized repository for historical data. ETL transforms data.", "dense_score": 0.72}], 0.35)
    assert "centralized repository" in result
