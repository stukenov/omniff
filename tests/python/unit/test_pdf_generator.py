import pytest


def test_pdf_generator_interface():
    from omniff.models.pdf_generator import PDFGeneratorModel

    model = PDFGeneratorModel()
    assert not model.is_loaded


def test_pdf_generator_infer_not_loaded():
    from omniff.models.pdf_generator import PDFGeneratorModel

    model = PDFGeneratorModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"text": "hello"})


def test_pdf_generator_creates_pdf(tmp_path):
    try:
        import reportlab  # noqa: F401
    except ImportError:
        pytest.skip("reportlab not installed")

    from omniff.models.pdf_generator import PDFGeneratorModel

    model = PDFGeneratorModel()
    model.load()
    assert model.is_loaded

    output = str(tmp_path / "test.pdf")
    result = model.infer(
        {
            "text": "This is a test document.\n\nWith multiple paragraphs.",
            "title": "Test Doc",
            "output_path": output,
        }
    )
    assert result["pdf_path"] == output
    assert (tmp_path / "test.pdf").exists()
    assert (tmp_path / "test.pdf").stat().st_size > 0
