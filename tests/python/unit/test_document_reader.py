import pytest

from omniff.models.document_reader import DocumentReaderModel


def test_document_reader_interface():
    model = DocumentReaderModel(llm_model_id="Qwen/Qwen3-4B", device="cpu")
    assert not model.is_loaded


def test_document_reader_infer_not_loaded():
    model = DocumentReaderModel(llm_model_id="Qwen/Qwen3-4B", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"document_path": "test.pdf"})
