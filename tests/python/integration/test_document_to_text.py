import pytest

from omniff.models.document_reader import DocumentReaderModel


@pytest.fixture(scope="module")
def reader():
    model = DocumentReaderModel(llm_model_id="Qwen/Qwen3-4B", device="auto", max_new_tokens=512)
    model.load()
    yield model
    model.unload()


@pytest.fixture
def test_txt_doc(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "The capital of Kazakhstan is Astana. "
        "Kazakhstan is the largest landlocked country in the world. "
        "It covers an area of 2.7 million square kilometers.",
        encoding="utf-8",
    )
    return str(path)


def test_extract_text(reader, test_txt_doc):
    result = reader.infer({"document_path": test_txt_doc})
    assert "text" in result
    assert "Kazakhstan" in result["text"]
    assert result["source"] == "extraction"


def test_summarize_document(reader, test_txt_doc):
    result = reader.infer(
        {
            "document_path": test_txt_doc,
            "prompt": "What is the capital mentioned here? One word answer.",
        }
    )
    assert "text" in result
    assert len(result["text"]) > 0, "LLM returned empty response"
    assert result["source"] == "llm"
