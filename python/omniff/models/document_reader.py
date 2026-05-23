from __future__ import annotations

from pathlib import Path
from typing import Any

from omniff.models.base import OmniModel


class DocumentReaderModel(OmniModel):
    """Document→text. Extracts text from PDF/DOCX/TXT then optionally summarizes via LLM."""

    def __init__(
        self,
        llm_model_id: str = "Qwen/Qwen3-4B",
        device: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        self.llm_model_id = llm_model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._llm = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        from omniff.models.llm import LLMModel

        self._llm = LLMModel(
            model_id=self.llm_model_id,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
        )
        self._llm.load()

    def unload(self) -> None:
        if self._llm:
            self._llm.unload()
        self._llm = None

    def _extract_text(self, doc_path: str) -> str:
        path = Path(doc_path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="replace")

        if suffix == ".pdf":
            return self._extract_pdf(str(path))

        if suffix in (".docx", ".doc"):
            return self._extract_docx(str(path))

        return path.read_text(encoding="utf-8", errors="replace")

    def _extract_pdf(self, pdf_path: str) -> str:
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            pass

        try:
            from pdfminer.high_level import extract_text

            return extract_text(pdf_path)
        except ImportError:
            return f"[PDF extraction requires PyMuPDF or pdfminer: {pdf_path}]"

    def _extract_docx(self, docx_path: str) -> str:
        try:
            import docx

            doc = docx.Document(docx_path)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return f"[DOCX extraction requires python-docx: {docx_path}]"

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        doc_path = inputs.get("document_path", "")
        prompt = inputs.get("prompt")

        raw_text = self._extract_text(doc_path)

        if not prompt:
            return {"text": raw_text, "source": "extraction"}

        # Truncate to ~4000 chars to fit in context
        context = raw_text[:4000]
        llm_prompt = f"Document content:\n{context}\n\nTask: {prompt}"
        result = self._llm.infer({"prompt": llm_prompt, "thinking": False})
        return {"text": result["text"], "source": "llm", "raw_length": len(raw_text)}
