from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel


class PDFGeneratorModel(OmniModel):
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        try:
            import reportlab  # noqa: F401
        except ImportError:
            raise ImportError("reportlab required: pip install reportlab") from None
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        text = inputs["text"]
        title = inputs.get("title", "OmniFF Generated Document")
        output_path = inputs.get("output_path", "output.pdf")

        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()

        story = []
        story.append(Paragraph(title, styles["Title"]))
        story.append(Spacer(1, 12))

        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(para.replace("\n", "<br/>"), styles["Normal"]))
                story.append(Spacer(1, 6))

        doc.build(story)

        return {
            "pdf_path": output_path,
            "pages": 1,
        }
