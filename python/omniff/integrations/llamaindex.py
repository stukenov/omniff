from __future__ import annotations

from pathlib import Path
from typing import Any


class OmniFFReader:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        supported_extensions: list[str] | None = None,
    ) -> None:
        self.base_url = base_url
        self.supported_extensions = supported_extensions or [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".mp3",
            ".wav",
            ".flac",
            ".pdf",
            ".docx",
            ".txt",
            ".mp4",
            ".avi",
            ".mov",
        ]

    def load_data(self, file_path: str | Path) -> list[dict[str, Any]]:
        from omniff.client import SyncClient

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        client = SyncClient(self.base_url)
        result = client.run(input=str(file_path))

        return [
            {
                "text": result.get("output_text", ""),
                "metadata": {
                    "source": str(file_path),
                    "route": result.get("route", ""),
                    **result.get("metadata", {}),
                },
            }
        ]

    def can_read(self, file_path: str | Path) -> bool:
        suffix = Path(file_path).suffix.lower()
        return suffix in self.supported_extensions
