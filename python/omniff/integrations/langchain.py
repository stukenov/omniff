from __future__ import annotations

from typing import Any


class OmniFFTool:
    name: str = "omniff"
    description: str = (
        "Universal multimodal AI tool. Send text, images, audio, video, or documents "
        "for analysis, generation, or transformation. Input: text string or file path. "
        "Output: text response or generated file path."
    )

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        default_thinking: str = "normal",
    ) -> None:
        self.base_url = base_url
        self.default_thinking = default_thinking

    def _run(self, query: str) -> str:
        from omniff.client import SyncClient
        client = SyncClient(self.base_url)
        result = client.run(input=query, thinking=self.default_thinking)
        return result.get("output_text", result.get("output_path", "No output"))

    async def _arun(self, query: str) -> str:
        from omniff.client import AsyncClient
        client = AsyncClient(self.base_url)
        result = await client.run(input=query, thinking=self.default_thinking)
        return result.get("output_text", result.get("output_path", "No output"))
