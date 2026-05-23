from __future__ import annotations

from typing import Any, AsyncGenerator


class AsyncClient:
    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    async def run(
        self,
        input: str,
        prompt: str | None = None,
        output_modality: str | None = None,
        thinking: str = "normal",
    ) -> dict[str, Any]:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        data = {"input_text": input, "thinking": thinking}
        if prompt:
            data["prompt"] = prompt
        if output_modality:
            data["output_modality"] = output_modality

        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.base_url}/run", data=data)
            resp.raise_for_status()
            return resp.json()

    async def run_stream(
        self,
        input: str,
        prompt: str | None = None,
        thinking: str = "normal",
    ) -> AsyncGenerator[str, None]:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        data = {"input_text": input, "thinking": thinking}
        if prompt:
            data["prompt"] = prompt

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{self.base_url}/run/stream", data=data) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        token = line[6:]
                        if token == "[DONE]":
                            break
                        yield token

    async def health(self) -> dict[str, str]:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/health")
            return resp.json()

    async def status(self) -> dict[str, Any]:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/status")
            return resp.json()

    async def routes(self) -> list[str]:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/routes")
            return resp.json().get("routes", [])


class SyncClient:
    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def run(
        self,
        input: str,
        prompt: str | None = None,
        output_modality: str | None = None,
        thinking: str = "normal",
    ) -> dict[str, Any]:
        import httpx

        data = {"input_text": input, "thinking": thinking}
        if prompt:
            data["prompt"] = prompt
        if output_modality:
            data["output_modality"] = output_modality

        resp = httpx.post(f"{self.base_url}/run", data=data)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict[str, str]:
        import httpx
        return httpx.get(f"{self.base_url}/health").json()
