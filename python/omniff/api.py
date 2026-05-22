from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from omniff.runtime.config import OmniFFConfig, RouterConfig
from omniff.runtime.engine import OmniFFRuntime


_runtime: OmniFFRuntime | None = None


def get_runtime() -> OmniFFRuntime:
    global _runtime
    if _runtime is None:
        config_path = Path(os.environ.get("OMNIFF_CONFIG", "omniff.yaml"))
        if config_path.exists():
            _runtime = OmniFFRuntime.from_yaml(config_path)
        else:
            config = OmniFFConfig(
                name="omniff",
                version="0.1",
                router=RouterConfig(router_type="keyword", path=""),
            )
            _runtime = OmniFFRuntime(config)
    return _runtime


def create_app():
    try:
        from fastapi import FastAPI, UploadFile, File, Form
        from fastapi.responses import FileResponse, JSONResponse
    except ImportError:
        raise ImportError("FastAPI required: pip install fastapi uvicorn")

    app = FastAPI(
        title="OmniFF API",
        description="FFmpeg-like multimodal AI runtime — HTTP API",
        version="1.0.0",
    )

    @app.post("/run")
    async def run_text(
        input_text: str = Form(...),
        prompt: str | None = Form(None),
        output_modality: str | None = Form(None),
        thinking: str = Form("normal"),
    ) -> dict[str, Any]:
        runtime = get_runtime()
        result = runtime.run(
            input=input_text,
            prompt=prompt,
            output_modality=output_modality,
            thinking=thinking,
        )
        return {
            "output_text": result.output_text,
            "output_path": result.output_path,
            "route": result.route,
            "metadata": result.metadata,
        }

    @app.post("/run/file")
    async def run_file(
        file: UploadFile = File(...),
        prompt: str | None = Form(None),
        output_modality: str | None = Form(None),
        thinking: str = Form("normal"),
    ) -> dict[str, Any]:
        suffix = Path(file.filename or "upload").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            runtime = get_runtime()
            result = runtime.run(
                input=tmp_path,
                prompt=prompt,
                output_modality=output_modality,
                thinking=thinking,
            )

            response: dict[str, Any] = {
                "route": result.route,
                "metadata": result.metadata,
            }
            if result.output_text:
                response["output_text"] = result.output_text
            if result.output_path and Path(result.output_path).exists():
                response["output_path"] = result.output_path
            return response
        finally:
            os.unlink(tmp_path)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "1.0.0"}

    @app.get("/routes")
    async def routes() -> dict[str, list[str]]:
        from omniff.graph.planner import GraphPlanner
        planner = GraphPlanner()
        return {"routes": planner.available_routes()}

    return app


def main():
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn required: pip install uvicorn")

    app = create_app()
    host = os.environ.get("OMNIFF_HOST", "0.0.0.0")
    port = int(os.environ.get("OMNIFF_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
