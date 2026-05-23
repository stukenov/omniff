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
        from fastapi import FastAPI, File, Form, UploadFile
        from fastapi.responses import FileResponse, JSONResponse  # noqa: F401
    except ImportError:
        raise ImportError("FastAPI required: pip install fastapi uvicorn") from None

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

    @app.post("/run/stream")
    async def run_stream(
        input_text: str = Form(...),
        prompt: str | None = Form(None),
        thinking: str = Form("normal"),
    ):
        from fastapi.responses import StreamingResponse

        runtime = get_runtime()

        def generate():
            for token in runtime.run_stream(
                input=input_text,
                prompt=prompt,
                thinking=thinking,
            ):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

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

    @app.websocket("/ws")
    async def websocket_endpoint(websocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                input_text = data.get("input", "")
                prompt = data.get("prompt")
                thinking = data.get("thinking", "normal")

                runtime = get_runtime()
                for token in runtime.run_stream(
                    input=input_text,
                    prompt=prompt,
                    thinking=thinking,
                ):
                    await websocket.send_json({"type": "token", "data": token})
                await websocket.send_json({"type": "done"})
        except Exception:
            pass

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "1.0.0"}

    @app.get("/routes")
    async def routes() -> dict[str, list[str]]:
        from omniff.graph.planner import GraphPlanner

        planner = GraphPlanner()
        return {"routes": planner.available_routes()}

    @app.get("/status")
    async def status() -> dict[str, Any]:
        runtime = get_runtime()
        loaded = []
        for name in runtime.models.list():
            m = runtime.models.get(name)
            loaded.append({"name": name, "loaded": m.is_loaded if m else False})

        gpu_info = []
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    free, total = torch.cuda.mem_get_info(i)
                    gpu_info.append(
                        {
                            "id": i,
                            "name": props.name,
                            "total_gb": round(total / 1024**3, 1),
                            "free_gb": round(free / 1024**3, 1),
                            "used_gb": round((total - free) / 1024**3, 1),
                        }
                    )
        except ImportError:
            pass

        return {
            "models": loaded,
            "gpus": gpu_info,
            "request_count": runtime._request_count,
        }

    @app.get("/metrics")
    async def metrics() -> str:
        from fastapi.responses import PlainTextResponse

        runtime = get_runtime()
        lines = [
            "# HELP omniff_requests_total Total requests processed",
            "# TYPE omniff_requests_total counter",
            f"omniff_requests_total {runtime._request_count}",
        ]

        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(i)
                    used = total - free
                    lines.append("# HELP omniff_gpu_memory_used_bytes GPU memory used")
                    lines.append("# TYPE omniff_gpu_memory_used_bytes gauge")
                    lines.append(f'omniff_gpu_memory_used_bytes{{gpu="{i}"}} {used}')
                    lines.append("# HELP omniff_gpu_memory_total_bytes GPU memory total")
                    lines.append("# TYPE omniff_gpu_memory_total_bytes gauge")
                    lines.append(f'omniff_gpu_memory_total_bytes{{gpu="{i}"}} {total}')
        except ImportError:
            pass

        loaded_count = sum(
            1 for name in runtime.models.list() if (m := runtime.models.get(name)) and m.is_loaded
        )
        lines.append("# HELP omniff_models_loaded Number of loaded models")
        lines.append("# TYPE omniff_models_loaded gauge")
        lines.append(f"omniff_models_loaded {loaded_count}")

        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")

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
