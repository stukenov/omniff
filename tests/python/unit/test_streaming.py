import pytest


def test_llm_infer_stream_not_loaded():
    from omniff.models.llm import LLMModel

    model = LLMModel(model_id="test")
    with pytest.raises(RuntimeError, match="not loaded"):
        list(model.infer_stream({"prompt": "hi"}))


def test_run_stream_is_generator():
    import types

    from omniff.runtime.config import OmniFFConfig, RouterConfig
    from omniff.runtime.engine import OmniFFRuntime

    config = OmniFFConfig(
        name="test",
        version="1.0",
        router=RouterConfig(router_type="keyword", path=""),
    )
    runtime = OmniFFRuntime(config)
    gen = runtime.run_stream(input="hello")
    assert isinstance(gen, types.GeneratorType)


def test_cli_has_stream_flag():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "omniff.cli", "run", "--help"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "python", "PATH": ""},
        cwd=str(__import__("pathlib").Path(__file__).parents[3]),
    )
    assert "--stream" in result.stdout
