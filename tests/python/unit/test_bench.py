import time
from omniff.bench.suite import BenchmarkSuite
from omniff.bench.profiler import LatencyProfiler
from omniff.bench.recommend import recommend_models, format_recommendation, MODELS


def test_benchmark_suite():
    suite = BenchmarkSuite(name="test")
    result = suite.run_benchmark(
        pipeline="text",
        fn=lambda: {"text": "hello"},
        warmup=0,
        iterations=2,
    )
    assert result.pipeline == "text"
    assert result.latency_ms >= 0


def test_benchmark_with_metric():
    suite = BenchmarkSuite()
    result = suite.run_benchmark(
        pipeline="test",
        fn=lambda: {"score": 0.95},
        metric_fn=lambda r: ("accuracy", r["score"]),
        warmup=0,
        iterations=1,
    )
    assert result.metric_name == "accuracy"
    assert result.metric_value == 0.95


def test_benchmark_summary():
    suite = BenchmarkSuite(name="test")
    suite.run_benchmark("pipe1", lambda: {}, warmup=0, iterations=1)
    summary = suite.summary()
    assert "pipe1" in summary


def test_benchmark_to_dict():
    suite = BenchmarkSuite()
    suite.run_benchmark("pipe1", lambda: {}, warmup=0, iterations=1)
    d = suite.to_dict()
    assert len(d) == 1
    assert d[0]["pipeline"] == "pipe1"


def test_profiler():
    p = LatencyProfiler()
    p.start("total")
    time.sleep(0.01)
    p.start("inner")
    time.sleep(0.01)
    p.stop()
    p.stop()
    assert len(p.entries) == 2
    assert p.entries[0].name == "inner"
    assert p.entries[1].name == "total"


def test_profiler_render():
    p = LatencyProfiler()
    p.start("op")
    p.stop()
    output = p.render()
    assert "op" in output
    assert "ms" in output


def test_profiler_empty():
    p = LatencyProfiler()
    assert p.render() == "(no profile data)"


def test_recommend_small_budget():
    models = recommend_models(5.0, ["text"])
    assert len(models) >= 1
    total = sum(m.vram_gb for m in models)
    assert total <= 5.0


def test_recommend_large_budget():
    models = recommend_models(40.0, ["text", "image", "audio"])
    assert len(models) >= 2


def test_recommend_specific_pipelines():
    models = recommend_models(10.0, ["audio"])
    for m in models:
        assert "audio" in m.pipelines


def test_format_recommendation():
    models = recommend_models(22.0)
    output = format_recommendation(models, 22.0)
    assert "22" in output
    assert "VRAM" in output


def test_models_catalog():
    assert len(MODELS) >= 5
    for m in MODELS:
        assert m.vram_gb > 0
        assert len(m.pipelines) > 0
