import time
from omniff.observability import RequestTrace, get_logger, setup_logging


def test_request_trace_creates_id():
    trace = RequestTrace()
    assert len(trace.request_id) == 12


def test_span_measures_time():
    trace = RequestTrace()
    with trace.span("test_op") as s:
        time.sleep(0.01)
    assert len(trace.spans) == 1
    assert trace.spans[0].name == "test_op"
    assert trace.spans[0].duration_ms >= 5


def test_timing_breakdown():
    trace = RequestTrace()
    with trace.span("a"):
        pass
    with trace.span("b"):
        pass
    bd = trace.timing_breakdown()
    assert "a" in bd
    assert "b" in bd


def test_total_ms():
    trace = RequestTrace()
    time.sleep(0.01)
    assert trace.total_ms >= 5


def test_span_metadata():
    trace = RequestTrace()
    with trace.span("load", model="test-model") as s:
        pass
    assert trace.spans[0].metadata["model"] == "test-model"


def test_get_logger():
    log = get_logger("test")
    assert log.name == "omniff.test"


def test_get_logger_root():
    log = get_logger()
    assert log.name == "omniff"


def test_setup_logging():
    setup_logging(level="DEBUG")
    log = get_logger("test_setup")
    assert log.getEffectiveLevel() == 10
