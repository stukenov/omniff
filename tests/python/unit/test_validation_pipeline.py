from omniff.validators.pipeline import ValidationPipeline, PipelineResult
from omniff.validators.text_validator import TextValidator, ValidationResult


def _always_pass(output):
    return ValidationResult(True, 1.0, "pass")


def _always_fail(output):
    return ValidationResult(False, 0.0, "fail", "forced failure")


def test_empty_pipeline():
    pipe = ValidationPipeline()
    result = pipe.run({"text": "hello"})
    assert result.passed
    assert result.results == []


def test_single_pass_success():
    pipe = ValidationPipeline()
    pipe.add_pass("length", TextValidator(min_length=1).validate)
    result = pipe.run({"text": "hello world"})
    assert result.passed
    assert len(result.results) == 1


def test_single_pass_failure():
    pipe = ValidationPipeline()
    pipe.add_pass("length", TextValidator(min_length=100).validate, required=True)
    result = pipe.run({"text": "short"})
    assert not result.passed
    assert result.failed_pass == "length"


def test_multi_pass_stops_on_required_failure():
    pipe = ValidationPipeline()
    pipe.add_pass("first", _always_fail, required=True)
    pipe.add_pass("second", _always_pass, required=True)
    result = pipe.run({})
    assert not result.passed
    assert len(result.results) == 1
    assert result.failed_pass == "first"


def test_optional_failure_continues():
    pipe = ValidationPipeline()
    pipe.add_pass("optional", _always_fail, required=False)
    pipe.add_pass("required", _always_pass, required=True)
    result = pipe.run({})
    assert result.passed
    assert len(result.results) == 2


def test_pipeline_score():
    pipe = ValidationPipeline()
    pipe.add_pass("a", _always_pass)
    pipe.add_pass("b", _always_pass)
    result = pipe.run({})
    assert result.score == 1.0


def test_pass_count():
    pipe = ValidationPipeline()
    pipe.add_pass("a", _always_pass)
    pipe.add_pass("b", _always_fail)
    assert pipe.pass_count == 2
