from omniff.scaling.compile import should_compile


def test_should_compile_cold():
    assert not should_compile("llm", hot_threshold=10, request_count=5)


def test_should_compile_hot():
    assert should_compile("llm", hot_threshold=10, request_count=15)


def test_should_compile_threshold():
    assert should_compile("llm", hot_threshold=10, request_count=10)
