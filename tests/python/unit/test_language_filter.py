from omniff.filters.language import detect_language


def test_detect_english():
    assert detect_language("Hello world") == "en"


def test_detect_russian():
    assert detect_language("Привет мир") == "ru"


def test_detect_kazakh():
    assert detect_language("Сәлем әлем") == "kk"


def test_detect_kazakh_specific_chars():
    assert detect_language("Қазақстан") == "kk"


def test_detect_empty():
    assert detect_language("") == "unknown"


def test_detect_numbers_only():
    assert detect_language("12345") == "unknown"
