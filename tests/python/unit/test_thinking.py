from omniff.thinking import ThinkingLevel, PromptControl


def test_thinking_levels():
    assert ThinkingLevel.OFF.value == "off"
    assert ThinkingLevel.RESEARCH.value == "research"


def test_from_level_normal():
    pc = PromptControl.from_level("normal")
    assert pc.thinking == ThinkingLevel.NORMAL
    assert pc.max_tokens == 512
    assert pc.temperature == 0.7


def test_from_level_deep():
    pc = PromptControl.from_level("deep")
    assert pc.thinking == ThinkingLevel.DEEP
    assert pc.max_tokens == 1024


def test_from_level_off():
    pc = PromptControl.from_level("off")
    assert not pc.enable_model_thinking


def test_from_level_fast():
    pc = PromptControl.from_level("fast")
    assert not pc.enable_model_thinking


def test_from_level_normal_enables_thinking():
    pc = PromptControl.from_level("normal")
    assert pc.enable_model_thinking


def test_from_level_invalid_defaults_normal():
    pc = PromptControl.from_level("invalid")
    assert pc.thinking == ThinkingLevel.NORMAL
