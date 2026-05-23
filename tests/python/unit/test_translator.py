"""Tests for TranslatorModel (NLLB-200)."""

from omniff.models.translator import TranslatorModel, _resolve_lang


def test_translator_interface():
    model = TranslatorModel()
    assert model.model_id == "facebook/nllb-200-distilled-1.3B"  # default
    assert not model.is_loaded


def test_translator_hymt_model_id():
    model = TranslatorModel(model_id="tencent/Hy-MT2-30B-A3B")
    assert model.model_id == "tencent/Hy-MT2-30B-A3B"
    assert not model.is_loaded


def test_translator_infer_not_loaded():
    model = TranslatorModel()
    try:
        model.infer({"text": "hello"})
        assert False
    except RuntimeError:
        pass


def test_resolve_lang_codes():
    assert _resolve_lang("en") == "eng_Latn"
    assert _resolve_lang("ru") == "rus_Cyrl"
    assert _resolve_lang("kk") == "kaz_Cyrl"
    assert _resolve_lang("zh") == "zho_Hans"
    assert _resolve_lang("fr") == "fra_Latn"
    assert _resolve_lang("de") == "deu_Latn"
    assert _resolve_lang("ja") == "jpn_Jpan"
    assert _resolve_lang("ko") == "kor_Hang"


def test_resolve_lang_full_names():
    assert _resolve_lang("english") == "eng_Latn"
    assert _resolve_lang("russian") == "rus_Cyrl"
    assert _resolve_lang("kazakh") == "kaz_Cyrl"
    assert _resolve_lang("chinese") == "zho_Hans"


def test_resolve_lang_nllb_codes():
    assert _resolve_lang("eng_Latn") == "eng_Latn"
    assert _resolve_lang("rus_Cyrl") == "rus_Cyrl"


def test_resolve_lang_unknown():
    assert _resolve_lang("unknown_xyz") == "eng_Latn"


def test_resolve_lang_hymt():
    from omniff.models.translator import _resolve_lang_hymt

    assert _resolve_lang_hymt("en") == "English"
    assert _resolve_lang_hymt("ru") == "Russian"
    assert _resolve_lang_hymt("kk") == "Kazakh"
    assert _resolve_lang_hymt("zh") == "Chinese"
    assert _resolve_lang_hymt("unknown") == "Unknown"
