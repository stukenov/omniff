import pytest

from omniff.models.base import OmniModel
from omniff.models.registry import ModelRegistry


class FakeModel(OmniModel):
    def __init__(self):
        self.loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def infer(self, inputs):
        return {"echo": inputs}


def test_register_and_get():
    reg = ModelRegistry()
    model = FakeModel()
    reg.register("test_model", model)
    assert reg.get("test_model") is model


def test_get_missing():
    reg = ModelRegistry()
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_load_model():
    reg = ModelRegistry()
    model = FakeModel()
    reg.register("m", model)
    reg.load("m")
    assert model.loaded


def test_unload_model():
    reg = ModelRegistry()
    model = FakeModel()
    reg.register("m", model)
    reg.load("m")
    reg.unload("m")
    assert not model.loaded


def test_list_models():
    reg = ModelRegistry()
    reg.register("a", FakeModel())
    reg.register("b", FakeModel())
    assert set(reg.list()) == {"a", "b"}
