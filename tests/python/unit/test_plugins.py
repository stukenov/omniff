import pytest

from omniff.models.base import OmniModel
from omniff.plugins import ModelPlugin, PluginRegistry


class DummyModel(OmniModel):
    def __init__(self, model_id="dummy", device="cpu"):
        self.model_id = model_id
        self.device = device
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def infer(self, inputs):
        return {"text": "dummy"}


def test_register_and_get():
    reg = PluginRegistry()
    plugin = ModelPlugin("test", DummyModel, "TEXT_SIMPLE")
    reg.register(plugin)
    assert reg.has("test")
    assert reg.get("test") is plugin


def test_get_missing():
    reg = PluginRegistry()
    with pytest.raises(KeyError, match="not registered"):
        reg.get("nope")


def test_list_plugins():
    reg = PluginRegistry()
    reg.register(ModelPlugin("a", DummyModel, "TEXT_SIMPLE"))
    reg.register(ModelPlugin("b", DummyModel, "IMAGE_CAPTION"))
    assert sorted(reg.list()) == ["a", "b"]


def test_create_model():
    reg = PluginRegistry()
    reg.register(ModelPlugin("test", DummyModel, "TEXT_SIMPLE", {"model_id": "default"}))
    model = reg.create_model("test")
    assert model.model_id == "default"


def test_create_model_with_overrides():
    reg = PluginRegistry()
    reg.register(ModelPlugin("test", DummyModel, "TEXT_SIMPLE", {"model_id": "default"}))
    model = reg.create_model("test", model_id="custom")
    assert model.model_id == "custom"
