import pytest
from pathlib import Path

from omniff.runtime.config import OmniFFConfig, RouterConfig, ExpertConfig


def test_load_config_from_yaml(tmp_path):
    config_file = tmp_path / "omniff.yaml"
    config_file.write_text("""
name: test-runtime
version: "0.1"

router:
  type: keyword
  path: ""

experts:
  text_small:
    name: text_small
    model_type: causal_lm
    path: models/llm_small
    loading: hot
""")
    config = OmniFFConfig.load(config_file)
    assert config.name == "test-runtime"
    assert config.version == "0.1"
    assert config.router.router_type == "keyword"
    assert "text_small" in config.experts
    assert config.experts["text_small"].model_type == "causal_lm"
    assert config.experts["text_small"].loading == "hot"


def test_config_missing_file():
    with pytest.raises(FileNotFoundError):
        OmniFFConfig.load(Path("/nonexistent/omniff.yaml"))


def test_config_expert_defaults(tmp_path):
    config_file = tmp_path / "omniff.yaml"
    config_file.write_text("""
name: minimal
version: "0.1"
router:
  type: keyword
  path: ""
experts: {}
""")
    config = OmniFFConfig.load(config_file)
    assert config.experts == {}
    assert config.graph_templates_dir is None


def test_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "omniff.yaml"
    config_file.write_text(": : : not valid yaml [[[")
    with pytest.raises(Exception):
        OmniFFConfig.load(config_file)


def test_config_missing_required_field(tmp_path):
    config_file = tmp_path / "omniff.yaml"
    config_file.write_text("""
version: "0.1"
router:
  type: keyword
""")
    with pytest.raises((KeyError, TypeError, Exception)):
        OmniFFConfig.load(config_file)


def test_config_missing_router(tmp_path):
    config_file = tmp_path / "omniff.yaml"
    config_file.write_text("""
name: test
version: "0.1"
""")
    with pytest.raises((KeyError, TypeError, Exception)):
        OmniFFConfig.load(config_file)


def test_config_missing_expert_model_type(tmp_path):
    config_file = tmp_path / "omniff.yaml"
    config_file.write_text("""
name: test
version: "0.1"
router:
  type: keyword
  path: ""
experts:
  bad_expert:
    name: bad
    path: models/bad
""")
    with pytest.raises((KeyError, Exception)):
        OmniFFConfig.load(config_file)


def test_expert_config_defaults():
    expert = ExpertConfig(
        name="test", model_type="causal_lm", path="models/test"
    )
    assert expert.loading == "warm"
    assert expert.quantization is None
    assert expert.device is None


def test_router_config_defaults():
    router = RouterConfig(router_type="keyword")
    assert router.path == ""


def test_config_graph_templates_dir(tmp_path):
    config_file = tmp_path / "omniff.yaml"
    config_file.write_text("""
name: test
version: "0.1"
router:
  type: keyword
  path: ""
graph_templates_dir: /tmp/templates
experts: {}
""")
    config = OmniFFConfig.load(config_file)
    assert config.graph_templates_dir == "/tmp/templates"
