import pytest
from pathlib import Path

from omniff.runtime.config import OmniFFConfig


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
