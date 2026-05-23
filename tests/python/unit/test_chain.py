import pytest
from omniff.graph.chain import load_chain, list_chains


def test_load_chain(tmp_path):
    chain_file = tmp_path / "voice.yaml"
    chain_file.write_text("""
id: voice_chain
name: Voice Pipeline
description: Audio to audio via LLM
nodes:
  - id: asr
    type: asr_infer
    config: {}
  - id: llm
    type: llm_infer
    config: {}
  - id: tts
    type: tts_infer
    config: {}
edges:
  - from: asr
    to: llm
  - from: llm
    to: tts
""")
    graph = load_chain(chain_file)
    assert graph.id == "voice_chain"
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2


def test_list_chains(tmp_path):
    (tmp_path / "a.yaml").write_text("id: a\nname: Chain A\nnodes: []\nedges: []")
    (tmp_path / "b.yaml").write_text("id: b\nname: Chain B\nnodes: []\nedges: []")
    chains = list_chains(tmp_path)
    assert len(chains) == 2
    names = {c["id"] for c in chains}
    assert "a" in names and "b" in names


def test_list_chains_empty(tmp_path):
    chains = list_chains(tmp_path)
    assert chains == []


def test_list_chains_nonexistent():
    chains = list_chains("/nonexistent/path")
    assert chains == []
