# OmniFF MVP (v0.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build working OmniFF runtime that handles text→text, image→text, audio→text, and image→image pipelines, deployed and tested on KazNU server (2x A10 22GB).

**Architecture:** Python-first MVP — implement model wrappers, router, graph executor, validators, and CLI in Python. Rust crates remain as scaffold for future performance layer. Each pipeline is a YAML graph template executed by the runtime.

**Tech Stack:** Python 3.10, PyTorch 2.5.1, transformers 4.57.6, diffusers (to install), Qwen3-4B (LLM), Qwen2.5-VL-3B (VLM), Whisper-large-v3 (ASR), SDXL/Flux (image edit), pytest.

**Server:** KazNU — `ssh -p 15126 root@164.138.46.36`, 2x A10 22GB, 119GB free disk, project_dir=/root/omniff

---

## File Structure

```
python/omniff/
├── __init__.py                    # package root, exports OmniFFRuntime
├── runtime/
│   ├── __init__.py
│   ├── engine.py                  # OmniFFRuntime.run(), .plan(), .execute()
│   ├── config.py                  # OmniFFConfig — load omniff.yaml
│   └── result.py                  # RunResult dataclass
├── router/
│   ├── __init__.py
│   └── keyword_router.py         # KeywordRouter (MVP), RouteDecision
├── graph/
│   ├── __init__.py
│   ├── types.py                   # OmniGraph, OmniNode, Edge (exists)
│   ├── executor.py                # GraphExecutor — topological DAG walk
│   └── loader.py                  # load graph templates from YAML
├── models/
│   ├── __init__.py
│   ├── base.py                    # OmniModel ABC (exists)
│   ├── registry.py                # ModelRegistry — load/unload/get by name
│   ├── llm.py                     # LLMModel — Qwen3 causal LM wrapper
│   ├── vlm.py                     # VLMModel — Qwen2.5-VL wrapper
│   ├── asr.py                     # ASRModel — Whisper wrapper
│   └── image_edit.py              # ImageEditModel — diffusion img2img
├── filters/
│   ├── __init__.py
│   └── language.py                # detect_language filter
├── validators/
│   ├── __init__.py
│   ├── text_validator.py          # TextValidator — length, language, format
│   └── image_validator.py         # ImageValidator — size, format, basic checks
├── nodes/
│   ├── __init__.py
│   └── registry.py                # NodeRegistry — maps node_type → callable
└── cli.py                         # Python CLI entry point

tests/python/
├── conftest.py                    # fixtures, test config
├── unit/
│   ├── test_config.py
│   ├── test_router.py
│   ├── test_graph_types.py
│   ├── test_graph_executor.py
│   ├── test_model_registry.py
│   └── test_validators.py
├── integration/
│   ├── test_text_to_text.py
│   ├── test_image_to_text.py
│   ├── test_audio_to_text.py
│   └── test_image_to_image.py
└── fixtures/
    ├── test_image.jpg             # small test image
    ├── test_audio.wav             # short test audio clip
    └── omniff_test.yaml           # test config
```

---

## Task 1: Config System

**Files:**
- Create: `python/omniff/runtime/config.py`
- Create: `tests/python/unit/test_config.py`
- Create: `tests/python/conftest.py`
- Modify: `python/omniff/runtime/engine.py`

- [ ] **Step 1: Write failing test for config loading**

```python
# tests/python/unit/test_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sakentukenov/omiff && python -m pytest tests/python/unit/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'omniff.runtime.config'`

- [ ] **Step 3: Write conftest.py**

```python
# tests/python/conftest.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
```

- [ ] **Step 4: Implement config module**

```python
# python/omniff/runtime/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RouterConfig:
    router_type: str
    path: str


@dataclass
class ExpertConfig:
    name: str
    model_type: str
    path: str
    loading: str = "warm"
    quantization: str | None = None
    device: str | None = None


@dataclass
class OmniFFConfig:
    name: str
    version: str
    router: RouterConfig
    experts: dict[str, ExpertConfig] = field(default_factory=dict)
    graph_templates_dir: str | None = None

    @classmethod
    def load(cls, path: Path) -> OmniFFConfig:
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)

        router = RouterConfig(
            router_type=raw["router"]["type"],
            path=raw["router"].get("path", ""),
        )

        experts = {}
        for name, spec in raw.get("experts", {}).items():
            experts[name] = ExpertConfig(
                name=spec.get("name", name),
                model_type=spec["model_type"],
                path=spec["path"],
                loading=spec.get("loading", "warm"),
                quantization=spec.get("quantization"),
                device=spec.get("device"),
            )

        return cls(
            name=raw["name"],
            version=raw["version"],
            router=router,
            experts=experts,
            graph_templates_dir=raw.get("graph_templates_dir"),
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/sakentukenov/omiff && python -m pytest tests/python/unit/test_config.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add python/omniff/runtime/config.py tests/python/conftest.py tests/python/unit/test_config.py
git commit -m "feat(config): add OmniFFConfig with YAML loading and tests"
```

---

## Task 2: Router

**Files:**
- Create: `python/omniff/router/__init__.py`
- Create: `python/omniff/router/keyword_router.py`
- Create: `tests/python/unit/test_router.py`

- [ ] **Step 1: Write failing test**

```python
# tests/python/unit/test_router.py
from omniff.router.keyword_router import KeywordRouter, RouteDecision


def test_route_text_simple():
    router = KeywordRouter()
    decision = router.route("Привет, как дела?")
    assert decision.route_class == "TEXT_SIMPLE"
    assert decision.confidence > 0.0


def test_route_image_caption():
    router = KeywordRouter()
    decision = router.route("describe this image", input_modality="image")
    assert decision.route_class == "IMAGE_CAPTION"


def test_route_audio_transcribe():
    router = KeywordRouter()
    decision = router.route("", input_modality="audio")
    assert decision.route_class == "AUDIO_TRANSCRIBE_ONLY"


def test_route_image_edit():
    router = KeywordRouter()
    decision = router.route(
        "make it matte graphite",
        input_modality="image",
        output_modality="image",
    )
    assert decision.route_class == "IMAGE_EDIT"


def test_route_text_complex():
    router = KeywordRouter()
    decision = router.route(
        "Проанализируй этот контракт на юридические риски и составь подробное резюме "
        "с указанием всех потенциальных проблем"
    )
    assert decision.route_class in ("TEXT_COMPLEX", "TEXT_NORMAL")


def test_route_decision_fields():
    router = KeywordRouter()
    decision = router.route("hello")
    assert isinstance(decision, RouteDecision)
    assert hasattr(decision, "route_class")
    assert hasattr(decision, "confidence")
    assert hasattr(decision, "thinking")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/python/unit/test_router.py -v`
Expected: FAIL

- [ ] **Step 3: Implement router**

```python
# python/omniff/router/__init__.py
from omniff.router.keyword_router import KeywordRouter, RouteDecision

__all__ = ["KeywordRouter", "RouteDecision"]
```

```python
# python/omniff/router/keyword_router.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouteDecision:
    route_class: str
    confidence: float
    thinking: str = "normal"


COMPLEX_KEYWORDS = [
    "проанализируй", "analyze", "research", "compare", "evaluate",
    "контракт", "contract", "legal", "юридич",
]


class KeywordRouter:
    def route(
        self,
        prompt: str,
        input_modality: str = "text",
        output_modality: str | None = None,
    ) -> RouteDecision:
        output_modality = output_modality or "text"
        prompt_lower = prompt.lower()

        if input_modality == "audio":
            if prompt.strip():
                return RouteDecision("AUDIO_QA", 0.8, "normal")
            return RouteDecision("AUDIO_TRANSCRIBE_ONLY", 0.9, "fast")

        if input_modality == "image" and output_modality == "image":
            return RouteDecision("IMAGE_EDIT", 0.85, "normal")

        if input_modality == "image":
            return RouteDecision("IMAGE_CAPTION", 0.85, "fast")

        if output_modality == "image":
            return RouteDecision("TEXT_TO_IMAGE", 0.8, "normal")

        if any(kw in prompt_lower for kw in COMPLEX_KEYWORDS):
            return RouteDecision("TEXT_COMPLEX", 0.7, "deep")

        if len(prompt.split()) > 50:
            return RouteDecision("TEXT_NORMAL", 0.6, "normal")

        return RouteDecision("TEXT_SIMPLE", 0.8, "fast")
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/python/unit/test_router.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add python/omniff/router/ tests/python/unit/test_router.py
git commit -m "feat(router): add KeywordRouter for MVP routing"
```

---

## Task 3: Graph Executor

**Files:**
- Create: `python/omniff/graph/executor.py`
- Create: `python/omniff/graph/loader.py`
- Create: `python/omniff/nodes/registry.py`
- Create: `tests/python/unit/test_graph_executor.py`
- Modify: `python/omniff/graph/types.py`

- [ ] **Step 1: Write failing test**

```python
# tests/python/unit/test_graph_executor.py
import pytest

from omniff.graph.types import OmniGraph, OmniNode, Edge
from omniff.graph.executor import GraphExecutor
from omniff.nodes.registry import NodeRegistry


def make_test_graph():
    g = OmniGraph(id="test")
    g.add_node(OmniNode(id="step1", node_type="echo", config={"value": "hello"}))
    g.add_node(OmniNode(id="step2", node_type="echo", config={"value": "world"}))
    g.add_node(OmniNode(id="step3", node_type="concat", config={}))
    g.add_edge("step1", "step3")
    g.add_edge("step2", "step3")
    return g


def echo_node(inputs: dict, config: dict) -> dict:
    return {"text": config["value"]}


def concat_node(inputs: dict, config: dict) -> dict:
    texts = [v["text"] for v in inputs.values() if "text" in v]
    return {"text": " ".join(texts)}


def test_executor_topological_order():
    graph = make_test_graph()
    registry = NodeRegistry()
    registry.register("echo", echo_node)
    registry.register("concat", concat_node)
    executor = GraphExecutor(registry)
    result = executor.execute(graph, {})
    assert result["step3"]["text"] in ("hello world", "world hello")


def test_executor_single_node():
    g = OmniGraph(id="single")
    g.add_node(OmniNode(id="only", node_type="echo", config={"value": "test"}))
    registry = NodeRegistry()
    registry.register("echo", echo_node)
    executor = GraphExecutor(registry)
    result = executor.execute(g, {})
    assert result["only"]["text"] == "test"


def test_executor_missing_node_type():
    g = OmniGraph(id="bad")
    g.add_node(OmniNode(id="n1", node_type="nonexistent", config={}))
    registry = NodeRegistry()
    executor = GraphExecutor(registry)
    with pytest.raises(KeyError, match="nonexistent"):
        executor.execute(g, {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/python/unit/test_graph_executor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement NodeRegistry**

```python
# python/omniff/nodes/__init__.py
from omniff.nodes.registry import NodeRegistry

__all__ = ["NodeRegistry"]
```

```python
# python/omniff/nodes/registry.py
from __future__ import annotations

from typing import Any, Callable

NodeFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class NodeRegistry:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeFn] = {}

    def register(self, name: str, fn: NodeFn) -> None:
        self._nodes[name] = fn

    def get(self, name: str) -> NodeFn:
        if name not in self._nodes:
            raise KeyError(f"Node type not registered: {name}")
        return self._nodes[name]

    def has(self, name: str) -> bool:
        return name in self._nodes
```

- [ ] **Step 4: Implement GraphExecutor**

```python
# python/omniff/graph/executor.py
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from omniff.graph.types import OmniGraph
from omniff.nodes.registry import NodeRegistry


class GraphExecutor:
    def __init__(self, registry: NodeRegistry) -> None:
        self.registry = registry

    def execute(self, graph: OmniGraph, initial_inputs: dict[str, Any]) -> dict[str, Any]:
        order = self._topological_sort(graph)
        results: dict[str, Any] = {}

        for node in order:
            fn = self.registry.get(node.node_type)
            node_inputs = {}
            for edge in graph.edges:
                if edge.target == node.id and edge.source in results:
                    node_inputs[edge.source] = results[edge.source]
            if not node_inputs and node.id in initial_inputs:
                node_inputs = initial_inputs[node.id]
            results[node.id] = fn(node_inputs, node.config)

        return results

    def _topological_sort(self, graph: OmniGraph) -> list:
        in_degree: dict[str, int] = defaultdict(int)
        adj: dict[str, list[str]] = defaultdict(list)
        node_map = {n.id: n for n in graph.nodes}

        for n in graph.nodes:
            in_degree.setdefault(n.id, 0)
        for e in graph.edges:
            adj[e.source].append(e.target)
            in_degree[e.target] += 1

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        result = []
        while queue:
            nid = queue.popleft()
            result.append(node_map[nid])
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(graph.nodes):
            raise ValueError("Graph has a cycle")
        return result
```

- [ ] **Step 5: Implement graph loader**

```python
# python/omniff/graph/loader.py
from __future__ import annotations

from pathlib import Path

import yaml

from omniff.graph.types import OmniGraph, OmniNode, Edge


def load_graph_template(path: Path) -> OmniGraph:
    with open(path) as f:
        raw = yaml.safe_load(f)

    g_raw = raw["graph"]
    graph = OmniGraph(id=g_raw["id"])

    for n in g_raw.get("nodes", []):
        node_type = n["node_type"]
        if isinstance(node_type, dict):
            for key, val in node_type.items():
                node_type = key
                break
        graph.add_node(OmniNode(
            id=n["id"],
            node_type=node_type,
            config=n.get("config", {}),
        ))

    for e in g_raw.get("edges", []):
        graph.add_edge(e["from"], e["to"])

    return graph
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/python/unit/test_graph_executor.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add python/omniff/graph/executor.py python/omniff/graph/loader.py python/omniff/nodes/registry.py tests/python/unit/test_graph_executor.py
git commit -m "feat(graph): add GraphExecutor with topological sort and NodeRegistry"
```

---

## Task 4: Model Base + Registry

**Files:**
- Create: `python/omniff/models/registry.py`
- Create: `tests/python/unit/test_model_registry.py`
- Modify: `python/omniff/models/base.py`

- [ ] **Step 1: Write failing test**

```python
# tests/python/unit/test_model_registry.py
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
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement ModelRegistry**

```python
# python/omniff/models/registry.py
from __future__ import annotations

from omniff.models.base import OmniModel


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, OmniModel] = {}

    def register(self, name: str, model: OmniModel) -> None:
        self._models[name] = model

    def get(self, name: str) -> OmniModel:
        if name not in self._models:
            raise KeyError(f"Model not registered: {name}")
        return self._models[name]

    def load(self, name: str) -> None:
        self.get(name).load()

    def unload(self, name: str) -> None:
        self.get(name).unload()

    def list(self) -> list[str]:
        return list(self._models.keys())

    def has(self, name: str) -> bool:
        return name in self._models
```

- [ ] **Step 4: Run test — expect 5 passed**

- [ ] **Step 5: Commit**

```bash
git add python/omniff/models/registry.py tests/python/unit/test_model_registry.py
git commit -m "feat(models): add ModelRegistry for model lifecycle management"
```

---

## Task 5: Validators

**Files:**
- Create: `python/omniff/validators/text_validator.py`
- Create: `python/omniff/validators/image_validator.py`
- Create: `tests/python/unit/test_validators.py`

- [ ] **Step 1: Write failing test**

```python
# tests/python/unit/test_validators.py
from omniff.validators.text_validator import TextValidator
from omniff.validators.image_validator import ImageValidator


def test_text_validator_pass():
    v = TextValidator(min_length=1)
    result = v.validate({"text": "Hello world"})
    assert result.passed
    assert result.score > 0.5


def test_text_validator_fail_empty():
    v = TextValidator(min_length=1)
    result = v.validate({"text": ""})
    assert not result.passed


def test_text_validator_fail_none():
    v = TextValidator()
    result = v.validate({})
    assert not result.passed


def test_image_validator_pass(tmp_path):
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    v = ImageValidator()
    result = v.validate({"image_path": str(img_path)})
    assert result.passed


def test_image_validator_fail_missing():
    v = ImageValidator()
    result = v.validate({"image_path": "/nonexistent.jpg"})
    assert not result.passed
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement validators**

```python
# python/omniff/validators/text_validator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    passed: bool
    score: float
    validator: str
    details: str | None = None


class TextValidator:
    def __init__(self, min_length: int = 0, max_length: int | None = None) -> None:
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, output: dict[str, Any]) -> ValidationResult:
        text = output.get("text")
        if text is None:
            return ValidationResult(False, 0.0, "text", "no text in output")
        if len(text) < self.min_length:
            return ValidationResult(False, 0.1, "text", f"too short: {len(text)} < {self.min_length}")
        if self.max_length and len(text) > self.max_length:
            return ValidationResult(False, 0.2, "text", f"too long: {len(text)} > {self.max_length}")
        return ValidationResult(True, 1.0, "text")
```

```python
# python/omniff/validators/image_validator.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from omniff.validators.text_validator import ValidationResult


class ImageValidator:
    def __init__(self, min_size: int = 10) -> None:
        self.min_size = min_size

    def validate(self, output: dict[str, Any]) -> ValidationResult:
        image_path = output.get("image_path")
        if not image_path:
            return ValidationResult(False, 0.0, "image", "no image_path in output")
        p = Path(image_path)
        if not p.exists():
            return ValidationResult(False, 0.0, "image", f"file not found: {image_path}")
        if p.stat().st_size < self.min_size:
            return ValidationResult(False, 0.1, "image", "file too small")
        return ValidationResult(True, 1.0, "image")
```

- [ ] **Step 4: Update validators __init__.py**

```python
# python/omniff/validators/__init__.py
from omniff.validators.text_validator import TextValidator, ValidationResult
from omniff.validators.image_validator import ImageValidator

__all__ = ["TextValidator", "ImageValidator", "ValidationResult"]
```

- [ ] **Step 5: Run test — expect 5 passed**

- [ ] **Step 6: Commit**

```bash
git add python/omniff/validators/ tests/python/unit/test_validators.py
git commit -m "feat(validators): add TextValidator and ImageValidator"
```

---

## Task 6: LLM Model Wrapper

**Files:**
- Create: `python/omniff/models/llm.py`
- Create: `tests/python/unit/test_llm.py` (mock-based unit test)
- Create: `tests/python/integration/test_text_to_text.py` (real model, runs on server)

- [ ] **Step 1: Write failing unit test (mock)**

```python
# tests/python/unit/test_llm.py
import pytest

from omniff.models.llm import LLMModel


def test_llm_interface():
    model = LLMModel(model_id="Qwen/Qwen3-0.6B", device="cpu", max_new_tokens=32)
    assert not model.is_loaded


def test_llm_infer_not_loaded():
    model = LLMModel(model_id="Qwen/Qwen3-0.6B", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"prompt": "hello"})
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement LLMModel**

```python
# python/omniff/models/llm.py
from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel


class LLMModel(OmniModel):
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        max_new_tokens: int = 512,
        torch_dtype: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        dtype = getattr(torch, self.torch_dtype) if self.torch_dtype != "auto" else "auto"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device,
        )

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        prompt = inputs.get("prompt", "")
        messages = inputs.get("messages") or [{"role": "user", "content": prompt}]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        generated = self._model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
        )
        new_tokens = generated[0][model_inputs["input_ids"].shape[-1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {"text": response}
```

- [ ] **Step 4: Run unit test — expect 2 passed**

- [ ] **Step 5: Write integration test (for server)**

```python
# tests/python/integration/test_text_to_text.py
import pytest

from omniff.models.llm import LLMModel


@pytest.fixture(scope="module")
def llm():
    model = LLMModel(model_id="Qwen/Qwen3-4B", device="auto", max_new_tokens=128)
    model.load()
    yield model
    model.unload()


def test_simple_response(llm):
    result = llm.infer({"prompt": "What is 2+2? Answer with just the number."})
    assert "4" in result["text"]


def test_kazakh_response(llm):
    result = llm.infer({"prompt": "Қазақстанның астанасы қай қала? Тек қала атын жаз."})
    assert "Астана" in result["text"] or "астана" in result["text"].lower()


def test_long_prompt(llm):
    result = llm.infer({"prompt": "List 3 colors. One word each, comma separated."})
    assert len(result["text"]) > 3
```

- [ ] **Step 6: Commit**

```bash
git add python/omniff/models/llm.py tests/python/unit/test_llm.py tests/python/integration/test_text_to_text.py
git commit -m "feat(models): add LLMModel wrapper for causal LM inference"
```

---

## Task 7: VLM Model Wrapper

**Files:**
- Create: `python/omniff/models/vlm.py`
- Create: `tests/python/unit/test_vlm.py`
- Create: `tests/python/integration/test_image_to_text.py`

- [ ] **Step 1: Write failing unit test**

```python
# tests/python/unit/test_vlm.py
import pytest

from omniff.models.vlm import VLMModel


def test_vlm_interface():
    model = VLMModel(model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="cpu")
    assert not model.is_loaded


def test_vlm_infer_not_loaded():
    model = VLMModel(model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"image_path": "test.jpg", "prompt": "describe"})
```

- [ ] **Step 2: Implement VLMModel**

```python
# python/omniff/models/vlm.py
from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel


class VLMModel(OmniModel):
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=self.device,
        )

    def unload(self) -> None:
        del self._model, self._processor
        self._model = None
        self._processor = None

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        from qwen_vl_utils import process_vision_info

        image_path = inputs.get("image_path", "")
        prompt = inputs.get("prompt", "Describe this image.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._model.device)

        generated = self._model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
        trimmed = generated[0][model_inputs["input_ids"].shape[-1]:]
        response = self._processor.decode(trimmed, skip_special_tokens=True)

        return {"text": response}
```

- [ ] **Step 3: Write integration test**

```python
# tests/python/integration/test_image_to_text.py
import pytest
from pathlib import Path

from omniff.models.vlm import VLMModel


@pytest.fixture(scope="module")
def vlm():
    model = VLMModel(model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="auto")
    model.load()
    yield model
    model.unload()


@pytest.fixture
def test_image(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (256, 256), color=(255, 0, 0))
    path = tmp_path / "red_square.png"
    img.save(path)
    return str(path)


def test_describe_image(vlm, test_image):
    result = vlm.infer({"image_path": test_image, "prompt": "What color is this image? One word."})
    assert "red" in result["text"].lower()


def test_image_qa(vlm, test_image):
    result = vlm.infer({"image_path": test_image, "prompt": "What shape is shown?"})
    assert len(result["text"]) > 0
```

- [ ] **Step 4: Run unit test — expect 2 passed**

- [ ] **Step 5: Commit**

```bash
git add python/omniff/models/vlm.py tests/python/unit/test_vlm.py tests/python/integration/test_image_to_text.py
git commit -m "feat(models): add VLMModel wrapper for vision-language inference"
```

---

## Task 8: ASR Model Wrapper

**Files:**
- Create: `python/omniff/models/asr.py`
- Create: `tests/python/unit/test_asr.py`
- Create: `tests/python/integration/test_audio_to_text.py`

- [ ] **Step 1: Write failing unit test**

```python
# tests/python/unit/test_asr.py
import pytest

from omniff.models.asr import ASRModel


def test_asr_interface():
    model = ASRModel(model_id="openai/whisper-large-v3", device="cpu")
    assert not model.is_loaded


def test_asr_infer_not_loaded():
    model = ASRModel(model_id="openai/whisper-large-v3", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"audio_path": "test.wav"})
```

- [ ] **Step 2: Implement ASRModel**

```python
# python/omniff/models/asr.py
from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel


class ASRModel(OmniModel):
    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        device: str = "auto",
        language: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.language = language
        self._pipe = None

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        import torch
        from transformers import pipeline

        device_arg = 0 if self.device == "auto" and torch.cuda.is_available() else -1
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            device_arg = int(self.device.split(":")[-1]) if ":" in self.device else 0

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            torch_dtype=torch.float16,
            device=device_arg,
        )

    def unload(self) -> None:
        del self._pipe
        self._pipe = None

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        audio_path = inputs.get("audio_path", "")
        generate_kwargs = {}
        lang = inputs.get("language") or self.language
        if lang:
            generate_kwargs["language"] = lang

        result = self._pipe(
            audio_path,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )

        return {
            "text": result["text"],
            "chunks": result.get("chunks", []),
        }
```

- [ ] **Step 3: Write integration test**

```python
# tests/python/integration/test_audio_to_text.py
import pytest
import numpy as np

from omniff.models.asr import ASRModel


@pytest.fixture(scope="module")
def asr():
    model = ASRModel(model_id="openai/whisper-large-v3", device="auto")
    model.load()
    yield model
    model.unload()


@pytest.fixture
def test_audio(tmp_path):
    import soundfile as sf
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "sine.wav"
    sf.write(str(path), audio, sr)
    return str(path)


def test_transcribe_audio(asr, test_audio):
    result = asr.infer({"audio_path": test_audio})
    assert "text" in result
    assert isinstance(result["text"], str)


def test_transcribe_returns_chunks(asr, test_audio):
    result = asr.infer({"audio_path": test_audio})
    assert "chunks" in result
```

- [ ] **Step 4: Run unit test — expect 2 passed**

- [ ] **Step 5: Commit**

```bash
git add python/omniff/models/asr.py tests/python/unit/test_asr.py tests/python/integration/test_audio_to_text.py
git commit -m "feat(models): add ASRModel wrapper for Whisper speech recognition"
```

---

## Task 9: Image Edit Model Wrapper

**Files:**
- Create: `python/omniff/models/image_edit.py`
- Create: `tests/python/unit/test_image_edit.py`
- Create: `tests/python/integration/test_image_to_image.py`

- [ ] **Step 1: Write failing unit test**

```python
# tests/python/unit/test_image_edit.py
import pytest

from omniff.models.image_edit import ImageEditModel


def test_image_edit_interface():
    model = ImageEditModel(model_id="stabilityai/sdxl-turbo", device="cpu")
    assert not model.is_loaded


def test_image_edit_infer_not_loaded():
    model = ImageEditModel(model_id="stabilityai/sdxl-turbo", device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        model.infer({"image_path": "test.jpg", "prompt": "make it blue"})
```

- [ ] **Step 2: Implement ImageEditModel**

```python
# python/omniff/models/image_edit.py
from __future__ import annotations

from typing import Any

from omniff.models.base import OmniModel


class ImageEditModel(OmniModel):
    def __init__(
        self,
        model_id: str = "stabilityai/sdxl-turbo",
        device: str = "auto",
        num_inference_steps: int = 4,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.num_inference_steps = num_inference_steps
        self._pipe = None

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        import torch
        from diffusers import AutoPipelineForImage2Image

        self._pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16" if "turbo" not in self.model_id else None,
        )
        if self.device == "auto":
            self._pipe = self._pipe.to("cuda")
        else:
            self._pipe = self._pipe.to(self.device)

    def unload(self) -> None:
        del self._pipe
        self._pipe = None

    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        from PIL import Image

        image_path = inputs["image_path"]
        prompt = inputs.get("prompt", "")
        negative_prompt = inputs.get("negative_prompt", "")
        strength = inputs.get("strength", 0.5)
        seed = inputs.get("seed")
        output_path = inputs.get("output_path", "output.png")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512))

        generator = None
        if seed is not None:
            import torch
            generator = torch.Generator(device=self._pipe.device).manual_seed(seed)

        result_image = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=image,
            strength=strength,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        ).images[0]

        result_image.save(output_path)
        return {"image_path": output_path}
```

- [ ] **Step 3: Write integration test**

```python
# tests/python/integration/test_image_to_image.py
import pytest
from pathlib import Path

from omniff.models.image_edit import ImageEditModel


@pytest.fixture(scope="module")
def image_edit():
    model = ImageEditModel(model_id="stabilityai/sdxl-turbo", device="auto", num_inference_steps=2)
    model.load()
    yield model
    model.unload()


@pytest.fixture
def test_image(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (512, 512), color=(100, 100, 200))
    path = tmp_path / "input.png"
    img.save(path)
    return str(path)


def test_image_edit_produces_output(image_edit, test_image, tmp_path):
    output_path = str(tmp_path / "output.png")
    result = image_edit.infer({
        "image_path": test_image,
        "prompt": "a beautiful sunset landscape",
        "strength": 0.6,
        "output_path": output_path,
        "seed": 42,
    })
    assert Path(result["image_path"]).exists()
    assert Path(result["image_path"]).stat().st_size > 1000


def test_image_edit_preserves_with_low_strength(image_edit, test_image, tmp_path):
    output_path = str(tmp_path / "output_low.png")
    result = image_edit.infer({
        "image_path": test_image,
        "prompt": "same image but slightly warmer tones",
        "strength": 0.15,
        "output_path": output_path,
        "seed": 42,
    })
    assert Path(result["image_path"]).exists()
```

- [ ] **Step 4: Run unit test — expect 2 passed**

- [ ] **Step 5: Commit**

```bash
git add python/omniff/models/image_edit.py tests/python/unit/test_image_edit.py tests/python/integration/test_image_to_image.py
git commit -m "feat(models): add ImageEditModel wrapper for diffusion img2img"
```

---

## Task 10: Language Filter

**Files:**
- Create: `python/omniff/filters/language.py`

- [ ] **Step 1: Implement language detection filter**

```python
# python/omniff/filters/language.py
from __future__ import annotations

import re


CYRILLIC_KAZAKH_CHARS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
CYRILLIC_RANGE = re.compile(r"[Ѐ-ӿ]")
LATIN_RANGE = re.compile(r"[a-zA-Z]")


def detect_language(text: str) -> str:
    if not text.strip():
        return "unknown"

    has_kazakh = any(c in CYRILLIC_KAZAKH_CHARS for c in text)
    cyrillic_count = len(CYRILLIC_RANGE.findall(text))
    latin_count = len(LATIN_RANGE.findall(text))
    total = cyrillic_count + latin_count

    if total == 0:
        return "unknown"

    if has_kazakh:
        return "kk"
    if cyrillic_count > latin_count:
        return "ru"
    return "en"
```

- [ ] **Step 2: Update filters __init__.py**

```python
# python/omniff/filters/__init__.py
from omniff.filters.language import detect_language

__all__ = ["detect_language"]
```

- [ ] **Step 3: Commit**

```bash
git add python/omniff/filters/
git commit -m "feat(filters): add language detection filter"
```

---

## Task 11: Runtime Engine — Wire Everything Together

**Files:**
- Rewrite: `python/omniff/runtime/engine.py`
- Create: `python/omniff/runtime/result.py`
- Modify: `python/omniff/runtime/__init__.py`
- Modify: `python/omniff/__init__.py`

- [ ] **Step 1: Create RunResult**

```python
# python/omniff/runtime/result.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    output_text: str | None = None
    output_path: str | None = None
    route: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        if self.output_text is not None:
            Path(path).write_text(self.output_text, encoding="utf-8")
        elif self.output_path is not None:
            import shutil
            shutil.copy2(self.output_path, path)
        else:
            raise ValueError("No output to save")

    def __repr__(self) -> str:
        if self.output_text:
            preview = self.output_text[:100] + ("..." if len(self.output_text) > 100 else "")
            return f"RunResult(text={preview!r}, route={self.route})"
        return f"RunResult(path={self.output_path!r}, route={self.route})"
```

- [ ] **Step 2: Rewrite engine.py**

```python
# python/omniff/runtime/engine.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from omniff.runtime.config import OmniFFConfig
from omniff.runtime.result import RunResult
from omniff.router.keyword_router import KeywordRouter
from omniff.models.registry import ModelRegistry
from omniff.filters.language import detect_language


def _detect_input_modality(input_path: str) -> str:
    if not Path(input_path).exists():
        return "text"
    suffix = Path(input_path).suffix.lower()
    if suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"):
        return "image"
    if suffix in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
        return "audio"
    if suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        return "video"
    if suffix in (".pdf", ".docx", ".doc", ".txt"):
        return "document"
    return "text"


class OmniFFRuntime:
    def __init__(self, config: OmniFFConfig) -> None:
        self.config = config
        self.router = KeywordRouter()
        self.models = ModelRegistry()
        self._initialized = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> OmniFFRuntime:
        config = OmniFFConfig.load(Path(path))
        return cls(config)

    def _ensure_model(self, name: str, model_cls: type, **kwargs: Any) -> Any:
        if not self.models.has(name):
            model = model_cls(**kwargs)
            self.models.register(name, model)
        m = self.models.get(name)
        if not m.is_loaded:
            m.load()
        return m

    def run(
        self,
        input: str,
        prompt: str | None = None,
        output_modality: str | None = None,
        thinking: str = "normal",
        controls: dict[str, Any] | None = None,
        output: str | None = None,
    ) -> RunResult:
        controls = controls or {}
        input_modality = _detect_input_modality(input)

        if input_modality == "text" and not Path(input).exists():
            prompt = prompt or input
            input_modality = "text"

        output_modality = output_modality or "text"

        route = self.router.route(prompt or "", input_modality, output_modality)

        if route.route_class in ("TEXT_SIMPLE", "TEXT_NORMAL", "TEXT_COMPLEX"):
            return self._run_text_to_text(prompt or input, route.route_class, controls)

        if route.route_class == "IMAGE_CAPTION":
            return self._run_image_to_text(input, prompt, controls)

        if route.route_class in ("AUDIO_TRANSCRIBE_ONLY", "AUDIO_QA"):
            return self._run_audio_to_text(input, prompt, controls)

        if route.route_class == "IMAGE_EDIT":
            return self._run_image_to_image(input, prompt or "", controls, output)

        return RunResult(output_text=f"Unsupported route: {route.route_class}", route=route.route_class)

    def _run_text_to_text(self, prompt: str, route: str, controls: dict) -> RunResult:
        from omniff.models.llm import LLMModel

        model_id = controls.get("model_id", "Qwen/Qwen3-4B")
        llm = self._ensure_model("llm", LLMModel, model_id=model_id, device="auto")
        result = llm.infer({"prompt": prompt})
        return RunResult(output_text=result["text"], route=route)

    def _run_image_to_text(self, image_path: str, prompt: str | None, controls: dict) -> RunResult:
        from omniff.models.vlm import VLMModel

        model_id = controls.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        vlm = self._ensure_model("vlm", VLMModel, model_id=model_id, device="auto")
        result = vlm.infer({"image_path": image_path, "prompt": prompt or "Describe this image."})
        return RunResult(output_text=result["text"], route="IMAGE_CAPTION")

    def _run_audio_to_text(self, audio_path: str, prompt: str | None, controls: dict) -> RunResult:
        from omniff.models.asr import ASRModel

        model_id = controls.get("model_id", "openai/whisper-large-v3")
        asr = self._ensure_model("asr", ASRModel, model_id=model_id, device="auto")
        result = asr.infer({"audio_path": audio_path, "language": controls.get("language")})
        text = result["text"]

        if prompt:
            from omniff.models.llm import LLMModel
            llm_id = controls.get("llm_model_id", "Qwen/Qwen3-4B")
            llm = self._ensure_model("llm", LLMModel, model_id=llm_id, device="auto")
            qa_result = llm.infer({"prompt": f"Transcript:\n{text}\n\nQuestion: {prompt}"})
            return RunResult(output_text=qa_result["text"], route="AUDIO_QA",
                             metadata={"transcript": text})

        return RunResult(output_text=text, route="AUDIO_TRANSCRIBE_ONLY")

    def _run_image_to_image(self, image_path: str, prompt: str, controls: dict, output: str | None) -> RunResult:
        from omniff.models.image_edit import ImageEditModel

        model_id = controls.get("model_id", "stabilityai/sdxl-turbo")
        editor = self._ensure_model("image_edit", ImageEditModel, model_id=model_id, device="auto")
        output_path = output or "output.png"
        result = editor.infer({
            "image_path": image_path,
            "prompt": prompt,
            "negative_prompt": controls.get("negative_prompt", ""),
            "strength": controls.get("strength", 0.5),
            "seed": controls.get("seed"),
            "output_path": output_path,
        })
        return RunResult(output_path=result["image_path"], route="IMAGE_EDIT")
```

- [ ] **Step 3: Update __init__ files**

```python
# python/omniff/runtime/__init__.py
from omniff.runtime.engine import OmniFFRuntime
from omniff.runtime.result import RunResult

__all__ = ["OmniFFRuntime", "RunResult"]
```

```python
# python/omniff/__init__.py
"""Saken OmniFF — FFmpeg-like multimodal AI runtime."""

__version__ = "0.1.0"

from omniff.runtime.engine import OmniFFRuntime
from omniff.runtime.result import RunResult

__all__ = ["OmniFFRuntime", "RunResult"]
```

- [ ] **Step 4: Commit**

```bash
git add python/omniff/runtime/ python/omniff/__init__.py
git commit -m "feat(runtime): wire OmniFFRuntime with router, models, and pipeline dispatch"
```

---

## Task 12: Python CLI

**Files:**
- Create: `python/omniff/cli.py`
- Modify: `python/pyproject.toml` (add entry point)

- [ ] **Step 1: Implement CLI**

```python
# python/omniff/cli.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="omniff",
        description="OmniFF — FFmpeg-like multimodal AI runtime",
    )
    parser.add_argument("-i", "--input", required=True, help="Input file or text")
    parser.add_argument("-p", "--prompt", help="Prompt / instruction")
    parser.add_argument("-f", "--of", dest="output_format", help="Output format: text, image, video, audio")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--thinking", default="normal", help="Thinking level: off, fast, normal, deep")
    parser.add_argument("--strength", type=float, help="Style/edit strength 0.0-1.0")
    parser.add_argument("--lang", help="Language hint")
    parser.add_argument("--model", default="auto", help="Model override")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--config", default="omniff.yaml", help="Config file path")
    parser.add_argument("--negative-prompt", help="Negative prompt for generation")

    args = parser.parse_args()

    from omniff.runtime.config import OmniFFConfig
    from omniff.runtime.engine import OmniFFRuntime

    config_path = Path(args.config)
    if config_path.exists():
        runtime = OmniFFRuntime.from_yaml(config_path)
    else:
        config = OmniFFConfig(name="omniff", version="0.1", router=type("R", (), {"router_type": "keyword", "path": ""})())
        runtime = OmniFFRuntime(config)

    controls = {}
    if args.model != "auto":
        controls["model_id"] = args.model
    if args.strength is not None:
        controls["strength"] = args.strength
    if args.lang:
        controls["language"] = args.lang
    if args.seed is not None:
        controls["seed"] = args.seed
    if args.negative_prompt:
        controls["negative_prompt"] = args.negative_prompt

    result = runtime.run(
        input=args.input,
        prompt=args.prompt,
        output_modality=args.output_format,
        thinking=args.thinking,
        controls=controls,
        output=args.output,
    )

    if result.output_text:
        print(result.output_text)
    if result.output_path:
        print(f"Output saved: {result.output_path}")
    if args.output and result.output_text:
        result.save(args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add entry point to pyproject.toml**

Add to `[project.scripts]`:
```toml
[project.scripts]
omniff = "omniff.cli:main"
```

- [ ] **Step 3: Commit**

```bash
git add python/omniff/cli.py python/pyproject.toml
git commit -m "feat(cli): add Python CLI entry point with FFmpeg-like flags"
```

---

## Task 13: Deploy and Test on KazNU Server

**Files:**
- Create: `deploy.sh`

- [ ] **Step 1: Create deploy script**

```bash
#!/usr/bin/env bash
# deploy.sh — rsync project to KazNU and install
set -euo pipefail

SERVER="root@164.138.46.36"
PORT=15126
REMOTE_DIR="/root/omniff"

echo "Syncing to KazNU..."
rsync -avz --exclude='target/' --exclude='.git/' --exclude='__pycache__/' \
  -e "ssh -p $PORT" \
  . "$SERVER:$REMOTE_DIR/"

echo "Installing on server..."
ssh -p "$PORT" "$SERVER" "cd $REMOTE_DIR/python && pip install -e '.[all]' && pip install diffusers qwen-vl-utils 2>&1 | tail -5"

echo "Running unit tests on server..."
ssh -p "$PORT" "$SERVER" "cd $REMOTE_DIR && python -m pytest tests/python/unit/ -v 2>&1"

echo "Done."
```

- [ ] **Step 2: Run deploy**

```bash
chmod +x deploy.sh && ./deploy.sh
```

- [ ] **Step 3: Run integration tests on server**

```bash
ssh -p 15126 root@164.138.46.36 "cd /root/omniff && python -m pytest tests/python/integration/ -v --timeout=300 2>&1"
```

- [ ] **Step 4: Test CLI on server**

```bash
ssh -p 15126 root@164.138.46.36 "cd /root/omniff && python -m omniff.cli -i 'What is 2+2?' -p 'Answer briefly' 2>&1"
```

- [ ] **Step 5: Commit deploy script**

```bash
git add deploy.sh
git commit -m "ops: add deploy script for KazNU server"
```

---

## Phase 2 Outline (v0.2) — after MVP is solid

- text→image: add text-to-image pipeline using SDXL/Flux
- video→text: ffmpeg demux + keyframe extraction + VLM + ASR + LLM summarization
- image→video: image animation via SVD or similar
- document→text: PDF parsing with pymupdf + OCR + LLM QA

## Phase 3 Outline (v0.3)

- video→video: shot detection + control maps + video diffusion + temporal consistency
- audio→audio: ASR → LLM → TTS pipeline
- document→document: structure planner + content gen + format rendering
- scheduler: model hot/warm/cold loading policy
- multi-pass validation

## Phase 4 Outline (v1.0)

- universal graph planner via Thinking+
- full prompt-control side data
- HTTP API server
- production scheduler with GPU affinity
- plugin model interface
- Rust CLI wiring to Python runtime via PyO3

---

Plan saved. 13 tasks for MVP, 3 future phase outlines.

**Plan complete and saved to `docs/superpowers/plans/2026-05-23-omniff-mvp.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
