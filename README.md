---
license: apache-2.0
tags:
  - multimodal
  - runtime
  - inference
  - ffmpeg
  - text-generation
  - image-to-text
  - text-to-image
  - automatic-speech-recognition
  - video-understanding
  - document-understanding
language:
  - en
  - ru
  - kk
  - zh
library_name: omniff
pipeline_tag: text-generation
---

<p align="center">
  <img src="https://img.shields.io/badge/OmniFF-FFmpeg_for_AI-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJMMyA3djEwbDkgNSA5LTVWN2wtOS01eiIgZmlsbD0id2hpdGUiLz48L3N2Zz4="/>
</p>

<h1 align="center">Saken OmniFF</h1>

<p align="center">
  <b>FFmpeg for AI</b> — universal multimodal runtime for inference, generation, and transformation
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#pipelines">Pipelines</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#api">API</a> •
  <a href="ARCHITECTURE.md">Whitepaper</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.10-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/rust-workspace-DEA584?logo=rust&logoColor=white"/>
  <img src="https://img.shields.io/badge/license-Apache_2.0-green"/>
  <img src="https://img.shields.io/badge/tests-100_passed-brightgreen"/>
</p>

---

## What is OmniFF?

OmniFF applies the FFmpeg philosophy to AI: **any input modality → any output modality**, through a managed graph of models, filters, and validators.

```
omniff -i "What is the capital of Kazakhstan?" -p "Answer briefly" --thinking off
# → Astana

omniff -i photo.jpg -p "Describe this image"
# → A sunset over mountains with golden light...

omniff -i meeting.wav
# → [Full transcription of the audio]

omniff -i sketch.png -p "Make it photorealistic" -o result.png
# → result.png (SDXL-turbo image-to-image)

omniff -i "A cyberpunk city at night" -f image -o city.png
# → city.png (text-to-image generation)

omniff -i lecture.mp4 -p "Summarize this video"
# → The lecture covers three main topics...

omniff -i report.pdf -p "Extract key findings"
# → Key findings: 1) Revenue grew 23%...
```

## Quickstart

```bash
# Install
pip install -e "python/.[all]"

# Text → Text
omniff -i "Explain quantum computing" --thinking normal

# Image → Text
omniff -i photo.jpg -p "What's in this image?"

# Audio → Text
omniff -i recording.wav --lang kk

# Text → Image
omniff -i "A red panda eating bamboo" -f image -o panda.png

# Start HTTP API
python -m omniff.api
```

## Pipelines

| Pipeline | Input | Output | Model | Status |
|----------|-------|--------|-------|--------|
| Text → Text | text / prompt | text | Qwen3.6-35B-A3B (MoE) | ✅ |
| Image → Text | image + prompt | text | Qwen2.5-VL-3B | ✅ |
| Audio → Text | audio file | text | Whisper-large-v3-turbo | ✅ |
| Image → Image | image + prompt | image | Z-Image-Turbo / SDXL-turbo | ✅ |
| Text → Image | prompt | image | Z-Image-Turbo / SDXL-turbo | ✅ |
| Video → Text | video + prompt | text | Qwen2.5-VL-3B | ✅ |
| Document → Text | PDF/DOCX/TXT | text | Extraction + Qwen3 | ✅ |
| Audio → Translation | audio + target lang | text | Whisper-turbo + HY-MT2 | ✅ |
| Audio → Dubbed Audio | audio + target lang | audio | Whisper-turbo + HY-MT2 + TTS | ✅ |
| Video → Dubbed Video | video + target lang | video | ffmpeg + full pipeline | ✅ |
| Text → Translation | text + target lang | text | Tencent HY-MT2 / NLLB-200 | ✅ |
| Voice Cloning | audio + text | audio | OmniVoice / CosyVoice / Bark | ✅ |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     OmniFF Runtime                       │
│                                                          │
│  Input ─→ Demuxer ─→ Router ─→ GraphPlanner             │
│                                    │                     │
│                          ┌─────────┴──────────┐          │
│                          │    OmniGraph DAG    │          │
│                          │                     │          │
│                          │  ┌─────┐  ┌──────┐ │          │
│                          │  │Model│─→│Filter│ │          │
│                          │  └─────┘  └──┬───┘ │          │
│                          │              │     │          │
│                          │         ┌────┴───┐ │          │
│                          │         │Validate│ │          │
│                          │         └────┬───┘ │          │
│                          └──────────────┘     │          │
│                                    │          │          │
│                               Muxer ─→ Output            │
│                                                          │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌───────────┐  │
│  │Scheduler │ │ Thinking+ │ │ Plugins  │ │ HTTP API  │  │
│  │hot/warm/ │ │off/fast/  │ │ custom   │ │ FastAPI   │  │
│  │cold/LRU  │ │normal/deep│ │ models   │ │ /run      │  │
│  └──────────┘ └───────────┘ └──────────┘ └───────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| **KeywordRouter** | Routes input to the right pipeline based on modality + keywords |
| **GraphPlanner** | Builds DAG execution plans (demux → model → validate → mux) |
| **GraphExecutor** | Walks DAG in topological order, passes data between nodes |
| **ModelScheduler** | Hot/warm/cold loading with TTL eviction and LRU |
| **ValidationPipeline** | Multi-pass validation chain with required/optional passes |
| **Thinking+** | Prompt control: off → fast → normal → deep → research |
| **PluginRegistry** | Register custom model implementations |

## API

### Python SDK

```python
from omniff.runtime.engine import OmniFFRuntime

runtime = OmniFFRuntime.from_yaml("omniff.yaml")

# Text → Text
result = runtime.run(input="Explain gravity", thinking="normal")
print(result.output_text)

# Image → Text  
result = runtime.run(input="photo.jpg", prompt="Describe this")
print(result.output_text)

# Text → Image
result = runtime.run(input="A sunset", output_modality="image", output="sunset.png")
print(result.output_path)
```

### HTTP API

```bash
# Start server
python -m omniff.api

# Text processing
curl -X POST http://localhost:8000/run \
  -F "input_text=What is AI?" \
  -F "thinking=fast"

# File processing
curl -X POST http://localhost:8000/run/file \
  -F "file=@photo.jpg" \
  -F "prompt=Describe this image"

# Health check
curl http://localhost:8000/health
```

### CLI (FFmpeg-style)

```bash
omniff -i <input> [-p <prompt>] [-f <format>] [-o <output>] [--thinking <level>]
       [--strength <0-1>] [--lang <code>] [--model <id>] [--seed <n>]
```

## Project Structure

```
omniff/
├── python/omniff/           # Python SDK (saken-omniff)
│   ├── models/              # Model wrappers (LLM, VLM, ASR, ImageEdit, ...)
│   ├── runtime/             # Engine, config, result
│   ├── router/              # KeywordRouter
│   ├── graph/               # OmniGraph, executor, planner, loader
│   ├── scheduler/           # ModelScheduler (hot/warm/cold)
│   ├── validators/          # Text/Image validators, pipeline
│   ├── filters/             # Language detection
│   ├── nodes/               # Node registry
│   ├── api.py               # FastAPI HTTP server
│   ├── cli.py               # CLI entry point
│   ├── thinking.py          # Thinking+ controller
│   └── plugins.py           # Plugin model interface
├── crates/                  # Rust workspace
│   ├── omniff-core/         # Core types (OmniPacket, OmniFrame, OmniNode)
│   ├── omniff-graph/        # Graph types and planner trait
│   ├── omniff-runtime/      # Runtime traits (Router, Executor, Scheduler)
│   └── omniff-cli/          # Rust CLI binary
├── tests/python/            # 85 unit + 15 integration tests
├── graph_templates/         # YAML pipeline templates
├── omniff.yaml              # Runtime configuration
└── ARCHITECTURE.md          # Full architectural whitepaper
```

## Configuration

```yaml
# omniff.yaml
runtime:
  name: omniff
  version: "1.0"

router:
  type: keyword

experts:
  text:
    model_id: Qwen/Qwen3.6-35B-A3B  # MoE, 3B active params
    loading_policy: warm
    ttl: 300
  vision:
    model_id: Qwen/Qwen2.5-VL-3B-Instruct
    loading_policy: warm
  asr:
    model_id: openai/whisper-large-v3-turbo
    loading_policy: cold
  image_gen:
    model_id: Tongyi-MAI/Z-Image-Turbo
    loading_policy: cold
  translator:
    model_id: tencent/Hy-MT2-30B-A3B
    loading_policy: cold
  voice_cloner:
    model_id: k2-fsa/OmniVoice
    loading_policy: cold
```

## Testing

```bash
# All tests
PYTHONPATH=python python -m pytest tests/python/ -v

# Unit tests only
PYTHONPATH=python python -m pytest tests/python/unit/ -v

# Integration tests (requires GPU)
PYTHONPATH=python python -m pytest tests/python/integration/ -v

# Rust
cargo check --workspace
```

Tested on **KazNU server** (2× NVIDIA A10 22GB).

## Naming Convention

| Surface | Identifier |
|---------|-----------|
| CLI | `omniff` |
| Python package | `saken-omniff` |
| Rust crates | `omniff-core`, `omniff-graph`, `omniff-runtime`, `omniff-cli` |
| GitHub | [`stukenov/omniff`](https://github.com/stukenov/omniff) |
| Hugging Face | [`stukenov/omniff-runtime`](https://huggingface.co/stukenov/omniff-runtime) |

## License

Apache 2.0

---

<p align="center">
  Built by <a href="https://github.com/stukenov">Saken Tukenov</a>
</p>
