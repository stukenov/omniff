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

# OmniFF — FFmpeg for AI

**Universal multimodal runtime** for inference, generation, and transformation.

OmniFF is not a model — it's a runtime that orchestrates models like FFmpeg orchestrates codecs.

## Supported Pipelines

| Pipeline | Models Used |
|----------|-------------|
| Text → Text | Qwen3-4B |
| Image → Text | Qwen2.5-VL-3B |
| Audio → Text | Whisper-large-v3 |
| Image → Image | SDXL-turbo |
| Text → Image | SDXL-turbo |
| Video → Text | Qwen2.5-VL-3B |
| Document → Text | Extraction + Qwen3-4B |

## Quick Start

```bash
pip install saken-omniff[all]

# CLI
omniff -i "What is AI?" --thinking fast
omniff -i photo.jpg -p "Describe this"
omniff -i audio.wav --lang en

# Python
from omniff.runtime.engine import OmniFFRuntime
runtime = OmniFFRuntime.from_yaml("omniff.yaml")
result = runtime.run(input="Hello world", thinking="normal")
print(result.output_text)
```

## Architecture

```
Input → Demuxer → Router → GraphPlanner → [Model → Filter → Validate] → Muxer → Output
```

Key components:
- **KeywordRouter**: modality-based routing
- **GraphPlanner**: DAG execution planning
- **ModelScheduler**: hot/warm/cold model loading with LRU eviction
- **Thinking+**: off/fast/normal/deep/research control
- **ValidationPipeline**: multi-pass output validation

## Links

- **GitHub**: [stukenov/omniff](https://github.com/stukenov/omniff)
- **Architecture**: [Whitepaper](https://github.com/stukenov/omniff/blob/main/ARCHITECTURE.md)

## License

Apache 2.0

## Author

Saken Tukenov — [@stukenov](https://github.com/stukenov)
