# Saken OmniFF — Architecture Whitepaper

**Full name:** Saken OmniFF
**Runtime name:** OmniFF Runtime
**Kazakh-first variant:** OmniFF-KZ
**Core formula:** FFmpeg for AI inference, generation, and multimodal transformation

**Document status:** architectural doctrine
**Purpose:** single canonical document for design, implementation, publication, and explanation

---

## Naming Policy

| Surface | Identifier |
|---------|-----------|
| CLI binary | `omniff` |
| Python package | `saken-omniff` |
| Rust crates | `omniff-core`, `omniff-graph`, `omniff-runtime`, `omniff-cli` |
| NPM package | `@saken/omniff` |
| GitHub | `stukenov/omniff` |
| Hugging Face | `stukenov/omniff-runtime` |

All public APIs, imports, configs, and docs must use these canonical names. No aliases.

---

## 1. Summary

OmniFF Runtime is not a neural network model. It is a universal multimodal runtime that accepts any input — text, audio, image, video, documents, structured data — and transforms them into any output modality through a managed graph of models, filters, validators, and planners.

Architecture inspired by FFmpeg:

```
container → demuxer → decoder → filtergraph → encoder → muxer
```

For AI this becomes:

```
input container
→ demuxer
→ modality decoder
→ OmniFrame normalization
→ Thinking+ planner
→ AI filtergraph
→ model experts
→ validators
→ output encoder
→ muxer
```

Key distinction from ordinary LLM systems: the model is not the center of the product. The model is one computational node. The center is the runtime, the graph, and the unified format for processing multimodal streams.

### Full modality matrix

```
text   → text, image, video, audio
image  → text, image, video
video  → text, image, video
audio  → text, audio, video
document → text, document
mixed input → mixed output
```

The system does not pretend to be one monolithic model. It is a unified runtime with many specialized models inside.

---

## 2. Core Doctrine

### 2.1. This is a runtime, not a model

OmniFF Runtime is not one Transformer, not one `.safetensors`, not one decoder-only LLM.

Correct definitions:

```
Omni inference runtime
Multimodal graph engine
AI media processing framework
Routed multimodal model system
FFmpeg-like AI runtime
```

A model is only one type of node inside the runtime.

### 2.2. FFmpeg Principle

FFmpeg became foundational not because it was one codec. It became foundational because it gave a common language for containers, streams, codecs, filters, transformations, and output.

OmniFF Runtime does the same for AI:

```
media streams    → AI streams
frames           → OmniFrames
filters          → AI filters
codecs           → model experts
filtergraph      → OmniGraph
metadata         → prompt/control side data
muxing           → multimodal output assembly
```

### 2.3. Models are codecs

```
LLM              = text codec / reasoning codec
Whisper          = audio perception codec
VLM              = vision perception codec
Image diffusion  = image generation codec
Video diffusion  = video generation codec
TTS              = speech generation codec
Encoder router   = routing codec
OCR              = document perception codec
```

A model does not control the system. A model executes its role in the graph.

---

## 3. Architectural Laws

### Law 1. Runtime over model

The product must not be hostage to one model, one provider, one tokenizer, one inference engine, or one weight format. Models can be swapped. The runtime must remain.

### Law 2. Graph over pipeline

The system must not be a collection of hardcoded pipeline scripts. All transformations must be expressed through a graph.

Wrong:

```
if image then do this
if video then do that
if audio then do another script
```

Right:

```
input → graph planner → DAG → execution → validation → output
```

### Law 3. Prompt is a control layer, not just a string

Prompt must be represented as structured side data:

- user prompt
- system prompt
- task prompt
- modality prompt
- generation prompt
- negative prompt
- control prompt
- validator prompt
- constraints, seed, strength, masks
- reference assets, preservation rules
- thinking budget

### Law 4. Thinking+ is a planner, not a final answer

Thinking+ is a control module that builds an execution plan, selects a graph, assigns models, sets constraints, launches validators, and decides on retry. The user receives a brief execution summary, not the internal chain of reasoning.

### Law 5. Router must be cheap

Router must not be a large LLM. Router must be an encoder-only or other cheap classifier.

```
prompt / normalized semantic state
→ encoder-only classifier
→ selected model / route / graph
```

Router does not generate answers. Router selects routes.

### Law 6. Do not run all models always

Wrong:

```
0.6B → 4B → 14B → 32B always
```

Right:

```
router → minimum sufficient model
```

Only on failure:

```
fallback / escalation / validation retry
```

### Law 7. Omni-directions require separate generative branches

An LLM or VLM alone cannot close image-to-image and video-to-video. Separate branches needed:

- image generation, editing, inpainting, ControlNet-like control
- video generation, video-to-video, temporal consistency
- audio generation, TTS
- document rendering

### Law 8. One external product, many internal experts

Outside: one API, one CLI, one SDK, one HF repo, one Docker.

Inside: modular —

```
router, ASR, VLM, LLM, image generator, video generator,
TTS, OCR, document parser, validators, scheduler
```

### Law 9. Architectural honesty over marketing

Cannot pretend this is one monolithic neural network.

```
A FFmpeg-like multimodal AI runtime with routed model experts
and thinking-controlled graph execution.
```

---

## 4. Core Architecture

### 4.1. Top-level flow

```
User input
  ↓
Input container
  ↓
Demuxer
  ↓
Modality decoder
  ↓
OmniPacket / OmniFrame
  ↓
Normalization
  ↓
Thinking+ Planner
  ↓
Router
  ↓
OmniGraph
  ↓
Model/filter execution
  ↓
Validators
  ↓
Output encoder
  ↓
Muxer
  ↓
Final output
```

### 4.2. Core libraries

By analogy with FFmpeg:

| Library | Responsibility |
|---------|---------------|
| `libomniformat` | input/output containers, demux/mux |
| `libomnimodel` | model loading and execution |
| `libomnifilter` | AI filters and transformations |
| `libomnigraph` | DAG planning and execution |
| `libomnimemory` | tensors, frames, cache, device placement |
| `libomnivalidate` | validators, critics, constraint checks |
| `libomnischedule` | scheduling, batching, GPU/CPU placement |
| `libomniapi` | CLI, SDK, HTTP API |

### 4.3. Runtime entities

#### OmniPacket

Raw input fragment:

```
text chunk, audio bytes, video packet, image bytes,
PDF page, JSON message, subtitle segment, metadata block
```

#### OmniFrame

Normalized processing object:

```
text tokens, audio PCM, image tensor, video frame,
embedding, mask, depth map, pose map, transcript,
OCR layer, scene graph, semantic state, control map
```

#### OmniGraph

DAG describing task execution:

```
nodes = models / filters / tools / validators
edges = data dependencies
side data = prompts / controls / constraints
```

#### OmniNode

One executable node:

```
ASR node, VLM node, LLM node, image generation node,
video generation node, OCR node, validator node,
ffmpeg utility node, scheduler node
```

#### OmniModel

Model wrapper:

```
load, unload, infer, generate, stream,
batch, quantize, cache
```

#### OmniFilter

Data transformation:

```
resize, crop, normalize, extract_depth, extract_edges,
extract_pose, detect_faces, track_objects, split_shots,
summarize, translate, style_transfer
```

#### OmniValidator

Result verification:

```
language check, schema check, visual prompt adherence,
face preservation, temporal consistency, OCR correctness,
toxicity/safety check, factuality check, format validation
```

---

## 5. Routing

### 5.1. Router role

Router does not answer the user. Router selects:

- which graph template to use
- which models to invoke
- which thinking level to enable
- which validator is needed
- which escalation policy applies

### 5.2. Encoder-only router

Preferred architecture:

```
XLM-R / ModernBERT / BGE-style encoder
+ classification head
→ route class
```

Output:

```json
{
  "selected_route": "image_to_image",
  "confidence": 0.87,
  "risk": "low",
  "thinking": "normal"
}
```

### 5.3. Route classes

```
TEXT_SIMPLE
TEXT_NORMAL
TEXT_COMPLEX
AUDIO_TRANSCRIBE_ONLY
AUDIO_QA
IMAGE_CAPTION
IMAGE_EDIT
TEXT_TO_IMAGE
TEXT_TO_VIDEO
IMAGE_TO_VIDEO
VIDEO_SUMMARY
VIDEO_TO_VIDEO
DOCUMENT_OCR_QA
DOCUMENT_TO_DOCUMENT
REJECT_OR_HUMAN_REVIEW
```

### 5.4. Model ladder (text/reasoning)

If using Qwen family as text/reasoning backbone:

```
Qwen3-0.6B     router-assistant / cheap tasks
Qwen3-4B       normal assistant
Qwen3-14B      hard tasks
Qwen3-32B      local high-quality / judge
```

Production minimum:

```
Qwen3-0.6B  (cheap/fast)
Qwen3-4B    (normal)
Qwen3-14B   (complex)
Qwen3-32B   (judge)
```

Router selects the minimum sufficient model. Escalation only on failure.

---

## 6. Thinking+

### 6.1. Purpose

Thinking+ is a control layer for planning and execution control, not just "the model thinks longer."

Thinking+ must:

1. Understand the task
2. Determine input/output modalities
3. Select graph template
4. Choose models
5. Assign validators
6. Set constraints
7. Define retry policy
8. Form execution plan

### 6.2. Thinking levels

```
thinking=off
  fast routing, single pass, minimal checking

thinking=fast
  router + simple graph

thinking=normal
  planner + executor + validator

thinking=deep
  planner + executor + critic + retry

thinking=research
  multiple candidates + judge + detailed validation
```

### 6.3. Execution plan example

```json
{
  "task": "video_to_video",
  "preserve": ["faces", "voice", "camera_structure"],
  "style": "premium minimal corporate",
  "required_nodes": [
    "shot_detection",
    "face_tracking",
    "audio_transcription",
    "style_transfer_video",
    "temporal_validator",
    "audio_mux"
  ],
  "risk": "high",
  "validator": "vlm_video_judge",
  "retry_policy": "up_to_2"
}
```

### 6.4. User-facing output

Internal reasoning chain is never mandatory output. User receives brief route explanation:

```json
{
  "mode": "deep",
  "route": "image_to_image",
  "controls": ["depth", "mask", "reference"],
  "generator": "image_edit_model",
  "validator": "vision_validator"
}
```

---

## 7. Prompt Control

### 7.1. Prompt as side data

Every OmniFrame carries side data:

```json
{
  "prompt": "matte graphite car wrap",
  "negative_prompt": "cartoon, distorted wheels, wrong car shape",
  "seed": 42,
  "strength": 0.35,
  "preserve_identity": true,
  "preserve_layout": true,
  "control_maps": ["depth", "canny", "mask"],
  "thinking_budget": 2048,
  "validator_threshold": 0.82
}
```

### 7.2. Prompt layers

```
system prompt       → global behavior
user prompt         → user intent
task prompt         → task-specific instructions
modality prompt     → per-modality hints
generation prompt   → enriched generation instruction
negative prompt     → what to avoid
control prompt      → structural control
validator prompt    → validation criteria
```

### 7.3. Image prompt control

```json
{
  "user_prompt": "Make the car matte graphite",
  "generation_prompt": "black Hyundai Elantra 2021, matte graphite wrap, premium realistic studio lighting",
  "negative_prompt": "cartoon, damaged car, wrong wheels, deformed body",
  "preserve": {
    "car_model": true,
    "camera_angle": true,
    "body_shape": true,
    "background": false
  },
  "strength": 0.38,
  "controls": ["canny", "depth"]
}
```

### 7.4. Video prompt control

```json
{
  "global_prompt": "cinematic corporate video, clean premium style",
  "shot_prompts": [
    {"shot": 1, "prompt": "slow dolly-in, preserve subject identity"},
    {"shot": 2, "prompt": "soft lighting, premium office mood"}
  ],
  "negative_prompt": "flickering, face distortion, unstable hands, warped text",
  "temporal_consistency": "high",
  "style_strength": 0.45
}
```

---

## 8. Omni-Directions

### 8.1. Text → Text

Purpose: answers, analysis, translation, correction, classification, legal, code, RAG, structured output.

```
text input → language detection → router → LLM expert → validator → text output
```

### 8.2. Audio → Text

Purpose: transcription, speech translation, meeting summary, call center, lectures.

```
audio → VAD/chunking → ASR → transcript cleanup → language correction
→ LLM/summary/QA → text output
```

### 8.3. Audio → Audio

Purpose: speech-to-speech assistant, dubbing, voice translation, call center automation.

```
audio → ASR → LLM → TTS → audio encoder
```

### 8.4. Audio → Video

Purpose: podcast visualization, music video generation, audio-driven animation.

```
audio → ASR/analysis → scene planner → video generation → audio mux → output
```

### 8.5. Image → Text

Purpose: captioning, OCR, visual QA, document analysis, screenshot understanding.

```
image → image decoder → VLM/OCR → normalized text/scene graph → router → LLM → text output
```

### 8.6. Text → Image

Purpose: image generation, visual concepts, design, ads, UI mockups.

```
text prompt → prompt planner → image generation model → image validator → image encoder
```

### 8.7. Image → Image

Purpose: image editing, stylization, inpainting, outpainting, color change, shape preservation, reference-based generation.

```
image + prompt → image analysis → mask/control extraction → edit planner
→ image edit model → vision validator → output image
```

Controls: mask, depth, canny, pose, segmentation, reference image, style strength, identity preservation, layout preservation, negative prompt, seed.

### 8.8. Text → Video

Purpose: clip generation, ads, storyboard-to-video, presentation videos.

```
text prompt → scene planner → shot list → video generation model
→ temporal validator → video encoder
```

### 8.9. Image → Video

Purpose: image animation, motion prompt, avatar video, product animation.

```
image + motion prompt → image analysis → motion planner
→ image-to-video model → temporal validator → video output
```

### 8.10. Video → Text

Purpose: video summary, lecture analysis, surveillance analysis, meeting extraction, content indexing.

```
video → demux audio/video → shot detection → keyframe extraction → ASR
→ VLM analysis → multimodal summary → text output
```

### 8.11. Video → Image

Purpose: keyframe extraction, thumbnail generation, scene capture.

```
video → shot detection → keyframe selection → VLM analysis → best frame selection
→ optional image enhancement → image output
```

### 8.12. Video → Video

Purpose: style transfer, enhancement, cinematic transformation, face/body/background preservation, corporate video transformation, generative editing.

```
video → demux → shot detection → keyframe extraction → audio transcription
→ motion analysis → depth/pose/edge maps → video edit planner
→ video generation/editing model → temporal consistency filter
→ audio restoration/mux → video output
```

Controls: global prompt, per-shot prompt, negative prompt, motion strength, style strength, identity preservation, camera preservation, seed per shot, control maps per frame, mask tracks, temporal consistency.

### 8.13. Document → Text

Purpose: PDF analysis, contract review, law analysis, table extraction, OCR, document QA.

```
document → parser/OCR → layout extraction → chunks → retrieval/reasoning → text output
```

### 8.14. Document → Document

Purpose: contracts, whitepapers, PRDs, technical specs, government letters, Word/PDF/slides generation.

```
document/input text → structure planner → content generator
→ format renderer → validator → output document
```

---

## 9. Image-to-Image Architecture

### 9.1. Why image-to-image is not LLM-only

LLM can understand an instruction but must not be the sole image generator.

```
LLM/VLM  = understand and plan
Image model = generate/edit
Validator = check
```

### 9.2. Image-to-image nodes

```
decode_image
analyze_image_with_vlm
extract_mask
extract_depth
extract_edges
extract_pose
plan_edit_with_thinking
run_image_edit_model
validate_prompt_adherence
validate_preservation
encode_image
```

### 9.3. Example graph

```json
{
  "nodes": [
    {"id": "analyze_image", "model": "vlm"},
    {"id": "extract_depth", "model": "depth"},
    {"id": "extract_mask", "model": "sam"},
    {"id": "plan_edit", "model": "llm_thinking"},
    {"id": "generate_image", "model": "image_edit"},
    {"id": "validate", "model": "vision_validator"}
  ],
  "edges": [
    ["analyze_image", "plan_edit"],
    ["extract_depth", "generate_image"],
    ["extract_mask", "generate_image"],
    ["plan_edit", "generate_image"],
    ["generate_image", "validate"]
  ]
}
```

---

## 10. Video-to-Video Architecture

### 10.1. Why video-to-video is harder than image-to-image

Processing each frame independently causes:

- flickering
- identity loss
- face distortion
- motion destruction
- unstable style
- inter-frame artifacts

Video-to-video requires temporal consistency.

### 10.2. Video-to-video nodes

```
demux_video_audio
decode_video_frames
shot_detection
keyframe_selection
transcribe_audio
analyze_keyframes
extract_depth_sequence
extract_pose_sequence
extract_edges_sequence
track_faces
track_objects
plan_shots_with_thinking
run_video_edit_model
temporal_consistency_filter
restore_audio
encode_video
mux_audio_video
validate_video
```

### 10.3. Example graph

```json
{
  "nodes": [
    {"id": "split_video", "tool": "ffmpeg"},
    {"id": "analyze_keyframes", "model": "vlm"},
    {"id": "transcribe_audio", "model": "asr"},
    {"id": "extract_motion", "tool": "optical_flow"},
    {"id": "make_control_maps", "models": ["depth", "canny", "pose"]},
    {"id": "plan_shots", "model": "llm_thinking"},
    {"id": "generate_video", "model": "video_diffusion"},
    {"id": "restore_audio", "tool": "ffmpeg"},
    {"id": "validate_video", "model": "video_validator"}
  ]
}
```

---

## 11. Runtime Scheduling

### 11.1. Scheduler as critical layer

Without a scheduler the system becomes a slow Python script. Scheduler must manage:

- CPU/GPU placement
- model loading and unloading
- batch processing and streaming
- caching and retry
- memory pressure and prioritization
- long-running jobs and device affinity

### 11.2. Example device distribution

```
CPU:   demux, decode, OCR preprocessing, ffmpeg ops, graph planning
GPU 0: ASR / Whisper
GPU 1: VLM / image analysis
GPU 2: LLM / planner / router
GPU 3: image/video generation
```

### 11.3. Model loading policy

Not all models should be in memory at all times.

```
hot models:   router, small LLM, ASR small
warm models:  VLM, medium LLM, image edit
cold models:  video generation, huge judge, rare experts
```

Scheduler capabilities:

```
preload, lazy load, unload, pin to GPU, move to CPU,
quantized load, batch requests, reuse cache
```

---

## 12. CLI

### 12.1. FFmpeg-like CLI

```bash
omniff -i input.jpg \
  -prompt "make it matte graphite, preserve body and angle" \
  -of image \
  -o result.png
```

```bash
omniff -i input.mp4 \
  -prompt "premium corporate ad style" \
  -thinking deep \
  -preserve faces,voice,structure \
  -strength 0.42 \
  -o output.mp4
```

```bash
omniff -i lesson_audio.mp3 \
  -task summarize \
  -lang kk \
  -model auto \
  -o output.md
```

```bash
omniff -i contract.pdf \
  -task "find risks and write brief summary" \
  -thinking deep \
  -o review.docx
```

### 12.2. Explicit graph CLI

```bash
omniff -i input.mp4 \
  -graph graphs/video_to_video_premium.yaml \
  -prompt "premium Apple-like corporate style" \
  -o output.mp4
```

---

## 13. SDK / API

### 13.1. Python API

```python
from omniff import OmniFFRuntime

runtime = OmniFFRuntime.from_pretrained("stukenov/omniff-runtime")

result = runtime.run(
    input="input.mp4",
    prompt="Video-to-video in premium style, preserve faces",
    output_modality="video",
    thinking="deep",
    controls={
        "preserve_identity": True,
        "preserve_audio": True,
        "style_strength": 0.45,
        "temporal_consistency": "high",
    },
)

result.save("output.mp4")
```

### 13.2. Planning API

```python
graph = runtime.plan(
    input="car.jpg",
    prompt="make it matte graphite",
    output_modality="image",
)

print(graph)
result = runtime.execute(graph)
```

### 13.3. HTTP API

```
POST /v1/omniff/run
```

```json
{
  "input": "file://input.mp4",
  "prompt": "premium style video",
  "output_modality": "video",
  "thinking": "deep",
  "controls": {
    "preserve_faces": true,
    "preserve_voice": true,
    "style_strength": 0.45
  }
}
```

---

## 14. Packaging

### 14.1. Not one safetensors

Production packaging:

```
omniff-runtime/
  omniff.yaml
  graph_templates/
  models/
    router/
    asr/
    vlm/
    llm_small/
    llm_large/
    image_generator/
    video_generator/
    tts/
  processors/
  validators/
  runtime/
  README.md
```

### 14.2. omniff.yaml

```yaml
name: omniff-runtime
version: 0.1

router:
  type: encoder_classifier
  path: models/router

experts:
  text_small:
    type: causal_lm
    path: models/llm_small

  text_large:
    type: causal_lm
    path: models/llm_large

  asr:
    type: speech_to_text
    path: models/asr

  vision:
    type: vision_language
    path: models/vlm

  image_edit:
    type: diffusion_image_edit
    path: models/image_generator

  video_edit:
    type: diffusion_video_edit
    path: models/video_generator

  tts:
    type: text_to_speech
    path: models/tts
```

### 14.3. Hugging Face custom architecture

For research/demo — HF repo with custom code:

```
configuration_omniff.py
modeling_omniff.py
processing_omniff.py
config.json
routing_config.yaml
```

Loading:

```python
model = OmniFFRuntime.from_pretrained(
    "stukenov/omniff-runtime",
    trust_remote_code=True,
)
```

Production must not depend on loading everything as one `AutoModelForCausalLM`.

---

## 15. Cascade Routing

### 15.1. Principle

```
simple    → small model
normal    → medium model
complex   → large model
critical  → judge / human review
```

Saves cost by orders of magnitude on real traffic — large model often never starts.

Quality depends on router accuracy.

### 15.2. Escalation flow

```
Router selects minimum sufficient model
→ model executes
→ validator checks result
→ on failure: escalate to stronger model or retry with adjusted controls
→ on repeated failure: mark as failed or route to human review
```

---

## 16. Safety and Quality

### 16.1. Validator-first philosophy

Every complex graph must have a validator.

**Text validators:**

```
language, format, JSON schema, citation, risk, factuality
```

**Image validators:**

```
prompt adherence, identity preservation, layout preservation,
NSFW/safety, artifact detection, OCR/text correctness
```

**Video validators:**

```
temporal consistency, face preservation, flicker detection,
motion coherence, audio-video sync, prompt adherence
```

### 16.2. Escalation

On validator failure:

```
retry same graph with adjusted controls
→ or escalate to stronger model
→ or ask for clarification
→ or mark as failed
→ or route to human review
```

---

## 17. Logging and Router Training

### 17.1. What to log

```
request_id, user_id/tenant_id, input modalities, output modality,
prompt hash, language, task_type, selected_route, selected_models,
thinking_mode, latency, cost estimate, success/failure,
validator scores, fallbacks, retries, user feedback, output metadata
```

### 17.2. Router training

Primary label: **cheapest sufficient route** — the cheapest model/graph that produced acceptable quality.

Process:

```
1. Collect real and synthetic prompts
2. Run through different route/model variants
3. Score outputs with judge model + partial human eval
4. Assign cheapest-sufficient label
5. Train encoder-only classifier
6. Export to ONNX/Candle
7. Embed in runtime
8. Continuously retrain on logs
```

---

## 18. Technical Stack

### 18.1. Runtime core

Start:

```
Python + PyTorch + Transformers + Diffusers + FFmpeg bindings
```

Production:

```
Rust/Go runtime shell
Python model workers where needed
ONNX/Candle/TensorRT acceleration where justified
```

### 18.2. No dependency on vLLM/SGLang

vLLM and SGLang may be optional backends but never the foundation.

```
OmniGraph owns routing, planning, graph execution, and scheduling.
Model backends are replaceable.
```

### 18.3. Model backend types

```
PyTorch, Transformers, Diffusers, ONNX Runtime,
Candle, GGUF/llama.cpp-style, custom CUDA kernels,
external API adapter
```

---

## 19. MVP Roadmap

### v0.1 — Prove the architecture

```
text → text
image → text
audio → text → text
image → image
```

Components:

```
OmniFF CLI
OmniFrame / OmniPacket
OmniGraph / OmniNode
OmniFFRuntime
encoder-only router
ASR module, VLM module, LLM module
image edit module, validator module
```

### v0.2

```
text → image
video → text
image → video
document → text
```

### v0.3

```
video → video
audio → audio
document → document
multi-pass validation
scheduler
model hot/warm/cold loading
```

### v1.0

```
universal graph planner
Thinking+ controller
prompt-control side data
full modality matrix
validators
production scheduler
CLI + SDK + HTTP API
plugin model interface
```

---

## 20. Product Identity

### Short positioning

```
OmniFF Runtime is a FFmpeg-like multimodal AI processing engine.
```

### Extended positioning

```
A multimodal graph runtime with encoder-only routing,
thinking-controlled planning, and pluggable model experts
for text, speech, vision, image generation, video generation,
documents, and structured outputs.
```

### Kazakh-first variant (OmniFF-KZ)

```
OmniFF-KZ is a Kazakh-first multimodal AI runtime that combines
Qwen expert hierarchy, ASR, vision, image/video generation,
and document intelligence through a unified graph execution engine
with native Kazakh language support.
```

---

## 21. What This Must Not Be

OmniFF Runtime must not be:

```
a LangChain pipeline
a collection of Python scripts
a wrapper over vLLM
a chatbot
a HuggingFace demo
a ComfyUI clone
one big safetensors
a gateway
a multimodal LLM
an agent framework
```

It must be:

```
runtime, format, graph engine, model orchestration layer,
scheduler, prompt-control system, validator system, CLI/API/SDK
```

---

## 22. Canonical Formula

```
input
→ demux
→ decode
→ normalize into OmniFrames
→ plan with Thinking+
→ execute graph of AI filters/models
→ validate
→ encode
→ mux
→ output
```

One repo. One config. One processor. One runtime. One CLI. One API. Many experts. One graph executor.

---

## 23. Conclusion

OmniFF Runtime is an infrastructure system of a new class: a multimodal AI runtime that relates to models the way FFmpeg relates to codecs.

It does not compete with Qwen, Whisper, VLMs, diffusion models, or TTS. It uses them as interchangeable computational nodes.

Its value is not "one model that does everything." Its value is a unified engineering way to build any transformation:

```
text ↔ audio ↔ image ↔ video ↔ document ↔ structured data
```

With control:

```
prompt, negative prompt, thinking, router, models,
validators, constraints, schedulers, quality thresholds
```

**Saken OmniFF Runtime:**
A FFmpeg-like AI runtime for routed, thinking-controlled, multimodal generation and transformation.

Not "one model." A stronger category: **an operating environment for multimodal AI inference and generation.**
