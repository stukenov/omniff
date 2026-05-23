# OmniFF — Product Roadmap

## Shipped

| Version | Scope | Tests |
|---------|-------|-------|
| v0.1 | text→text, image→text, audio→text, image→image, CLI, router, graph executor | 53 |
| v0.2 | text→image, video→text, document→text | 65 |
| v0.3 | ModelScheduler (hot/warm/cold/LRU), ValidationPipeline, thinking control | 77 |
| v1.0 | GraphPlanner, Thinking+, HTTP API (FastAPI), plugin interface, Gradio demo | 100 |

**Now:** 7 pipelines, 36 source files, 100 tests, GitHub + HF ready.

---

## v1.1 — Developer Experience

**Goal:** Anyone can install, run, and trust the output in 5 minutes.

### Distribution
- [ ] **PyPI** — `pip install saken-omniff` with `[cpu]`, `[gpu]`, `[all]` extras
- [ ] **Docker** — `ghcr.io/stukenov/omniff:latest` with CUDA 12 + all models pre-cached
- [ ] **GitHub Actions CI** — ruff lint, mypy type check, unit tests, cargo check on every push
- [ ] **Pre-commit hooks** — ruff + mypy for contributors

### DX improvements
- [ ] **Pydantic config** — replace raw dataclasses, validate config at load time, clear error messages
- [ ] **Progress bars** — model download + loading progress (tqdm/rich)
- [ ] **`omniff doctor`** — CLI command: check GPU, VRAM, installed models, dependency versions
- [ ] **`omniff models list/pull/remove`** — model management CLI (wraps huggingface_hub)
- [ ] **Better error messages** — "Model X requires N GB VRAM, you have M" instead of CUDA OOM

### Tests
- [ ] Config validation tests (bad YAML, missing fields, wrong types)
- [ ] CLI error path tests
- [ ] Docker smoke test in CI

---

## v1.2 — Reliability & Observability

**Goal:** Production-safe. Know what's happening, recover from failures.

### Reliability
- [ ] **Retry with backoff** — auto-retry on model OOM (unload others, retry)
- [ ] **Timeout control** — per-pipeline max execution time
- [ ] **Graceful degradation** — if VLM unavailable, fall back to text description of image metadata
- [ ] **Memory guard** — refuse to load model if VRAM insufficient, suggest alternatives
- [ ] **Concurrent requests** — async request handling with model mutex (no two requests use same model simultaneously)

### Observability
- [ ] **Structured logging** — structlog with JSON output, log level per component
- [ ] **Request tracing** — correlation ID per request through entire pipeline
- [ ] **Timing breakdown** — per-node execution time in result metadata (demux: 1ms, model: 340ms, validate: 2ms)
- [ ] **Metrics endpoint** — `/metrics` Prometheus: request count, latency p50/p95/p99, GPU util, model load count, cache hit/miss
- [ ] **Model status dashboard data** — `/status` JSON: loaded models, VRAM usage, request queue depth

### Tests
- [ ] Retry/recovery integration tests (mock OOM, verify retry succeeds)
- [ ] Concurrent request test (10 parallel requests, no crashes)
- [ ] Metrics endpoint unit tests

---

## v1.3 — Complete Modality Matrix

**Goal:** Cover every practical input→output combination.

### New pipelines

| Pipeline | Model | Approach |
|----------|-------|----------|
| Text → Speech | Piper / Bark / XTTS-v2 | Direct TTS inference |
| Audio → Audio | Whisper + LLM + TTS | Transcribe → transform → synthesize |
| Video → Video | Frame extract + SDXL + reencode | Per-frame edit with temporal smoothing |
| Document → Document | Extract + LLM + reportlab | Read → summarize/transform → generate PDF |
| Code → Code | Qwen3-4B (code mode) | Code completion, refactor, explain |
| Multi-hop chains | Arbitrary graph | Audio → text → summarize → translate → TTS |

### Implementation
- [ ] **TTS model wrapper** — `TTSModel` with Piper (CPU-friendly, fast)
- [ ] **Voice pipeline** — audio→text→LLM→TTS chain as single `omniff -i input.wav -f audio -o output.wav`
- [ ] **Video editor** — frame extraction → batch SDXL → ffmpeg reencode with matching FPS/codec
- [ ] **PDF generator** — reportlab-based output for document→document pipeline
- [ ] **Code pipeline** — detect code input, route to code-tuned model, preserve formatting
- [ ] **Chain execution** — multi-step pipelines defined in YAML graph templates

### Router upgrade
- [ ] **Encoder-based router** — replace keyword router with small classifier (MiniLM) for better route accuracy
- [ ] **Auto output modality** — infer output format from prompt ("read this aloud" → audio)
- [ ] **Multi-output** — single request producing text + audio (e.g., transcribe + summarize)

### Tests
- [ ] TTS integration test (text in → wav out, valid audio)
- [ ] Voice chain end-to-end test
- [ ] Router accuracy benchmark (100 labeled prompts, measure precision)

---

## v1.4 — Streaming & Real-time

**Goal:** Token-by-token LLM output. Real-time audio processing.

- [ ] **SSE streaming** — `/run/stream` endpoint, Server-Sent Events for LLM tokens
- [ ] **WebSocket** — `/ws` for bidirectional real-time (audio in → text out live)
- [ ] **CLI streaming** — `omniff -i "..." --stream` prints tokens as generated
- [ ] **Chunked audio** — process audio in chunks for real-time transcription
- [ ] **Python SDK streaming** — `for chunk in runtime.run_stream(...):`

### Tests
- [ ] SSE streaming integration test
- [ ] WebSocket echo test
- [ ] Chunked audio accuracy vs full-file accuracy comparison

---

## v2.0 — Scaling & Performance

**Goal:** Handle production load. Multi-GPU, batching, queues.

### Multi-GPU
- [ ] **Device map** — assign models to specific GPUs based on VRAM profile
- [ ] **Auto placement** — scheduler picks GPU with most free VRAM
- [ ] **Model parallelism** — large models split across 2+ GPUs
- [ ] **Pipeline parallelism** — different pipeline stages on different GPUs

### Performance
- [ ] **Request queue** — in-memory async queue with priority levels
- [ ] **Batch inference** — group N text requests, batch tokenize, single forward pass
- [ ] **KV-cache reuse** — for repeated system prompts, cache KV across requests
- [ ] **Model quantization** — GPTQ/AWQ/GGUF support via auto-detect
- [ ] **Compilation** — torch.compile() for hot models, measure speedup

### Deployment
- [ ] **Kubernetes manifests** — Deployment + Service + HPA + GPU resource limits
- [ ] **Helm chart** — `helm install omniff stukenov/omniff`
- [ ] **Health probes** — liveness (process alive), readiness (model loaded), startup (first model ready)
- [ ] **Horizontal scaling** — multiple replicas with sticky sessions for model affinity

### Tests
- [ ] Multi-GPU placement test (mock 2 GPUs, verify correct assignment)
- [ ] Batch inference correctness (batched output == individual outputs)
- [ ] Load test (locust: 50 concurrent users, measure p99)

---

## v2.1 — Ecosystem & Integrations

**Goal:** OmniFF as a building block in larger systems.

### SDK & Clients
- [ ] **TypeScript SDK** — `@saken/omniff` npm package, typed client for HTTP API
- [ ] **Python async client** — `omniff.AsyncClient` with httpx for remote OmniFF servers
- [ ] **gRPC API** — alternative to REST for low-latency internal services

### Framework integrations
- [ ] **LangChain tool** — `OmniFFTool` for agents: any modality as a tool call
- [ ] **LlamaIndex reader** — `OmniFFReader` for document/image/audio ingestion
- [ ] **Jupyter magic** — `%%omniff` cell magic, inline image/audio rendering
- [ ] **Gradio components** — custom Gradio blocks wrapping OmniFF pipelines

### Developer tools
- [ ] **Graph visualizer** — `omniff graph show template.yaml` renders DAG as ASCII/SVG
- [ ] **Plugin scaffold** — `omniff plugin init my-model` generates plugin boilerplate
- [ ] **Documentation site** — MkDocs Material with tutorials, API reference, graph template cookbook
- [ ] **OpenAPI spec** — auto-generated from FastAPI, published at `/docs`

### Tests
- [ ] TypeScript SDK e2e test (start server, call from Node)
- [ ] LangChain integration test (agent uses OmniFF tool)
- [ ] Graph visualizer snapshot tests

---

## v3.0 — Research & Benchmarks

**Goal:** Measure everything. Publish results.

- [ ] **Benchmark suite** — standardized eval per pipeline: BLEU, WER, FID, CLIPScore, latency
- [ ] **Comparison matrix** — OmniFF vs raw transformers vs vLLM vs LangChain vs Triton
- [ ] **Latency profiler** — `omniff bench -i ... --profile` → flamegraph-style breakdown
- [ ] **Model recommendation** — given VRAM budget, recommend optimal model set
- [ ] **ArXiv paper** — "OmniFF: A Universal Multimodal Runtime Inspired by FFmpeg"

---

## Product Coverage Map

What percentage of the product surface each version covers:

```
                Install  Pipelines  Reliability  Performance  Ecosystem  Docs
v1.0 (now)     ██░░░░   ███████░░  ██░░░░░░░░   █░░░░░░░░░   █░░░░░░   █░░░
v1.1 (DX)      ████░░   ███████░░  ███░░░░░░░   ██░░░░░░░░   █░░░░░░   ███░
v1.2 (Rely)    ████░░   ███████░░  ████████░░   ██░░░░░░░░   █░░░░░░   ███░
v1.3 (Modal)   ████░░   ██████████ ████████░░   ██░░░░░░░░   ██░░░░░   ███░
v1.4 (Stream)  █████░   ██████████ █████████░   ███░░░░░░░   ██░░░░░   ████
v2.0 (Scale)   ██████   ██████████ ██████████   ████████░░   ███░░░░   ████
v2.1 (Eco)     ██████   ██████████ ██████████   ████████░░   ████████   █████
v3.0 (Bench)   ██████   ██████████ ██████████   ██████████   ████████   ██████
```

## Execution Order

```
v1.1 ──→ v1.2 ──→ v1.3 ──→ v1.4 ──→ v2.0 ──→ v2.1 ──→ v3.0
 DX      Rely     Modal   Stream   Scale     Eco      Bench
1-2w     2w       3w      1-2w     3-4w      3-4w     2-3w
```

Total estimated: ~16-20 weeks to v3.0.
