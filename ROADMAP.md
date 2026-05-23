# OmniFF — Roadmap

## Done

| Phase | Version | What |
|-------|---------|------|
| MVP | v0.1 | text→text, image→text, audio→text, image→image, CLI, router, graph executor |
| Expand | v0.2 | text→image, video→text, document→text |
| Infra | v0.3 | ModelScheduler (hot/warm/cold), ValidationPipeline, thinking control |
| Full | v1.0 | GraphPlanner, Thinking+, HTTP API, plugins, Gradio demo |

**Current state:** 7 pipelines, 100 tests, GitHub live, HF card ready.

---

## Phase 5 — Production Hardening (v1.1)

**Goal:** Make it deployable and reliable.

- [ ] **CI/CD pipeline** — GitHub Actions: lint (ruff), type check (mypy), unit tests, Rust cargo check
- [ ] **PyPI publish** — `saken-omniff` on PyPI via `python -m build && twine upload`
- [ ] **Docker image** — GPU-ready container with all dependencies
- [ ] **Streaming output** — SSE/WebSocket for LLM token streaming in API
- [ ] **Error recovery** — retry logic, model reload on OOM, graceful degradation
- [ ] **Logging** — structured logging (structlog), request tracing with correlation IDs
- [ ] **Config validation** — pydantic models instead of raw dataclasses
- [ ] **Rate limiting** — token bucket for API endpoints
- [ ] **Auth** — API key middleware for HTTP server

**Tests:** error recovery tests, streaming tests, config validation tests.

---

## Phase 6 — OmniFF-KZ (v1.2)

**Goal:** Kazakh-first AI runtime. Unique value proposition.

- [ ] **Kazakh ASR** — fine-tuned Whisper on Kazakh speech data (KSC, Common Voice kk)
- [ ] **Kazakh TTS** — text-to-speech pipeline (VITS/Coqui with kk dataset)
- [ ] **Kazakh LLM routing** — detect kk → route to Qwen with Kazakh system prompt
- [ ] **Cyrillic/Latin toggle** — auto-detect and convert between Kazakh scripts
- [ ] **Kazakh document OCR** — Tesseract/EasyOCR with kk language pack
- [ ] **kk benchmarks** — eval suite for Kazakh text quality, ASR WER, TTS MOS

**Models to explore:**
- `google/gemma-3-4b` (multilingual, good kk)
- `Qwen/Qwen3-8B` (strong multilingual)
- Custom fine-tune on Kazakh Wikipedia + news corpus

---

## Phase 7 — Extended Modalities (v1.3)

**Goal:** Complete the modality matrix.

| Pipeline | Model Candidate | Priority |
|----------|----------------|----------|
| Text → Speech | VITS / Bark / Piper | High |
| Audio → Audio | Whisper + TTS (voice translate) | Medium |
| Video → Video | frame extraction + SDXL + ffmpeg stitch | Medium |
| Image → Video | AnimateDiff / Stable Video Diffusion | Low |
| Document → Document | extract + LLM + PDF generation | Medium |
| Code → Code | Qwen3-4B (code mode) | Medium |

- [ ] **TTS pipeline** — text→speech with language selection
- [ ] **Voice translation** — audio→text→translate→speech chain
- [ ] **Video editing** — frame-by-frame SDXL with temporal consistency
- [ ] **Code assistant** — code completion and refactoring pipeline

---

## Phase 8 — Multi-GPU & Scaling (v2.0)

**Goal:** Production-grade distributed inference.

- [ ] **Multi-GPU dispatch** — model placement across GPUs based on VRAM
- [ ] **Model parallelism** — split large models across devices
- [ ] **Request queue** — async job queue with priority (Redis/in-memory)
- [ ] **Batch inference** — group similar requests for throughput
- [ ] **Auto-scaling** — dynamic model loading based on request patterns
- [ ] **Metrics** — Prometheus endpoints: latency, throughput, GPU util, model load times
- [ ] **Health probes** — Kubernetes-ready liveness/readiness checks

---

## Phase 9 — Ecosystem (v2.1)

**Goal:** Make OmniFF a platform.

- [ ] **Plugin marketplace** — community model plugins via pip namespace packages
- [ ] **Graph template library** — shared YAML templates for common workflows
- [ ] **Web UI** — React dashboard for runtime management, model status, request history
- [ ] **Jupyter integration** — `%%omniff` magic for notebooks
- [ ] **LangChain/LlamaIndex tool** — OmniFF as a tool in agent frameworks
- [ ] **Documentation site** — MkDocs/Docusaurus with tutorials, API reference
- [ ] **NPM package** — `@saken/omniff` JS/TS client SDK

---

## Phase 10 — Research & Paper (v3.0)

**Goal:** Academic publication and benchmarks.

- [ ] **Benchmark suite** — standardized eval across all modality pairs
- [ ] **Latency profiling** — per-component timing breakdown
- [ ] **Comparison paper** — OmniFF vs raw model inference vs LangChain vs other runtimes
- [ ] **Kazakh AI benchmark** — first comprehensive Kazakh multimodal eval
- [ ] **ArXiv paper** — "OmniFF: A Universal Multimodal Runtime Inspired by FFmpeg"
- [ ] **Conference submission** — EMNLP / ACL / NeurIPS demo track

---

## Priority Matrix

```
                    High Impact
                        │
     Phase 6 (KZ)  ────┼──── Phase 5 (Production)
                        │
     Phase 10 (Paper) ──┼──── Phase 7 (Modalities)
                        │
     Phase 9 (Eco)  ────┼──── Phase 8 (Scaling)
                        │
                    Low Impact

         Low Effort ────────── High Effort
```

**Recommended order:**
1. **Phase 5** — CI/CD + PyPI + Docker (1-2 weeks)
2. **Phase 6** — OmniFF-KZ (2-3 weeks, high differentiation)
3. **Phase 7** — TTS + voice translate (2 weeks)
4. **Phase 10** — paper draft (parallel with above)
5. **Phase 8** — scaling when demand arrives
6. **Phase 9** — ecosystem when community grows

---

## Quick Wins (can do today)

- [ ] `ruff check python/` — add linting
- [ ] GitHub Actions CI for unit tests
- [ ] PyPI publish (`twine upload`)
- [ ] HuggingFace Space deploy
- [ ] Add GitHub topics: `ai`, `multimodal`, `ffmpeg`, `kazakh`, `runtime`
