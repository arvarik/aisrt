# aisrt Status
[STATE: INITIALIZED]

Last updated: 2026-04-14

_This file tracks the detailed explore/plan/build/test sub-phases per feature. It is the single source of truth for "where am I?" Agents should update this file after completing tasks or making progress._

## Current Focus
Project Bootstrapped. Ready for feature ideation.

## Current Milestone Tracking
- **Target**: Initial Daemon Stabilization / MPS Support Tuning
- **Progress**: Awaiting final pipeline benchmarking tests.

## State of Work
- [ ] Ideate: `docs/explorations/YYYY-MM-DD-{topic}.md`
- [ ] Design (Architecture): `docs/designs/YYYY-MM-DD-{feature}-architecture.md`
- [ ] Plan (Backend): `docs/plans/YYYY-MM-DD-{feature}-backend.md`
- [ ] Build (Backend)
- [ ] Test
- [ ] Review
- [ ] Ship

## Recently Completed
- [Project Bootstrapped] (shipped 2026-04-14)

## Known Issues
- (None currently)

## Blockers & Upstream Dependencies
- **Upstream Libraries**: Keep track if `faster-whisper` or `CTranslate2` releases break VRAM calculation heuristics.
- N/A — no active blockers.

## What's Next
Ready for first feature ideation.

## Relevant Files for Current Task
- (none)

## Review Results
### Review Results — 2026-04-14
- **Architecture**: pass
- **Security**: pass
- **Product fit**: pass

### Action Items
| Item | Severity | Route To | Status |
|------|----------|----------|--------|
| _None_ | | | |

## Active Worktrees
(none — sequential execution)

---

## Stub Audit Tracker

N/A — Not a full-stack project. No frontend stubs.

---

## Prompt Versioning Changelog

_Track changes to ML model configurations or processing parameters. Since this project uses ML inference (not LLM prompts), track model configuration changes here._

| Version | Date | Change Description | Quality Impact | Delta | Config File |
|---------|------|--------------------|----------------|-------|-------------|
| v1.0 | 2026-04-14 | Baseline model routing: `large-v3` (CUDA ≥ 10 GB), `large-v3-turbo` (CUDA ≥ 6 GB / Apple Silicon / high-RAM CPU), `small.en` (low-resource CPU). Transcribe params: `beam_size=5`, `vad_filter=True`, `word_timestamps=True`, `no_speech_threshold=0.6`, `compression_ratio_threshold=2.4`. | Baseline | — | `src/aisrt/hardware.py`, `src/aisrt/pipeline.py` |
