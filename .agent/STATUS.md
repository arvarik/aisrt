# aisrt Status
[STATE: HARDENED]

Last updated: 2026-08-18

_This file tracks the detailed explore/plan/build/test sub-phases per feature. It is the single source of truth for "where am I?" Agents should update this file after completing tasks or making progress._

## Current Focus
Comprehensive correctness, performance, robustness, and open-source review applied
across every module. See `CHANGELOG.md` under `Unreleased` for the full list.

## Current Milestone Tracking
- **Target**: Correctness, performance, and open-source hardening
- **Progress**: Complete. Lint, strict types, 319 tests, and an 85 percent coverage floor all pass.

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
- [Comprehensive hardening review] (2026-08-18)
  - Subtitle formatter rewritten to the Netflix Timed Text Style Guide.
  - Audio extraction rewritten onto a preallocated buffer, cutting peak memory by a third.
  - Discovery made streaming, iterative, and case-insensitive for exclude patterns.
  - State migrations made non-destructive.
  - Signal handling, exit codes, and byte-denominated memory bounds added.
  - Environment variables made functional, and the README verified against the code by a test.

## Known Issues
- `Formula/aisrt.rb` cannot install as written. Homebrew builds are network sandboxed and
  the formula declares no `resource` blocks. Either generate them with
  `brew update-python-resources` or retire the formula in favour of
  `uv tool install aisrt`.

## Blockers & Upstream Dependencies
- **Upstream Libraries**: Keep track if `faster-whisper` or `CTranslate2` releases break VRAM calculation heuristics.
- N/A — no active blockers.

## What's Next
- Decide the Homebrew question above.
- Consider a second inference backend for Apple Silicon GPUs, which CTranslate2 cannot
  reach. `whisper.cpp` with Metal or `mlx-whisper` are the candidates. This is a backend,
  not a flag.

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
