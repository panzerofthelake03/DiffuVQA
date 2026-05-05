# DiffMed-VQA Changelog

All notable project decisions and phase outcomes are tracked in this file.

Format:
- Date: YYYY-MM-DD
- Phase: Phase N
- Type: Added, Changed, Fixed, Decision, Risk, Validation
- Notes: concise summary with impact

Update protocol:
- Update this file immediately after each phase is completed.
- Record both what was done and why decisions were made.
- Link relevant artifacts (checkpoint names, script paths, test outputs).

## [Unreleased]

### 2026-04-20 - Phase 1 - Stabilization Progress

- Type: Fixed
  - Fixed training microbatch behavior and moved per-microbatch training updates to the correct loop in [shared/train_util.py](shared/train_util.py).
  - Restored distributed setup call and microbatch argument wiring in [train.py](train.py).
  - Fixed sample tensor shape and conditional mask behavior in [sample_vqa_GPU.py](sample_vqa_GPU.py).
  - Fixed DataLoader iteration in module smoke test path in [diffuvqa/vqa_datasets.py](diffuvqa/vqa_datasets.py).
  - Fixed CUDA device selection to use indexed device (e.g., cuda:0) in [diffuvqa/utils/dist_util.py](diffuvqa/utils/dist_util.py) for Colab compatibility.

- Type: Validation
  - Verified mask behavior via smoke test: legacy path no-ops, patched path performs diffusion updates.
  - Verified training startup and progression on Colab after device fix (Cell 11 now trains successfully).
  - Verified vqa_datasets module smoke test runs past tokenization after DataLoader iterator fix.

- Type: Decision
  - Kept Phase 1 status as In Progress until an uninterrupted 5,000-step run and final checkpoint export are complete.

### 2026-04-19 - Phase 0 - Planning Baseline

- Type: Added
  - Created roadmap in [shared/DiffuVQA_Roadmap.md](shared/DiffuVQA_Roadmap.md).
  - Added execution tracker in [shared/TODO.md](shared/TODO.md).

- Type: Decision
  - Delivery sequence selected: Phase 1 training baseline, then Phase 2 inference, then Phase 4 API integration, then Phase 3 UI integration.
  - Initial UI path selected: Streamlit for rapid prototype.
  - Initial backend path selected: FastAPI for typed API and validation.
  - Integration path selected: REST first, WebSocket only if streaming is needed.

- Type: Risk
  - Checkpoint compatibility risk between training args and inference loader identified.
  - GPU memory pressure risk identified for training and sampling workflows.

## Phase Completion Template

Use this section as a copy template whenever a phase is completed.

### YYYY-MM-DD - Phase N - Completion

- Type: Added
  - Implemented:
  - Artifacts:

- Type: Changed
  - Scope changes:
  - Parameter changes:

- Type: Fixed
  - Issues resolved:

- Type: Decision
  - Final stack choices:
  - Trade-offs accepted:

- Type: Validation
  - Tests run:
  - Metrics observed:
  - Acceptance criteria status:

- Type: Risk
  - Remaining risks:
  - Mitigation next step: