# DiffMed-VQA TODO

This TODO is derived from [shared/DiffuVQA_Roadmap.md](shared/DiffuVQA_Roadmap.md).

Update rule:
- When a phase is completed, mark all completed tasks with [x].
- Add final notes under the phase Notes section.
- Add major decisions and deviations to [shared/CHANGELOG.md](shared/CHANGELOG.md) on the same day.

## Phase 1: Model Training and Optimization

Status: In Progress

- [x] Task 1.1 Training environment setup
- [x] Install and verify dependencies
- [x] Configure hyperparameters for baseline run
- [x] Prepare data loaders for medical image-text pairs
- [x] Verify GPU and memory setup
- [x] Configure logging and checkpoint output directory

- [ ] Task 1.2 Model execution for exactly 5,000 global steps
- [x] Start baseline training run
- [x] Monitor loss and validation metrics
- [ ] Save intermediate checkpoints (target cadence: every 500 steps)
- [ ] Confirm run completes without OOM or runtime failure

- [ ] Task 1.3 Checkpoint exporting
- [ ] Save final 5,000-step checkpoint
- [ ] Export training args with checkpoint
- [ ] Create run metadata summary (dataset, steps, metrics, notes)
- [ ] Validate checkpoint reload in inference path
- [ ] Document checkpoint naming and location

Phase 1 Notes:
- Completed stabilization and smoke-test fixes in training/inference path:
- Fixed microbatch and loss-backprop flow issues in shared/train_util.py.
- Re-enabled distributed initialization and microbatch argument wiring in train.py.
- Fixed sample shape and conditional mask behavior in sample_vqa_GPU.py.
- Fixed direct module run issue in diffuvqa/vqa_datasets.py by iterating DataLoader correctly.
- Fixed Colab CUDA device selection in diffuvqa/utils/dist_util.py (cuda index).
- Current state: Cell 11 training on Colab runs successfully; 5,000-step full run is still pending.

## Phase 2: Inference Script Engineering

Status: Not Started

- [ ] Task 2.1 Model loading and initialization
- [ ] Implement reusable checkpoint loader for 5,000-step weights
- [ ] Rebuild model and diffusion objects from training config
- [ ] Set eval mode and device placement
- [ ] Verify state dict compatibility and fallback behavior

- [ ] Task 2.2 Input preprocessing pipeline
- [ ] Implement image resize and normalization
- [ ] Implement question tokenization matching training format
- [ ] Build masks and tensors compatible with model input
- [ ] Validate preprocessing output shapes

- [ ] Task 2.3 Core inference logic
- [ ] Implement single image + single prompt inference loop
- [ ] Generate answer or reasoning output
- [ ] Add robust runtime error handling
- [ ] Return reusable response object for API layer

- [ ] Task 2.4 Result post-processing and logging
- [ ] Remove special tokens and formatting artifacts
- [ ] Return readable response and structured JSON
- [ ] Log prompt, image id, checkpoint, latency, output
- [ ] Persist inference logs for debugging

Phase 2 Notes:
- Pending

## Phase 3: AI Chat Interface UI and UX

Status: Not Started

- [ ] Task 3.1 UI layout design
- [ ] Build dual-panel layout (chat + image upload/preview)
- [ ] Add clear labels, status, and response regions

- [ ] Task 3.2 Component development
- [ ] Build text input, image upload, and response bubble components
- [ ] Add conversation reset action
- [ ] Implement Streamlit first version
- [ ] Evaluate migration path to React if needed

- [ ] Task 3.3 Interactive feedback
- [ ] Add loading spinners and processing indicators
- [ ] Disable inputs during inference request
- [ ] Add user-friendly error states

Phase 3 Notes:
- Pending

## Phase 4: System Integration and Deployment

Status: Not Started

- [ ] Task 4.1 Backend API development
- [ ] Wrap Phase 2 inference in FastAPI service
- [ ] Implement endpoint for image + question input
- [ ] Return structured JSON response
- [ ] Reuse core inference module with no duplicated logic

- [ ] Task 4.2 REST or WebSocket integration
- [ ] Implement REST integration first
- [ ] Define stable request and response schema
- [ ] Add WebSocket only if streaming is required

- [ ] Task 4.3 End-to-end testing
- [ ] Test full upload-to-answer flow
- [ ] Validate preprocessing and rendering correctness
- [ ] Compare sample outputs with evaluation tooling
- [ ] Verify failure logging and reproducibility

Phase 4 Notes:
- Pending

## Cross-Phase Quality Gates

- [ ] Every phase completion has a changelog entry in [shared/CHANGELOG.md](shared/CHANGELOG.md)
- [ ] Every phase completion updates this file status and notes
- [ ] Key run artifacts are stored with clear naming
- [ ] Decisions that alter scope, stack, or metrics are documented