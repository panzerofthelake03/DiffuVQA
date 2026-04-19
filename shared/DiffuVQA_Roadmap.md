# DiffMed-VQA Roadmap

This roadmap is tailored to the current DiffuVQA codebase and focuses on turning the trained model into a usable inference service and chat interface.

## Phase 1: Model Training and Optimization

### Task 1.1: Training Environment Setup
Configure the training pipeline with the correct hyperparameters, data loaders, and compute resources for medical image-text pairs.

Deliverables:
- Set up the training environment with all required dependencies.
- Configure hyperparameters (learning rate, batch size, diffusion steps, noise schedule, etc.).
- Prepare data loaders for SLAKE, Kvasir-VQA, and Med-VQA-2019 datasets.
- Verify GPU availability and memory allocation.
- Set up logging and checkpoint directories.

Acceptance criteria:
- The training pipeline is ready to start without errors.
- All datasets are correctly loaded and preprocessed.
- Hyperparameters match the desired configuration (e.g., diff_steps=2000, lr=0.00001, hidden_dim=64).

### Task 1.2: Model Execution (5,000 Steps)
Initiate the training process for exactly 5,000 global steps to establish the baseline model weights.

Deliverables:
- Execute the training loop with the configured hyperparameters.
- Monitor loss metrics and validation performance during training.
- Save intermediate checkpoints at regular intervals (e.g., every 500 steps).
- Log training progress including loss, learning rate, and time per step.
- Ensure the training completes successfully without interruption.

Acceptance criteria:
- Training runs to completion without crashes or OOM errors.
- Loss curves show expected convergence behavior.
- Intermediate checkpoints are saved consistently.

### Task 1.3: Checkpoint Exporting
Save and export the trained model weights for use in the inference stage.

Deliverables:
- Save the final model state dictionary after 5,000 steps.
- Export the training configuration (hyperparameters) alongside the checkpoint.
- Create a metadata file documenting the training run (dataset, steps, performance metrics).
- Verify checkpoint integrity by attempting to reload it.
- Document the checkpoint location and naming convention for downstream inference.

Acceptance criteria:
- The checkpoint file is created and can be loaded without errors.
- Configuration and metadata are saved alongside the checkpoint.
- Checkpoint size and format are appropriate for deployment.

## Phase 2: Inference Script Engineering

### Task 2.1: Model Loading and Initialization
Build a reusable inference script that loads the trained checkpoint and reconstructs the same Diffusion/Transformer architecture used in training.

Deliverables:
- Load the 5,000-step model weights from the trained checkpoint.
- Recreate the model and diffusion objects using the same configuration path as training.
- Move the model to the target device and set it to evaluation mode.
- Verify compatibility with the saved training arguments.

Acceptance criteria:
- A single command can load the model successfully on GPU or CPU fallback.
- The checkpoint loads without shape mismatches or missing keys.

### Task 2.2: Input Pre-processing Pipeline
Implement preprocessing functions that match the training data format for both image and text inputs.

Deliverables:
- Resize medical images to the expected resolution.
- Normalize images with the same mean and standard deviation used during training.
- Tokenize the user question with the same tokenizer and special tokens.
- Build the model input tensors and masks in the same format as the training loader.

Acceptance criteria:
- A user image and question are transformed into tensors compatible with the model.
- Preprocessing output matches the shape and structure expected by the inference loop.

### Task 2.3: Core Inference Logic
Build the execution loop that produces an answer or reasoning trace for a single image-question pair.

Deliverables:
- Accept one image and one prompt as input.
- Run the model through the diffusion sampling path.
- Support answer generation and, if available, optional reasoning-step output.
- Handle runtime errors gracefully and return informative messages.

Acceptance criteria:
- The script returns a stable answer for a known test image and prompt.
- The inference path is isolated and reusable by the backend API.

### Task 2.4: Result Post-processing and Logging
Format outputs for readability and add logging for debugging and traceability.

Deliverables:
- Clean model output text by removing special tokens and formatting artifacts.
- Return both human-readable text and structured JSON.
- Log prompt, image identifier, checkpoint path, latency, and generated response.
- Save inference records for later debugging and comparison.

Acceptance criteria:
- Outputs are readable in both console and UI contexts.
- Each inference call leaves a traceable log entry.

## Phase 3: AI Chat Interface UI and UX

### Task 3.1: UI Layout Design
Design a dual-panel interface with a chat history on one side and an image preview or upload area on the other.

Deliverables:
- Left panel for chat history and follow-up questions.
- Right panel for image upload and preview.
- Clear headers, status indicators, and response areas.

Acceptance criteria:
- Users can immediately understand where to upload an image and where to ask questions.

### Task 3.2: Component Development
Build the frontend components needed for input and response display using Streamlit or React.

Deliverables:
- Text input for the medical question.
- File upload control for images.
- Message bubbles or response cards for the conversation.
- Optional conversation reset control.

Recommended implementation path:
- Use Streamlit for the first prototype.
- Move to React only if you need richer interaction or production-grade UI control.

Acceptance criteria:
- Users can upload an image, ask a question, and view the model response in the same session.

### Task 3.3: Interactive Feedback
Add response states that make the inference experience feel responsive.

Deliverables:
- Loading spinners during inference.
- Processing indicators while the model is running.
- Disabled controls while a request is in flight.
- Friendly error messages for invalid uploads or backend failures.

Acceptance criteria:
- The UI clearly shows when inference is running and when the result is ready.

## Phase 4: System Integration and Deployment

### Task 4.1: Backend API Development
Wrap the inference script in a backend service so the UI can communicate with the model.

Deliverables:
- Expose an inference endpoint using FastAPI or Flask.
- Accept image upload plus question text.
- Return the generated answer in JSON format.
- Reuse the Phase 2 inference code directly instead of duplicating logic.

Recommended implementation path:
- Use FastAPI for typed request models, validation, and easier deployment.

Acceptance criteria:
- The backend can process a request independently of the UI.

### Task 4.2: WebSocket or REST Integration
Connect the frontend and backend for request and response exchange.

Deliverables:
- Use REST for the first stable integration.
- Add WebSocket support only if you need streaming or incremental updates.
- Define a clean request schema for image-text pairs.

Acceptance criteria:
- The frontend can send an image and prompt and receive the model response reliably.

### Task 4.3: End-to-End Testing
Verify the complete flow from image upload to rendered answer.

Deliverables:
- Test known images and prompts.
- Confirm image preprocessing is correct.
- Check that backend responses render correctly in the UI.
- Compare outputs against existing evaluation and sampling flows when needed.

Acceptance criteria:
- A full user journey works without manual intervention.
- Failures are reproducible and logged.

## Suggested Delivery Order

1. Set up the training environment and execute the model for 5,000 steps (Phase 1).
2. Export and validate the checkpoint (Phase 1.3).
3. Implement the standalone inference script (Phase 2).
4. Validate preprocessing and output formatting with known examples.
5. Wrap the inference module in a backend API (Phase 4).
6. Build the Streamlit UI and connect it to the backend (Phase 3).
7. Add end-to-end tests and log tracing.

## Repository Anchors

**Training Phase:**
- Training and setup documentation: [README.md](README.md)
- Main training script: [train.py](train.py)
- Training utilities and helpers: [shared/train_util.py](shared/train_util.py)
- Training run script: [scripts/run_train.py](scripts/run_train.py)
- Training shell script: [scripts/train.sh](scripts/train.sh)

**Inference Phase:**
- Existing sampling path: [sample_vqa_GPU.py](sample_vqa_GPU.py)
- Memory-efficient sampling reference: [efficient_sample.py](efficient_sample.py)
- Model architecture: [diffuvqa/vqa_model.py](diffuvqa/vqa_model.py)
- Data loading and tokenization: [diffuvqa/vqa_datasets.py](diffuvqa/vqa_datasets.py)

**Evaluation Phase:**
- Evaluation entry points: [eval_DiffuVQA.py](eval_DiffuVQA.py) and [eva_DiffuVQA.py](eva_DiffuVQA.py)
- Enhanced metrics module: [enhanced_eval_metrics.py](enhanced_eval_metrics.py)
- Excel export utilities: [shared/excel_export_module.py](shared/excel_export_module.py)

## Practical Recommendation

Start with a CLI inference module, then expose it through FastAPI, and use Streamlit for the first UI version. That sequence minimizes risk because each layer can be validated before the next one is added.