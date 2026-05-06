# Decisions and Concerns

## 2026-05-06

### Decision 1: Training diffusion mask built explicitly in [fuse | answer] space
- File: diffuvqa/gaussian_diffusion.py
- Change: Removed repeat and pad based mask expansion logic.
- Change: Added explicit mask construction with fuse region as 0 and answer region as 1.
- Reason: Repeat and pad based expansion can misalign semantics between conditioned and generated regions.
- Concern: Existing checkpoints trained with old semantics may not transfer behavior cleanly to the fixed setup.
- Follow-up: Run a short training and sampling sanity check to confirm shape stability and expected generation behavior.

### Decision 2: Sampling answer segment initialized without ground-truth answer embeddings
- File: sample_vqa_GPU.py
- Change: Removed use of model.get_embeds(input_a_id) in x_start answer segment initialization.
- Change: Added zero initialization for the answer segment while preserving fuse segment from model conditioning.
- Reason: Using ground-truth answer embeddings in x_start can leak answer information and distort generation quality assessment.
- Concern: Immediate output quality may drop until model learns stronger answer generation under corrected semantics.
- Follow-up: Re-run tiny end-to-end evaluation and monitor empty answer rate and exact match trend across checkpoints.

### Decision 3: Ongoing tracking requirement implementation
- File: shared/TODO.md
- Change: Added Engineering Execution Log section for active execution tasks.
- Reason: Maintain a single todo checklist for all next actions.
- Concern: TODO can grow quickly if not pruned after milestone completion.
- Follow-up: Keep only active and near-term items in execution log and move completed long-term notes to changelog.

### Decision 4: Restore executable mask validation script in repository
- File: scripts/test_sampling_mask_leakage.py
- Change: Added test utility file to workspace because runtime invocation failed due to missing script path.
- Reason: New fuse or answer mask semantics require reproducible validation from repository state.
- Concern: Script defaults still compare legacy and fixed behavior; users can misread legacy fail as current code fail.
- Follow-up: Keep using tiny-mode compare only as regression signal and interpret legacy section as baseline reference.

### Decision 5: Validate new fuse|answer semantics with tiny compare run
- Files: scripts/test_sampling_mask_leakage.py, diffuvqa/gaussian_diffusion.py, sample_vqa_GPU.py
- Change: Executed module test on cpu with known checkpoint and SLAKE dataset.
- Result: legacy_generation_ratio=0.0, fixed_generation_ratio=1.0, leakage_detected=true.
- Result: tiny e2e on 4 samples -> legacy exact_match=1.0, fixed exact_match=0.0.
- Reason: Confirms mask semantics split is functioning and legacy leakage signature remains detectable.
- Concern: Fixed exact match remains low on weak checkpoint; this reflects model quality/training stage, not mask wiring failure.
- Follow-up: Re-evaluate after additional training steps and monitor fixed-mode empty rate plus exact match trend.

### Clarification: Is current implementation leaking answer initialization?
- Short answer: Current fixed implementation is non-leaky; legacy behavior is leaky.
- Evidence: In test output, legacy_generation_ratio=0.0 and fixed_generation_ratio=1.0.
- Interpretation: Legacy mask fails to diffuse the answer region; fixed mask diffuses the full answer region as intended.
- Important note: leakage_detected=true in the report refers to legacy baseline detection, not a failure of the current fixed path.

### Decision 6: Fix answer slicing robustness in sampling output
- File: sample_vqa_GPU.py
- Change: Replaced half split slicing with explicit [fuse_len : fuse_len + answer_len] slicing.
- Reason: sample.size(1)//2 is fragile when fuse_len and answer_len diverge; explicit indexing keeps answer extraction aligned to mask semantics.
- Concern: If upstream code changes answer_len semantics, slice bounds should be revalidated in smoke tests.
- Follow-up: Keep shape assertion or quick check in future sampling tests for answer segment length.

### Decision 7: Resolve undefined BertLayer in custom BertEncoder path
- File: diffuvqa/language_encoders/bert_model.py
- Change: Added import for BertLayer from transformers.models.bert.modeling_bert.
- Reason: Custom BertEncoder instantiated BertLayer without a local definition, causing NameError risk in non-pretrained init paths.
- Concern: Transformers internal API paths may change across major versions.
- Follow-up: If upgrading transformers, verify BertLayer import compatibility and add fallback guard if needed.

### Decision 8: Align diffusion training losses to answer segment semantics
- File: diffuvqa/gaussian_diffusion.py
- Change: MSE target now uses only answer segment embeddings (x_start_mean) instead of full [fuse|answer] concat target.
- Change: t0_loss compares answer prediction against clean answer embeddings.
- Change: tT_loss and decoder_nll now use x_start_mean (clean answer embeddings), not noisy x_start.
- Reason: Prevent fuse region from dominating training loss and keep auxiliary regularizers tied to clean answer semantics.
- Concern: q_sample add_information path still injects auxiliary conditioning and should be monitored for shortcut behavior.
- Follow-up: Compare short-run metrics (empty rate, exact match trend, answer-token confidence) before/after this change.
