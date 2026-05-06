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
