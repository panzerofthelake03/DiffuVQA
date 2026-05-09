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

### Decision 9: Stabilize BERTScore reporting in evaluation script
- File: eval_DiffuVQA.py
- Change: Replaced split BERTScore paths with one helper (lazy import + fallback model: distilbert-base-uncased).
- Reason: Avoid silent 0.0 due to startup import gating and inconsistent function usage.
- Concern: If bert_score package is missing entirely, metric still falls back to 0.0 with warning.
- Follow-up: Verify environment package install when avg_bert_score remains 0 after this patch.

### Decision 10: Align Bio-Bert notebook training log/save cadence with PubMedBERT
- File: shared/run_diffuvqa_colab.ipynb
- Change: SAVE_INTERVAL updated 200 -> 2000 and LOG_INTERVAL updated 20 -> 100 in config cell.
- Reason: Reduce Colab/Drive I/O and logging overhead to match faster PubMedBERT notebook cadence.
- Concern: Sparser checkpoints increase potential progress loss window on interruptions.
- Follow-up: If frequent recovery is needed, consider SAVE_INTERVAL=1000 as middle ground.

### Decision 11: Implement true training continuation from intermediate checkpoints
- File: shared/train_util.py
- Change: Added real resume model loading in _load_and_sync_parameters, total-step-aware loop stop condition, optimizer state save/load (optXXXXXX.pt), and robust resume-step parsing.
- Change: Re-enabled main model checkpoint save alongside EMA for safer resume targets.
- Reason: Continue training from existing checkpoints with optimizer dynamics preserved, not just warm-start weights.
- Concern: Resuming from EMA-only checkpoints remains possible but may not perfectly match non-EMA continuation.
- Follow-up: Prefer RESUME_CHECKPOINT pointing to main model checkpoint when available for exact continuation.

## 2026-05-08

### Decision 12: Normalize runtime model family from model, vocab, config, and init signals
- File: shared/basic_utils.py
- Change: Added a runtime preset table for bert, bio-bert, and roberta families.
- Change: Added normalization helpers that resolve and validate model family from model, vocab, config_name, and use_plm_init.
- Reason: The previous setup allowed mixed runtime identities such as BioBERT tokenizer plus BERT model config, which made training and sampling behavior ambiguous.
- Concern: Older checkpoints with inconsistent metadata will now be flagged instead of being sampled silently.
- Follow-up: Audit existing historical checkpoints before reuse and repair metadata only when the true training family is known.

### Decision 13: Make transformer construction args-driven instead of hardcoded per branch
- File: shared/basic_utils.py
- Change: Replaced hardcoded transformer-bert, transformer-bio-bert, and transformer-roberta constructor constants with args-driven values for hidden dims, dropout, config_name, vocab_size, and init mode.
- Reason: The active training or sampling configuration should determine the runtime model, not stale hardcoded constants in the factory.
- Concern: New training runs may diverge from older notebook behavior because they now use the intended preset consistently.
- Follow-up: Treat pre-fix and post-fix checkpoints as separate experiment families when comparing results.

### Decision 14: Fail fast on conflicting checkpoint runtime metadata during sampling
- File: sample_vqa_GPU.py
- Change: Added explicit validation after loading checkpoint training_args.json and before model creation.
- Change: Sampling now raises a clear error when model, vocab, config_name, and use_plm_init imply different model families.
- Reason: Silent fallback or partial normalization can hide root-cause metadata errors and produce misleading generations.
- Concern: Some legacy Bio-Bert checkpoints that previously sampled under ambiguous settings may no longer run without metadata review.
- Follow-up: If legacy checkpoint reuse is required, inspect training_args.json and only repair it when the true pretrained family is verified from the original run.

### Decision 15: Bind Colab notebook model selection to explicit presets and add smoke tests
- File: shared/run_diffuvqa_colab.ipynb
- Change: Replaced free-form model naming with MODEL_PRESET_KEY and derived MODEL_ARCH, VOCAB_NAME, CONFIG_NAME, and USE_PLM_INIT.
- Change: Added a preflight smoke test before training and a sampling smoke test before actual sampling.
- Reason: The notebook should launch consistent runtime settings and catch broken dataset, image path, or sampling-mask issues before expensive runs.
- Concern: Users may assume old checkpoints remain drop-in compatible with the new notebook flow, which is not guaranteed for inconsistent historical runs.
- Follow-up: When reusing older checkpoints, prefer a compatibility audit first, then run the notebook smoke tests before full sampling.

### Decision 16: Cross-repo follow-up required for PubMedBERT repository parity
- Files: shared/DECISIONS_AND_CONCERNS.md, shared/PUBMEDBERT_REPO_AUDIT_PROMPT.md
- Change: Added an explicit follow-up to audit the PubMedBERT repository for the same model-family mismatch, checkpoint metadata, notebook preset, and smoke-test gaps.
- Reason: Bio-Bert and PubMedBERT repos appear to share workflow patterns, so the same class of failures may exist there too.
- Concern: Applying only the Bio-Bert fix set can leave PubMedBERT results inconsistent and invalidate branch-to-branch comparisons.
- Follow-up: Run the prepared audit-and-fix prompt against the PubMedBERT repo and compare checkpoint compatibility rules before new experiments.

## 2026-05-10

### Decision 17: Add inference-time reliable span extraction with confidence filtering
- File: sample_vqa_GPU.py
- Change: Added confidence_threshold runtime arg to decode defaults (default 0.3).
- Change: Added per-position max probability computation from model logits and masked low-confidence predicted tokens to PAD before decoding.
- Change: Added post-processing stop rule to decode only until first SEP or PAD token, yielding the shortest reliable answer span.
- Reason: Generated answers were often overlong; this fix trims noisy tails without retraining and is compatible with existing checkpoints.
- Concern: Threshold sensitivity can affect recall vs precision tradeoff, especially for weaker checkpoints.
- Follow-up: Run threshold sweep at 0.2, 0.3, 0.4 and compare empty-answer rate, exact match, and qualitative answer brevity.
