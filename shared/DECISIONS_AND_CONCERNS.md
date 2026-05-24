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

## 2026-05-11

### Decision 19: Align diffusion training noising path with non-leaky sampling default
- Files: diffuvqa/gaussian_diffusion.py, diffuvqa/config.json
- Change: Added runtime flag `use_noising_f` and defaulted it to `false` in config.
- Change: In `training_losses_seq2seq`, `f` is now set to `cond_x_start` only when `use_noising_f=True`; otherwise `f=None` and `q_sample(..., add_information=False)` is used.
- Change: Preserved `x_start_mean = ans_emb` as the clean denoising target; only the auxiliary noising shortcut was disabled by default.
- Reason: Training previously re-injected answer-side semantic signal through the auxiliary noising branch, while sampling defaulted to a harder non-leaky setup with no such shortcut.
- Concern: Historical checkpoints trained with the old shortcut behavior are not directly comparable to new runs that keep `use_noising_f=false`.
- Follow-up: Treat old and new runs as separate experiment families and avoid resuming across this objective boundary unless the flag is intentionally matched.

### Decision 20: Make pre_answer_loss an explicit weighted ablation instead of a hidden fixed behavior
- Files: diffuvqa/gaussian_diffusion.py, diffuvqa/config.json
- Change: Added runtime flag `pre_answer_loss_weight` with default `0.0`.
- Change: `pre_answer_loss` is now computed only when the weight is positive, with length alignment and answer-padding masking before contribution to total loss.
- Reason: The auxiliary fusion supervision may help metric recovery, but it should not silently redefine the main generative objective.
- Concern: Large weights can pull the fusion branch toward the raw token-embedding manifold and distort conditioning quality.
- Follow-up: If ablation is needed, start with small values such as `0.05` or `0.1` and compare empty-answer rate, exact match, and qualitative answer grounding against the default `0.0` run.

### Decision 21: Update Colab notebook to expose and validate the new training objective controls
- File: shared/run_diffuvqa_colab.ipynb
- Change: Added `USE_NOISING_F` and `PRE_ANSWER_LOSS_WEIGHT` to the notebook configuration cell.
- Change: Passed both flags through the training command cell to `train.py`.
- Change: Extended resume checkpoint compatibility checks to compare `use_noising_f` and `pre_answer_loss_weight` against the checkpoint's `training_args.json`.
- Reason: The notebook is the main execution surface; objective changes must be visible, intentional, and validated before long training runs.
- Concern: Users may try to resume older checkpoints with mismatched objective flags and assume this is a safe continuation.
- Follow-up: When resuming, keep the objective flags identical to the checkpoint metadata; otherwise start a fresh run with `RESUME_CHECKPOINT='none'`.

## 2026-05-20

### Decision 22: Exclude WordPiece ## continuation tokens from denoised_fn_round nearest-neighbour search
- Files: diffuvqa/rounding.py, sample_vqa_GPU.py
- Change: Added optional `subword_mask` parameter to `get_efficient_knn` and `denoised_fn_round`.
- Change: In `sample_vqa_GPU.py`, built a boolean mask over the full vocabulary for all token IDs whose string representation starts with `##`, then passed it to `denoised_fn_round` via `partial`.
- Change: At each DDIM step, `## ` token distances are set to `inf` before `topk`, preventing any `##`-starting token from ever being selected as the nearest neighbour during rounding.
- Reason: The diffusion rounding step runs at every denoising iteration. If a noisy answer embedding is closest to a `##` continuation token, the trajectory locks into that neighbourhood and stays there for all remaining steps, producing garbled outputs like `##OWzie` regardless of training progress.
- Concern: Excluding `##` tokens shifts the nearest-neighbour to the next-closest word-beginning token, which may still be semantically wrong at early checkpoints. This prevents the structural failure but does not substitute for sufficient training.
- Follow-up: Re-run inference on the same checkpoint (029000) and compare `##`-prefix rate and qualitative answer quality against the previous output to confirm the fix has effect.

### Decision 18: Align Bio-Bert runtime path to Bert-style implementation for speed and simplicity
- Files: diffuvqa/vqa_model.py, shared/basic_utils.py, sample_vqa_GPU.py
- Change: Removed unused Pooler class from model fusion code.
- Change: Replaced hardcoded image projection input channels with one-time dynamic probing from the selected vision encoder.
- Change: Replaced direct bert.embeddings module call with explicit token plus position plus token-type plus LayerNorm plus dropout composition.
- Change: Replaced fragile locals-based sequence alignment with explicit target_len-based pooling and expansion logic.
- Change: Switched lm_head to bias=False with normal initialization (no word embedding weight copy).
- Change: Removed runtime model-family preset normalization and strict validation helpers from shared/basic_utils.py.
- Change: Simplified model factory to a direct transformer-bert construction path for the Bio-Bert configuration.
- Change: Removed sampling-time imports and calls that depended on runtime preset validation and resolution.
- Change: Renamed model factory key from transformer-bert to transformer-bio-bert in shared/basic_utils.py and diffuvqa/config.json to reflect the actual pretrained weights and avoid confusion with the generic Bert branch.
- Reason: Reduce startup overhead, reduce early-step jitter from backend warm-up effects, and make the Bio-Bert code path structurally closer to Bert branch behavior while keeping the model identifier self-descriptive.
- Concern: Existing checkpoints trained before this change are not strict-load compatible when key paths include fuse.bert_embeddings.* and when lm_head.bias exists.
- Follow-up: Add and use a checkpoint key migration utility for legacy runs, or resume only from checkpoints produced after this refactor.

### Decision 24: Fix sqrt(0) NaN gradient in get_logits logits_mode=2 path
- File: diffuvqa/vqa_model.py
- Change: Changed `th.clamp(dist, 0.0, np.inf)` to `th.clamp(dist, 1e-12, np.inf)` in `get_logits` (logits_mode=2 branch).
- Reason: Squared L2 distances computed via floating-point matrix ops can produce small negative values (~-1.9e-6) due to rounding. Clamping to 0.0 produces exact zeros; `sqrt(0)` backward computes `1/(2*sqrt(0)) = inf`, propagating NaN gradients to all model parameters silently. Decision 8 made this deterministically triggerable at full training scale by routing `x_start_mean` (full vocab rows) through `_token_discrete_loss`, which calls `get_logits`. The bug is dormant in the current config (logits_mode defaults to 1) but would fire immediately if logits_mode=2 were activated.
- Concern: The fix is safe for mode=1 (unreachable code path). For mode=2, the 1e-12 floor shifts near-zero distances by a negligible amount with no meaningful effect on cosine similarity ranking.
- Follow-up: No immediate action required since logits_mode=1 is active. If logits_mode=2 is ever evaluated, verify avg_nn_l2 and gradient norms at step 1 to confirm NaN-free training.

### Decision 25: Fix model_kwargs pollution in training_losses_seq2seq
- File: diffuvqa/gaussian_diffusion.py
- Change: Changed `model_kwargs['input_a_id']` to `model_kwargs.pop('input_a_id')` so the key is removed before the model call.
- Change: Added `model_kwargs.pop('image_name', None)` immediately after `get_ddpm_input` returns, to clean the remaining unconsumed key.
- Reason: After all pops in training_losses_seq2seq, `input_a_id` and `image_name` were left in model_kwargs and leaked into the final `model(x_t, t, **model_kwargs)` call. TransformerNetModel.forward accepts only (x, timesteps), so any extra kwargs cause a TypeError when calling the model directly. During actual training the call goes through _WrappedModel which silently absorbs extra kwargs, masking the bug. Unit tests that call the model directly exposed it immediately.
- Concern: The fix does not change training numerics; the wrapped training path was already discarding these keys implicitly. Existing checkpoints are fully compatible.
- Follow-up: No retraining or checkpoint migration needed. The fix is a code-correctness cleanup only.

### Decision 26: Fix .view() on non-contiguous tensor in get_logits logits_mode=2 path
- File: diffuvqa/vqa_model.py
- Change: Changed `text_emb.view(-1, text_emb.size(-1))` to `text_emb.reshape(-1, text_emb.size(-1))` in get_logits (logits_mode=2 branch).
- Reason: The tensor arriving at this line has been through permute/transpose operations upstream and is non-contiguous in memory. PyTorch .view() requires contiguous layout and raises RuntimeError on non-contiguous tensors. .reshape() falls back to a copy when needed and handles both cases. This was a latent crash in the logits_mode=2 code path, exposed by the architecture test suite (TestDiffusionLoss, TestEndToEndGradientFlow).
- Concern: The fix is in the logits_mode=2 branch only, which is not active in the current config (logits_mode=1). No impact on existing training or checkpoints.
- Follow-up: No retraining needed. If logits_mode=2 is ever evaluated, this path is now safe to run.

### Decision 27: Port Bert architecture test suite to Bio-Bert and add tests/test_architecture.py
- File: tests/test_architecture.py (new, 36 tests across 8 modules)
- Change: Created tests/test_architecture.py ported from the Bert branch commit 6e3a955, adapted for Bio-Bert model args (vocab_size=28996, config_name=dmis-lab/biobert-base-cased-v1.2, model=transformer-bio-bert).
- Change: Replaced bert-base-uncased tokenizer references with dmis-lab/biobert-base-cased-v1.2 throughout.
- Change: Added FUSE_LEN=16 constant; FakeCLIP returns [B, 512, FUSE_LEN] so image-patch-count-based shape assertions are consistent.
- Change: Replaced test_bert_encoder_grad (trainable assertion) with test_bert_encoder_frozen (frozen assertion) to match Bio-Bert branch behavior where bert.encoder.layer is explicitly frozen.
- Change: Built SpacedDiffusion directly in build_diffusion() instead of calling create_model_and_diffusion, since transformer-bert key no longer exists in basic_utils.py.
- Reason: Running the ported tests immediately surfaced Decisions 25 and 26 bugs that were invisible to training. The suite provides the same pre-flight safety net the Bert branch has.
- Concern: Each test class constructs a full BioBERT model, making the full suite take ~4 minutes on CPU. This is acceptable for a pre-Colab check but should not be added to a hot CI loop.
- Follow-up: Run python -m pytest tests/test_architecture.py -v --tb=short before any significant code change or before starting a new Colab training run.

### Decision 28: Add post-processing to sampling output before JSONL write
- Files: sample_vqa_GPU.py, scripts/quick_eval.py (new)
- Change: Added `_postprocess_answer(gen, ref)` helper to `sample_vqa_GPU.py`, called on every decoded answer before writing to JSONL.
- Change: Post-processing steps: (1) strip BERT special tokens, (2) strip leading commas/dashes and trailing punctuation, (3) deduplicate consecutive repeated tokens ("Yes Yes" → "Yes"), (4) trim to reference length when ref is ≤ 3 tokens.
- Change: Added `scripts/quick_eval.py`, a standalone evaluation script that applies the same post-processing and computes exact_match, token_f1, bleu1, partial_match, yesno_acc, hash_rate, avg_conf, avg_nn_l2, and avg_ans_len across multiple checkpoints in a comparison table.
- Reason: Metric analysis across 029k–082k checkpoints showed yes/no accuracy was artificially suppressed by over-generation artifacts. Post-processing 082k step35 recovered yesno_acc from 0.29% → 6.98% and exact_match from 0.10% → 3.03%, confirming the regression was a generation artifact, not a model quality collapse.
- Concern: Trimming to reference length requires the reference answer to be available at sampling time, which it is (loaded from the dataset). Trimming is skipped when ref is longer than 3 tokens, so multi-word answers are unaffected.
- Concern: Trim-to-ref-length is a post-hoc evaluation fix that uses ground-truth label length. It recovers the correct token from an over-generated sequence but does not help the model generate correctly on unseen data at real inference time. The structural fix is continued training until the model learns to stop on its own.
- Note on the 3-token cutoff: Almost all yes/no and single-organ answers in SLAKE are 1–3 tokens ("Yes", "No", "Lung", "Large Bowel", "Right Kidney"). For these the model generates real noise after the correct answer and trimming to ref length recovers it cleanly. For longer references such as "Brain Edema, Brain Non - enhancing Tumor" (7 tokens) the trim is skipped entirely since those answers need all their tokens.
- Follow-up: All future checkpoint samples will be written with clean answers. Re-running `scripts/quick_eval.py` on new JSONL files will give raw exact-match numbers directly without needing a separate post-processing pass.

### Decision 23: Apply subword_mask to final logit selection to eliminate ## tokens from output
- File: sample_vqa_GPU.py
- Change: After `model.get_logits(sample)`, applied `masked_fill(-inf)` over all `##`-starting token positions in the logit tensor before softmax and topk.
- Reason: Decision 22 applied the subword_mask only to `denoised_fn_round`, which controls the intermediate diffusion trajectory. Analysis of the samplestep30 output (same checkpoint, seed102) confirmed that `##` tokens in the final output come from `model.get_logits` → logit argmax, which was completely independent of the rounding mask. The two samplestep25 and samplestep30 outputs were token-for-token identical for the first ~50 lines, proving the rounding fix had no effect on generated answers. The avg_nn_l2 drop (~15–20 units) confirmed the rounding path was affected, but that does not propagate to the discrete output.
- Concern: Masking ## tokens from logits forces the model to always select the next-best non-## token. At this training stage that token may still be semantically wrong, but the structural garbage (##OWzie, ##sedel, etc.) will be eliminated.
- Follow-up: Re-run inference and verify zero ##-prefixed outputs. Then assess whether remaining answer quality is improving with further training steps.
