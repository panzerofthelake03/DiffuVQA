# PubMedBERT Repo Audit Prompt

Use the prompt below in the PubMedBERT repository.

```text
You are auditing the PubMedBERT DiffuVQA repository for the same runtime and notebook issues that were fixed in the Bio-Bert repository.

Goals:
1. Determine whether the PubMedBERT repo has the same model-family mismatch problem between model, vocab, config_name, and use_plm_init.
2. Determine whether transformer construction is still hardcoded instead of args-driven.
3. Determine whether sampling accepts inconsistent checkpoint metadata silently.
4. Determine whether the Colab notebook still uses free-form model naming instead of an explicit preset table.
5. Determine whether the notebook is missing pre-training and pre-sampling smoke tests.
6. If any of these issues exist, implement the same class of fixes with minimal repo-appropriate changes.

Audit scope:
1. Inspect shared/basic_utils.py or the equivalent runtime factory file.
2. Inspect sample_vqa_GPU.py or the equivalent sampling entrypoint.
3. Inspect the Colab notebook or notebook-equivalent experiment runner.
4. Inspect any training_args.json defaults and recent notebook logs if present.
5. Check whether old checkpoints may become incompatible after strict validation.

Required checks:
1. Verify whether tokenizer loading can select PubMedBERT or related vocab while model creation still uses BERT constants.
2. Verify whether config_name is hardcoded in transformer factory branches.
3. Verify whether vocab_size and hidden dims are hardcoded in the model factory.
4. Verify whether sampling loads training_args.json and merges it into runtime args without validating family consistency.
5. Verify whether notebook training cells derive model, vocab, config_name, and use_plm_init from one preset source.
6. Verify whether notebook workflow includes a lightweight smoke test before training.
7. Verify whether notebook workflow includes a sampling smoke test before actual sample generation.

If problems are found, apply this fix strategy:
1. Add a runtime preset table for the supported model families.
2. Add normalization helpers that infer the runtime family from model, vocab, config_name, and use_plm_init.
3. Add strict validation to reject conflicting runtime metadata during sampling.
4. Make model construction args-driven instead of hardcoded.
5. Update notebook model selection to use an explicit MODEL_PRESET_KEY-style flow.
6. Add notebook smoke tests before training and sampling.
7. Preserve existing public behavior unless the old behavior was ambiguous or unsafe.

Output requirements:
1. Report whether each of the five issue classes exists.
2. For every confirmed issue, cite the exact files and the local root cause.
3. If code is changed, summarize the edits and run the narrowest possible validation after each substantive edit.
4. State clearly whether older PubMedBERT checkpoints remain directly sample-compatible after the fix set.
5. If legacy checkpoints are at risk, describe the compatibility rule precisely.

Constraints:
1. Make minimal, focused edits.
2. Do not change unrelated training logic.
3. Do not rewrite notebooks broadly; only update the configuration and smoke-test flow.
4. Prefer fail-fast validation over silent fallback when checkpoint metadata is inconsistent.
```