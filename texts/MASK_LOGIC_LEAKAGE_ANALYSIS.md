# Mask Logic Leakage Analysis (Sampling vs Inference)

## Summary
This branch showed a mismatch:
- Sampling evaluation looked high-accuracy.
- Single-image inference looked low-quality.

Root cause is mask logic in sampling paths: old mask construction can leave answer tokens conditioned instead of generated.

## Symptom
- In batch sampling, outputs can closely match references even with early checkpoints.
- In individual inference, answers may be empty or low-quality.

This is expected if sampling pipeline leaks answer information while inference does not.

## Key Findings
From `scripts/test_sampling_mask_leakage.py` runs:
- `legacy_generation_ratio = 0.0`
- `fixed_generation_ratio = 1.0`
- `leakage_detected = true`

Interpretation:
- Old mask logic generated/noised 0% of non-pad answer tokens.
- Fixed mask logic generated/noised 100% of non-pad answer tokens.

## Why This Happens
Dataset preprocessing creates `input_mask` for `[question | answer]`:
- question region: `0`
- answer region: `1`

But diffusion input is shaped as `[fuse_feats | answer_emb]`.
If mask is built by concatenating `[fuse_zeros | input_mask]` and then truncating, the answer region can accidentally align with question zeros. That conditions true answer embeddings instead of forcing generation.

## Related Code Snippets

### 1) Dataset mask definition (`0` for question, `1` for answer)
From `diffuvqa/vqa_datasets.py`:

```python
# Merge question and answer ids and create the mask (0 for question, 1 for answer)
input_ids = [q + a for q, a in zip(input_id_q, input_id_a)]
input_mask = [[0] * len(q) + [1] * len(a) for q, a in zip(input_id_q, input_id_a)]
```

### 2) Old mask logic pattern (diagnostic reproduction)
From `scripts/test_sampling_mask_leakage.py`:

```python
if mode == "legacy":
    fuse_mask = th.zeros((bsz, fuse_len), dtype=input_mask.dtype, device=input_mask.device)
    full_mask = th.cat([fuse_mask, input_mask], dim=1)
```

Then padded/truncated to `x_start` length:

```python
total_len = x_start.size(1)
cur_len = full_mask.size(1)
if cur_len < total_len:
    pad_len = total_len - cur_len
    pad_tensor = th.zeros((bsz, pad_len), dtype=full_mask.dtype, device=full_mask.device)
    full_mask = th.cat([full_mask, pad_tensor], dim=1)
elif cur_len > total_len:
    full_mask = full_mask[:, :total_len]
```

### 3) Fixed mask logic (correct)
From `sample_vqa_GPU.py`:

```python
# Build generation mask aligned with x_start = [fuse | answer].
# Keep fuse region fixed (0) and diffuse answer region (1).
fuse_len = fuse_feats.size(1)
bsz = input_ids_mask.size(0)
answer_len = input_emb.size(1)
full_mask = th.cat([
    th.zeros((bsz, fuse_len), dtype=input_ids_mask.dtype, device=input_ids_mask.device),
    th.ones((bsz, answer_len), dtype=input_ids_mask.dtype, device=input_ids_mask.device),
], dim=1)
```

And apply noising only where mask is `1`:

```python
input_ids_mask = th.broadcast_to(full_mask.unsqueeze(dim=-1), x_start.shape)
x_noised = th.where(input_ids_mask == 0, x_start, noise)
```

### 4) Inference pipeline uses fixed logic
From `Inference_Script_Part/pipeline.py`:

```python
# The fuse region is fixed (mask=0); the answer region must be noised (mask=1)
bsz = fuse_feats.size(0)
fuse_len = fuse_feats.size(1)
answer_len_mask = input_emb.size(1)
full_mask = torch.cat([
    torch.zeros((bsz, fuse_len), dtype=torch.long, device=self.device),
    torch.ones((bsz, answer_len_mask), dtype=torch.long, device=self.device),
], dim=1)
```

## Practical Consequence
- Sampling with old mask logic can overestimate model quality.
- Inference quality is the trustworthy behavior because reference answers are not provided there.

## Recommendation
- Keep mask construction explicit for diffusion input layout `[fuse | answer]`:
  - fuse part -> `0`
  - answer part -> `1`
- Avoid deriving generation mask directly from dataset `input_mask` (`[question | answer]`) unless remapped carefully.
- Use `scripts/test_sampling_mask_leakage.py` as a regression check when porting across branches.
