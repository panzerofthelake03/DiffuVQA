# Sampling vs Inference Mismatch: Mask Logic Leakage

## Summary

This branch can show **high sampling accuracy** but **low real inference quality** because sampling may unintentionally keep ground-truth answer tokens conditioned.

The mismatch happens when the diffusion mask for `[fuse | answer]` is built from dataset `input_mask` (`[question(0) | answer(1)]`) in a way that does not align with `x_start` layout.

## Observable Symptom

- Batch sampling (`sample_vqa_GPU.py`) may produce suspiciously strong exact-match behavior.
- Single-image inference (no reference answer available) performs much worse.

This is a classic leakage signature.

## Root Cause (Current Code Paths)

### 1) Dataset mask semantics are for Q+A token stream

From `diffuvqa/vqa_datasets.py`:

```python
# Merge question and answer ids and create the mask (0 for question, 1 for answer)
input_ids = [q + a for q, a in zip(input_id_q, input_id_a)]
input_mask = [[0] * len(q) + [1] * len(a) for q, a in zip(input_id_q, input_id_a)]
```

So `input_mask` describes `[Q | A]`, not `[fuse | A]`.

### 2) Sampling script currently builds diffusion mask by concatenating fuse zeros + input_mask

From `sample_vqa_GPU.py`:

```python
input_ids_mask = cond.pop('input_mask').to(device)
...
x_start = torch.cat([fuse_feats, input_emb], dim=1)

# Build a full mask that covers the image-fuse tokens (zeros) + text tokens (input_ids_mask)
fuse_len = fuse_feats.size(1)
bsz = input_ids_mask.size(0)
fuse_mask = th.zeros((bsz, fuse_len), dtype=input_ids_mask.dtype, device=input_ids_mask.device)
full_mask = th.cat([fuse_mask, input_ids_mask], dim=1)

# Ensure full_mask length matches x_start sequence length; pad or truncate as needed
...
input_ids_mask = th.broadcast_to(full_mask.unsqueeze(dim=-1), x_start.shape).to(device)
```

But `x_start` is `[fuse | answer_emb]`, not `[fuse | question | answer]`.

### 3) Diffusion engine trusts the provided mask and anchors where mask==0

From `diffuvqa/gaussian_diffusion.py`:

```python
if mask == None:
    return x_t
else:
    mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
    return th.where(mask == 0, x_start, x_t)
```

So any wrong zero region in the provided mask will keep original tokens fixed.

### 4) Inference pipeline uses the corrected mask logic

From `Inference_Script_Part/pipeline.py`:

```python
# Build the generation mask aligned with x_start = [fuse(N) | answer(A)].
# The fuse region is fixed (mask=0); the answer region must be noised (mask=1)
bsz = fuse_feats.size(0)
fuse_len = fuse_feats.size(1)
answer_len_mask = input_emb.size(1)
full_mask = torch.cat([
    torch.zeros((bsz, fuse_len), dtype=torch.long, device=self.device),
    torch.ones((bsz, answer_len_mask), dtype=torch.long, device=self.device),
], dim=1)
```

This path correctly forces answer generation.

## Why This Creates High Sampling but Low Inference

1. In sampling, reference answers exist in the batch (`input_a_id`) and enter `x_start`.
2. If mask alignment is wrong, answer region may remain partially conditioned/fixed.
3. Output can then mirror reference answers and inflate metrics.
4. In single-image inference, no reference answer is available, so this advantage disappears.
5. Real generation quality is exposed and looks much lower.

## Additional Risk in Current Sampling Script

Current sampling shape uses:

```python
sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)
```

while `x_start` is `[fuse_len + answer_len, hidden_dim]`. Shape mismatch can further destabilize behavior.

## Recommended Fixes

1. In `sample_vqa_GPU.py`, build mask directly for `[fuse | answer]`:
   - fuse part: zeros
   - answer part: ones
2. Set sampling shape to `tuple(x_start.shape)`.
3. Keep `gaussian_diffusion.py` unchanged for this issue; it should consume, not reinterpret, mask semantics.
4. Continue using `scripts/test_sampling_mask_leakage.py` to verify:
   - old/legacy generation ratio near 0 indicates leakage risk
   - fixed generation ratio near 1 indicates proper answer generation

## Validation Checklist

- Run leakage metric mode (fast):
  - Expect old generation ratio << fixed generation ratio
- Run tiny E2E compare mode:
  - If old path has much higher exact-match than fixed on early checkpoints, leakage is likely
- Compare against single-image inference quality:
  - Large gap confirms pipeline artifact rather than genuine model learning
