"""
Architecture diagnostic tests for DiffuVQA.

Each test isolates one module and checks:
  - output shapes are correct
  - gradients flow where expected
  - no NaN/Inf values
  - semantic sanity (embeddings move toward vocab, conditioning has effect)

Run:
    cd /path/to/DiffuVQA
    python -m pytest tests/test_architecture.py -v --tb=short
  or standalone:
    python tests/test_architecture.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import types
try:
    import pytest
except ImportError:
    pytest = None

def _approx(expected, tol=1e-5):
    class _Approx:
        def __eq__(self, other):
            return abs(other - expected) <= tol
    return _Approx()

# ── Minimal args stub ──────────────────────────────────────────────────────────
def make_args(**overrides):
    args = types.SimpleNamespace(
        hidden_dim=768,
        hidden_t_dim=768,
        hidden_size=768,
        extend_hidden_size=1024,
        input_image_embed_size=768,
        input_text_embed_size=768,
        image_encoder="ViT-B/32",
        image_resolution=384,
        num_heads=8,
        vocab="bert",
        vocab_size=30522,
        config_name="bert-base-uncased",
        seq_len=32,
        input_dims=768,
        output_dims=768,
        dropout=0.1,
        logits_mode=2,
        use_plm_init="bert",
        diffusion_steps=100,
        noise_schedule="sqrt",
        timestep_respacing="",
        learn_sigma=False,
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=True,
        rescale_learned_sigmas=False,
        sigma_small=False,
        use_noising_f=False,
        pre_answer_loss_weight=0.0,
        pad_token_id=0,
        checkpoint_path="/tmp",
        use_resnet152=False,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


# ── Fixtures ───────────────────────────────────────────────────────────────────
B = 2        # batch size
Q_LEN = 16  # question token length
A_LEN = 8   # answer token length
H = 768      # hidden dim
DEVICE = torch.device("cpu")


class _FakeCLIP(nn.Module):
    """CPU-compatible stub that mimics CLIP ViT-B/32 output shape [B, 512, 144]."""
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        B = x.size(0)
        return torch.randn(B, 512, 144, device=x.device)


def _patch_clip():
    """Monkey-patch CLIP at the point where vqa_model imports it, before model build."""
    import diffuvqa.vision_encoders.clip_model as clip_mod
    import diffuvqa.vqa_model as vqa_mod

    def _cpu_build(name, resolution_after=128, jit=False):
        return _FakeCLIP()

    # Patch both the clip_model module and the already-imported reference in vqa_model
    clip_mod.build_model = _cpu_build
    vqa_mod.build_model  = _cpu_build


_patch_clip()


def build_model():
    from diffuvqa.vqa_model import TransformerNetModel
    args = make_args()
    model = TransformerNetModel(
        input_dims=args.input_dims,
        output_dims=args.output_dims,
        hidden_t_dim=args.hidden_t_dim,
        dropout=args.dropout,
        config_name=args.config_name,
        vocab_size=args.vocab_size,
        init_pretrained=args.use_plm_init,
        logits_mode=args.logits_mode,
        args=args,
    )
    return model.to(DEVICE)


def build_diffusion(steps=100):
    from shared.basic_utils import create_model_and_diffusion
    args = make_args(diffusion_steps=steps, model="transformer-bert")
    _, diffusion = create_model_and_diffusion(args=args)
    return diffusion


def fake_image():
    return torch.randn(B, 3, 384, 384, device=DEVICE)


def fake_cond(q_len=Q_LEN, a_len=A_LEN):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    questions = ["what organ is shown?", "is this image normal?"][:B]
    answers   = ["liver", "no"][:B]
    q_enc = tok(questions, padding="max_length", max_length=q_len,
                truncation=True, return_tensors="pt")
    a_enc = tok(answers, padding="max_length", max_length=a_len,
                truncation=True, return_tensors="pt")
    mask = a_enc["attention_mask"]
    return {
        "input_q_id":  q_enc["input_ids"].to(DEVICE),
        "input_ids":   q_enc["input_ids"].to(DEVICE),
        "input_a_id":  a_enc["input_ids"].to(DEVICE),
        "input_mask":  mask.to(DEVICE),
        "image_name":  ["img1.jpg", "img2.jpg"][:B],
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Vision encoder (CLIP, frozen)
# ══════════════════════════════════════════════════════════════════════════════
class TestVisionEncoder:

    def setup_method(self):
        self.model = build_model()
        self.vision_enc = self.model.fuse.vision_encoder.to(DEVICE)

    def test_output_shape(self):
        img = fake_image()
        with torch.no_grad():
            out = self.vision_enc(img)
        # CLIP ViT-B/32 → [B, C, num_patches]
        assert out.dim() == 3, f"Expected 3D output, got shape {out.shape}"
        assert out.size(0) == B, f"Batch dim mismatch: {out.shape}"
        print(f"  CLIP output: {out.shape}")

    def test_frozen(self):
        for name, p in self.vision_enc.named_parameters():
            assert not p.requires_grad, f"CLIP param {name} is NOT frozen"

    def test_no_nan(self):
        img = fake_image()
        with torch.no_grad():
            out = self.vision_enc(img)
        assert not torch.isnan(out).any(), "NaN in CLIP output"
        assert not torch.isinf(out).any(), "Inf in CLIP output"

    def test_different_images_give_different_features(self):
        img1 = fake_image()
        img2 = torch.randn_like(img1)
        with torch.no_grad():
            f1 = self.vision_enc(img1)
            f2 = self.vision_enc(img2)
        assert not torch.allclose(f1, f2), "CLIP returns identical features for different images"


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Text encoder (BERT question encoding)
# ══════════════════════════════════════════════════════════════════════════════
class TestTextEncoder:

    def setup_method(self):
        self.model = build_model()
        self.fuse = self.model.fuse.to(DEVICE)

    def test_word_embedding_shape(self):
        cond = fake_cond()
        q_ids = cond["input_q_id"]
        emb = self.model.word_embedding(q_ids)
        assert emb.shape == (B, Q_LEN, H), f"word_embedding shape wrong: {emb.shape}"

    def test_question_encoding_no_nan(self):
        """Full question encoding through BERT: token+pos+type → LayerNorm → encoder."""
        import torch
        cond = fake_cond()
        q_ids = cond["input_q_id"]
        q_mask = (q_ids != 0).long()

        token_emb  = self.fuse.language_encoder(q_ids)
        pos_ids    = torch.arange(Q_LEN, device=DEVICE).unsqueeze(0)
        pos_emb    = self.fuse.position_embeddings(pos_ids)
        tte        = self.fuse.token_type_embeddings(torch.zeros_like(q_ids))
        emb        = self.fuse.bert_embedding_layer_norm(token_emb + pos_emb + tte)
        emb        = self.fuse.bert_embedding_dropout(emb)

        assert not torch.isnan(emb).any(), "NaN after BERT embedding composition"
        assert emb.shape == (B, Q_LEN, H), f"Embedding shape: {emb.shape}"

    def test_lm_head_weight_tied(self):
        """lm_head.weight must be the same tensor as word_embedding.weight."""
        assert self.model.lm_head.weight.data_ptr() == self.model.word_embedding.weight.data_ptr(), \
            "lm_head.weight is NOT tied to word_embedding.weight — embedding drift will occur"

    def test_bert_encoder_grad(self):
        """BERT encoder must be trainable (not frozen)."""
        trainable = [p for p in self.model.fuse.bert.encoder.parameters() if p.requires_grad]
        assert len(trainable) > 0, "BERT encoder is fully frozen — no gradients will flow"

    def test_different_questions_give_different_features(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        q1 = tok(["what organ is shown?"], padding="max_length", max_length=Q_LEN,
                  truncation=True, return_tensors="pt")["input_ids"].to(DEVICE)
        q2 = tok(["is this image normal?"], padding="max_length", max_length=Q_LEN,
                  truncation=True, return_tensors="pt")["input_ids"].to(DEVICE)
        emb1 = self.model.word_embedding(q1)
        emb2 = self.model.word_embedding(q2)
        assert not torch.allclose(emb1, emb2), "Same embedding for different questions"


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — Feature fusion (cross-modal fusion)
# ══════════════════════════════════════════════════════════════════════════════
class TestFeatureFusion:

    def setup_method(self):
        self.model = build_model()

    def test_output_shape(self):
        img  = fake_image()
        cond = fake_cond()
        cond_for_fuse = {k: v for k, v in cond.items()
                         if k not in ("input_ids", "input_a_id", "input_mask", "image_name")}
        with torch.no_grad():
            fuse_out, pre_ans = self.model.fuse(img, cond_for_fuse)
        # fuse output must be [B, Q_LEN, H]
        assert fuse_out.shape == (B, Q_LEN, H), \
            f"feature_fusion output shape wrong: {fuse_out.shape}, expected ({B}, {Q_LEN}, {H})"
        assert pre_ans.shape == (B, Q_LEN, H), \
            f"pre_simu_answer shape wrong: {pre_ans.shape}"
        print(f"  fuse output: {fuse_out.shape}")

    def test_no_nan_in_fusion(self):
        img  = fake_image()
        cond = fake_cond()
        cond_for_fuse = {k: v for k, v in cond.items()
                         if k not in ("input_ids", "input_a_id", "input_mask", "image_name")}
        with torch.no_grad():
            fuse_out, _ = self.model.fuse(img, cond_for_fuse)
        assert not torch.isnan(fuse_out).any(),  "NaN in feature_fusion output"
        assert not torch.isinf(fuse_out).any(),  "Inf in feature_fusion output"

    def test_image_affects_fusion(self):
        """Different images must produce different fusion outputs."""
        img1  = fake_image()
        img2  = torch.randn_like(img1)
        cond1 = fake_cond()
        cond2 = fake_cond()
        c1 = {k: v for k, v in cond1.items()
              if k not in ("input_ids", "input_a_id", "input_mask", "image_name")}
        c2 = {k: v for k, v in cond2.items()
              if k not in ("input_ids", "input_a_id", "input_mask", "image_name")}
        with torch.no_grad():
            f1, _ = self.model.fuse(img1, c1)
            f2, _ = self.model.fuse(img2, c2)
        assert not torch.allclose(f1, f2, atol=1e-4), \
            "Fusion output identical for different images — image signal not reaching fusion"

    def test_question_affects_fusion(self):
        """Different questions must produce different fusion outputs."""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        img = fake_image()

        q1_ids = tok(["what organ is shown?"] * B, padding="max_length",
                     max_length=Q_LEN, truncation=True, return_tensors="pt")["input_ids"]
        q2_ids = tok(["is this image normal?"] * B, padding="max_length",
                     max_length=Q_LEN, truncation=True, return_tensors="pt")["input_ids"]

        c1 = {"input_q_id": q1_ids.to(DEVICE)}
        c2 = {"input_q_id": q2_ids.to(DEVICE)}

        with torch.no_grad():
            f1, _ = self.model.fuse(img, c1)
            f2, _ = self.model.fuse(img, c2)
        assert not torch.allclose(f1, f2, atol=1e-4), \
            "Fusion output identical for different questions — question signal not reaching fusion"

    def test_fusion_grad_flows_to_trainable_params(self):
        """Gradients must reach alpha/beta/theta and projection layers."""
        img   = fake_image()
        cond  = fake_cond()
        c     = {k: v for k, v in cond.items()
                 if k not in ("input_ids", "input_a_id", "input_mask", "image_name")}
        fuse_out, _ = self.model.fuse(img, c)
        loss = fuse_out.mean()
        loss.backward()

        for name, param in [
            ("alpha", self.model.fuse.alpha),
            ("beta",  self.model.fuse.beta),
            ("theta", self.model.fuse.theta),
        ]:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_alpha_beta_theta_initial_values(self):
        """All three scalar weights should start at 1.0."""
        assert float(self.model.fuse.alpha.detach()) == _approx(1.0), "alpha != 1.0"
        assert float(self.model.fuse.beta.detach())  == _approx(1.0), "beta != 1.0"
        assert float(self.model.fuse.theta.detach()) == _approx(1.0), "theta != 1.0"


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — Concat (fuse + answer → diffusion input)
# ══════════════════════════════════════════════════════════════════════════════
class TestConcatenation:

    def setup_method(self):
        self.model = build_model()

    def test_get_ddpm_input_shape(self):
        img  = fake_image()
        cond = fake_cond()
        c    = {k: v for k, v in cond.items()
                if k not in ("input_ids", "input_a_id", "input_mask", "image_name")}
        with torch.no_grad():
            fuse_feats, pre_ans = self.model.get_ddpm_input(img, c)
        assert fuse_feats.shape == (B, Q_LEN, H), \
            f"get_ddpm_input shape: {fuse_feats.shape}"

    def test_concat_shape(self):
        img  = fake_image()
        cond = fake_cond()
        c    = {k: v for k, v in cond.items()
                if k not in ("input_ids", "input_a_id", "input_mask", "image_name")}
        with torch.no_grad():
            fuse_feats, _ = self.model.get_ddpm_input(img, c)
            ans_emb = self.model.get_embeds(cond["input_a_id"])
        concat = torch.cat([fuse_feats, ans_emb], dim=1)
        assert concat.shape == (B, Q_LEN + A_LEN, H), \
            f"Concat shape wrong: {concat.shape}, expected ({B}, {Q_LEN + A_LEN}, {H})"
        print(f"  fuse_len={Q_LEN}, ans_len={A_LEN}, total={Q_LEN+A_LEN}")

    def test_ans_emb_no_nan(self):
        cond = fake_cond()
        with torch.no_grad():
            ans_emb = self.model.get_embeds(cond["input_a_id"])
        assert not torch.isnan(ans_emb).any(), "NaN in answer embeddings"
        assert ans_emb.shape == (B, A_LEN, H)

    def test_ans_emb_close_to_vocab(self):
        """Answer embeddings must be on the vocab manifold — nn_l2 should be near 0."""
        from diffuvqa.rounding import get_efficient_knn
        cond = fake_cond()
        with torch.no_grad():
            ans_emb = self.model.get_embeds(cond["input_a_id"])
            flat    = ans_emb.reshape(-1, H)
            val, _  = get_efficient_knn(self.model.word_embedding.weight, flat)
        sq_dist = (-val).clamp(min=0.0)
        avg_l2  = sq_dist.sqrt().mean().item()
        print(f"  ans_emb avg_nn_l2 = {avg_l2:.4f}  (should be ~0 — embeddings are vocab rows)")
        assert avg_l2 < 0.01, \
            f"Answer embeddings NOT on vocab manifold: avg_nn_l2={avg_l2:.4f}. " \
            f"lm_head weight tying may be broken."


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — Diffusion backbone (TransformerNetModel.forward)
# ══════════════════════════════════════════════════════════════════════════════
class TestDiffusionBackbone:

    def setup_method(self):
        self.model = build_model()

    def test_forward_shape(self):
        total_len = Q_LEN + A_LEN
        x         = torch.randn(B, total_len, H, device=DEVICE)
        t         = torch.randint(0, 100, (B,), device=DEVICE)
        with torch.no_grad():
            out = self.model(x, t)
        assert out.shape == (B, total_len, H), \
            f"TransformerNetModel output shape: {out.shape}, expected ({B}, {total_len}, {H})"

    def test_no_nan_forward(self):
        total_len = Q_LEN + A_LEN
        x = torch.randn(B, total_len, H, device=DEVICE)
        t = torch.randint(0, 100, (B,), device=DEVICE)
        with torch.no_grad():
            out = self.model(x, t)
        assert not torch.isnan(out).any(), "NaN in diffusion backbone output"
        assert not torch.isinf(out).any(), "Inf in diffusion backbone output"

    def test_timestep_affects_output(self):
        """Different timesteps must produce different denoising outputs."""
        total_len = Q_LEN + A_LEN
        x  = torch.randn(B, total_len, H, device=DEVICE)
        t1 = torch.zeros(B, dtype=torch.long, device=DEVICE)
        t2 = torch.full((B,), 50, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            o1 = self.model(x, t1)
            o2 = self.model(x, t2)
        assert not torch.allclose(o1, o2, atol=1e-4), \
            "Different timesteps give identical output — time embedding has no effect"

    def test_grad_flows_through_backbone(self):
        total_len = Q_LEN + A_LEN
        x = torch.randn(B, total_len, H, device=DEVICE, requires_grad=True)
        t = torch.randint(0, 100, (B,), device=DEVICE)
        out = self.model(x, t)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None, "No gradient flowing through diffusion backbone"
        assert not torch.isnan(x.grad).any(), "NaN gradient in diffusion backbone"


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — get_logits (embedding → vocab scores)
# ══════════════════════════════════════════════════════════════════════════════
class TestGetLogits:

    def setup_method(self):
        self.model = build_model()

    def test_logits_shape(self):
        emb = torch.randn(B, A_LEN, H, device=DEVICE)
        with torch.no_grad():
            logits = self.model.get_logits(emb)
        V = self.model.lm_head.weight.size(0)
        assert logits.shape == (B, A_LEN, V), \
            f"get_logits shape: {logits.shape}, expected ({B}, {A_LEN}, {V})"

    def test_logits_no_nan(self):
        emb = torch.randn(B, A_LEN, H, device=DEVICE)
        with torch.no_grad():
            logits = self.model.get_logits(emb)
        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"

    def test_correct_token_gets_best_logit(self):
        """If we feed the exact embedding of token X, logits must rank X first."""
        token_id = 2023  # arbitrary mid-vocab token ("this")
        emb = self.model.word_embedding(
            torch.tensor([[token_id]] * B, device=DEVICE)
        )  # [B, 1, H]
        with torch.no_grad():
            logits = self.model.get_logits(emb)  # [B, 1, V]
        top1 = logits[:, 0, :].argmax(dim=-1)
        for b in range(B):
            assert top1[b].item() == token_id, \
                f"Exact embedding of token {token_id} ranked #{logits[0,0,token_id].item():.2f} " \
                f"but argmax={top1[b].item()} — logits_mode=2 L2 distance broken or lm_head untied"

    def test_subword_mask_eliminates_hash_tokens(self):
        """After masking, no ## token should appear in top-1."""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        vocab = tok.get_vocab()
        subword_mask = torch.zeros(len(vocab), dtype=torch.bool)
        for t_str, idx in vocab.items():
            if t_str.startswith("##"):
                subword_mask[idx] = True

        emb    = torch.randn(B, A_LEN, H, device=DEVICE)
        with torch.no_grad():
            logits = self.model.get_logits(emb)
        logits_masked = logits.masked_fill(subword_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        top1   = logits_masked.argmax(dim=-1)  # [B, A_LEN]

        for b in range(B):
            for pos in range(A_LEN):
                tid = top1[b, pos].item()
                tok_str = tok.convert_ids_to_tokens([tid])[0]
                assert not tok_str.startswith("##"), \
                    f"## token '{tok_str}' (id={tid}) appeared despite masking at batch={b} pos={pos}"


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — Diffusion loss (training_losses_seq2seq)
# ══════════════════════════════════════════════════════════════════════════════
class TestDiffusionLoss:

    def setup_method(self):
        self.model = build_model()
        self.diffusion = build_diffusion(steps=100)

    def _make_model_kwargs(self):
        cond = fake_cond()
        return {
            "input_ids":   cond["input_ids"],
            "input_q_id":  cond["input_q_id"],
            "input_a_id":  cond["input_a_id"],
            "input_mask":  cond["input_mask"],
            "image_name":  cond["image_name"],
        }

    def test_loss_keys_present(self):
        img    = fake_image()
        kwargs = self._make_model_kwargs()
        t      = torch.randint(0, 100, (B,), device=DEVICE)
        terms  = self.diffusion.training_losses_seq2seq(
            self.model, img, t, model_kwargs=kwargs)
        for key in ("loss", "mse", "nll"):
            assert key in terms, f"Missing loss key: {key}"

    def test_loss_shape(self):
        img    = fake_image()
        kwargs = self._make_model_kwargs()
        t      = torch.randint(0, 100, (B,), device=DEVICE)
        terms  = self.diffusion.training_losses_seq2seq(
            self.model, img, t, model_kwargs=kwargs)
        assert terms["loss"].shape == (B,), \
            f"loss shape: {terms['loss'].shape}, expected ({B},)"

    def test_loss_no_nan(self):
        img    = fake_image()
        kwargs = self._make_model_kwargs()
        t      = torch.randint(0, 100, (B,), device=DEVICE)
        terms  = self.diffusion.training_losses_seq2seq(
            self.model, img, t, model_kwargs=kwargs)
        for k, v in terms.items():
            assert not torch.isnan(v).any(), f"NaN in loss term '{k}'"
            assert not torch.isinf(v).any(), f"Inf in loss term '{k}'"

    def test_loss_positive(self):
        img    = fake_image()
        kwargs = self._make_model_kwargs()
        t      = torch.randint(1, 100, (B,), device=DEVICE)
        terms  = self.diffusion.training_losses_seq2seq(
            self.model, img, t, model_kwargs=kwargs)
        assert (terms["loss"] >= 0).all(), f"Negative loss values: {terms['loss']}"

    def test_loss_backward(self):
        """Backprop through the full training loss must not crash."""
        img    = fake_image()
        kwargs = self._make_model_kwargs()
        t      = torch.randint(0, 100, (B,), device=DEVICE)
        terms  = self.diffusion.training_losses_seq2seq(
            self.model, img, t, model_kwargs=kwargs)
        terms["loss"].mean().backward()
        # Check at least one parameter received a gradient
        grads = [p.grad for p in self.model.parameters()
                 if p.grad is not None and p.requires_grad]
        assert len(grads) > 0, "No gradients after loss.backward()"

    def test_decoder_nll_included_in_loss(self):
        """decoder_nll must contribute to loss — loss > nll alone."""
        img    = fake_image()
        kwargs = self._make_model_kwargs()
        t      = torch.randint(1, 100, (B,), device=DEVICE)
        terms  = self.diffusion.training_losses_seq2seq(
            self.model, img, t, model_kwargs=kwargs)
        # loss = mse + nll + decoder_nll, so loss >= nll
        assert (terms["loss"] >= terms["nll"] - 1e-4).all(), \
            "loss < nll — decoder_nll may be missing or negative"

    def test_padding_mask_reduces_loss_vs_unmasked(self):
        """MSE loss on padding-only answer should be zero (all PAD tokens)."""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        # All-PAD answer
        pad_ids = torch.zeros(B, A_LEN, dtype=torch.long, device=DEVICE)
        mask    = torch.zeros(B, A_LEN, dtype=torch.long, device=DEVICE)
        img     = fake_image()
        q_ids   = tok(["q"] * B, padding="max_length", max_length=Q_LEN,
                      truncation=True, return_tensors="pt")["input_ids"].to(DEVICE)
        kwargs  = {
            "input_ids":  q_ids,
            "input_q_id": q_ids,
            "input_a_id": pad_ids,
            "input_mask": mask,
            "image_name": ["img.jpg"] * B,
        }
        t     = torch.randint(1, 100, (B,), device=DEVICE)
        terms = self.diffusion.training_losses_seq2seq(
            self.model, img, t, model_kwargs=kwargs)
        print(f"  MSE with all-PAD answer: {terms['mse'].mean().item():.4f}  (should be 0.0)")
        assert (terms["mse"].abs() < 1e-4).all(), \
            f"MSE non-zero for all-PAD answer — padding mask may be broken: {terms['mse']}"


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 8 — End-to-end gradient flow
# ══════════════════════════════════════════════════════════════════════════════
class TestEndToEndGradientFlow:

    def setup_method(self):
        self.model     = build_model()
        self.diffusion = build_diffusion(steps=100)

    def test_all_trainable_params_receive_grad(self):
        img    = fake_image()
        cond   = fake_cond()
        kwargs = {
            "input_ids":  cond["input_ids"],
            "input_q_id": cond["input_q_id"],
            "input_a_id": cond["input_a_id"],
            "input_mask": cond["input_mask"],
            "image_name": cond["image_name"],
        }
        t     = torch.randint(0, 100, (B,), device=DEVICE)
        terms = self.diffusion.training_losses_seq2seq(
            self.model, img, t, model_kwargs=kwargs)
        terms["loss"].mean().backward()

        no_grad, has_nan_grad = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                no_grad.append(name)
            elif torch.isnan(param.grad).any():
                has_nan_grad.append(name)

        if no_grad:
            print(f"\n  [WARN] Params with no gradient ({len(no_grad)}):")
            for n in no_grad[:10]:
                print(f"    - {n}")
        if has_nan_grad:
            print(f"\n  [WARN] Params with NaN gradient ({len(has_nan_grad)}):")
            for n in has_nan_grad[:10]:
                print(f"    - {n}")

        assert len(has_nan_grad) == 0, \
            f"NaN gradients detected in: {has_nan_grad[:5]}"
        # Allow some params with None grad (e.g., unused heads) but not too many
        trainable_count = sum(1 for p in self.model.parameters() if p.requires_grad)
        no_grad_frac = len(no_grad) / max(trainable_count, 1)
        assert no_grad_frac < 0.5, \
            f"More than 50% of trainable params have no gradient ({len(no_grad)}/{trainable_count})"

    def test_clip_params_have_no_grad(self):
        img    = fake_image()
        cond   = fake_cond()
        kwargs = {
            "input_ids":  cond["input_ids"],
            "input_q_id": cond["input_q_id"],
            "input_a_id": cond["input_a_id"],
            "input_mask": cond["input_mask"],
            "image_name": cond["image_name"],
        }
        t     = torch.randint(0, 100, (B,), device=DEVICE)
        terms = self.diffusion.training_losses_seq2seq(
            self.model, img, t, model_kwargs=kwargs)
        terms["loss"].mean().backward()

        for name, param in self.model.fuse.vision_encoder.named_parameters():
            assert param.grad is None or (param.grad == 0).all(), \
                f"CLIP param {name} received a gradient — freeze broken"


# ══════════════════════════════════════════════════════════════════════════════
# Standalone runner
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import traceback

    test_classes = [
        TestVisionEncoder,
        TestTextEncoder,
        TestFeatureFusion,
        TestConcatenation,
        TestDiffusionBackbone,
        TestGetLogits,
        TestDiffusionLoss,
        TestEndToEndGradientFlow,
    ]

    passed = failed = 0
    results = []

    for cls in test_classes:
        print(f"\n{'='*60}")
        print(f"  {cls.__name__}")
        print(f"{'='*60}")
        instance = cls()
        methods  = [m for m in dir(instance) if m.startswith("test_")]
        for method in methods:
            instance.setup_method()
            try:
                getattr(instance, method)()
                print(f"  ✓ {method}")
                passed += 1
                results.append((cls.__name__, method, "PASS", None))
            except Exception as e:
                print(f"  ✗ {method}")
                print(f"      {type(e).__name__}: {e}")
                failed += 1
                results.append((cls.__name__, method, "FAIL", str(e)))

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed:
        print("\nFailed tests:")
        for cls_name, method, status, err in results:
            if status == "FAIL":
                print(f"  [{cls_name}] {method}")
                print(f"    → {err}")
        sys.exit(1)
