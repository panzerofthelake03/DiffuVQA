from transformers import AutoConfig
from transformers import RobertaConfig, RobertaModel
from transformers.models.bert.modeling_bert import BertConfig, BertModel, BertEncoder

import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from diffuvqa.attention.attention_model import MultiHeadedAttention, cross_attention
from diffuvqa.vision_encoders.clip_model import build_model
from diffuvqa.language_encoders.bert_model import BertCrossLayer, answer_fusion_module, pre_training_module
from diffuvqa.utils.nn import SiLU, linear, timestep_embedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from diffuvqa.utils.answer_pre import find_most_similar_answers


class CVAE(nn.Module):
    def __init__(self, embedding_dim):
        super(CVAE, self).__init__()
        self.process_f = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )

    def forward(self, x):
        return self.process_f(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class feature_fusion(nn.Module):
    """
    Fuses image and question features into a conditioning vector for the diffusion model.

    Pipeline:
      1. Question: BERT full embedding (token + position + token_type + LayerNorm + dropout) → encoder layers → project
      2. Image: CLIP visual encoder (frozen) → transpose → MLP → project
      3. Fusion: CVAE pre-answer simulation → cross-attention (q↔q, f↔img) → multi-head self-attention → layer norm → proj
      4. Output: alpha*f4 + beta*image + theta*(question_feats + question_emb)  [shape: B, seq_len, hidden_dim]
    """
    def __init__(self, language_encoder, bert, args):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.theta = nn.Parameter(torch.tensor(1.0))

        self.multi_attention = MultiHeadedAttention(args.num_heads, args.hidden_dim)
        self.cross_attention = cross_attention(args.hidden_dim)
        self.multi_attention.apply(self.init_weights)
        self.cross_attention.apply(self.init_weights)

        self.language_encoder = language_encoder
        self.bert = bert
        # Cache embedding submodules — parent deletes temp_bert.embeddings after init.
        self.position_embeddings = bert.embeddings.position_embeddings
        self.token_type_embeddings = bert.embeddings.token_type_embeddings
        self.bert_embedding_layer_norm = bert.embeddings.LayerNorm
        self.bert_embedding_dropout = bert.embeddings.dropout

        self.modality_type_embeddings = nn.Embedding(2, args.hidden_dim)
        self.modality_type_embeddings.apply(self.init_weights)

        self.vision_encoder = build_model(args.image_encoder, resolution_after=args.image_resolution)
        for p in self.vision_encoder.parameters():
            p.requires_grad_(False)

        # Detect vision encoder output channels via dummy forward to avoid hardcoding.
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, args.image_resolution, args.image_resolution)
            _vision_channels = self.vision_encoder(_dummy).shape[1]

        self.image_MLP = nn.Linear(_vision_channels, args.input_image_embed_size)
        self.image_MLP.apply(self.init_weights)

        self.image_feature_proj = nn.Sequential(
            nn.Linear(args.input_image_embed_size, args.extend_hidden_size),
            nn.GELU(),
            nn.Linear(args.extend_hidden_size, args.hidden_dim),
        )
        self.image_feature_proj.apply(self.init_weights)

        bert_hidden_size = getattr(self.bert.config, 'hidden_size', args.hidden_dim)
        self.question_feature_proj = nn.Sequential(
            nn.Linear(bert_hidden_size, args.extend_hidden_size),
            nn.GELU(),
            nn.Linear(args.extend_hidden_size, args.hidden_dim),
        )
        self.question_feature_proj.apply(self.init_weights)

        self.feature_proj = nn.Sequential(
            nn.Linear(args.hidden_dim, args.extend_hidden_size),
            nn.GELU(),
            nn.Linear(args.extend_hidden_size, args.hidden_dim)
        )
        self.feature_proj.apply(self.init_weights)

        self.layer_norm = LayerNorm(args.hidden_dim)
        self.layer_norm.apply(self.init_weights)

        self.cvae = CVAE(args.hidden_dim)
        self.cvae.apply(self.init_weights)

    def forward(self, image, cond):
        # --- Question encoding ---
        q_ids = cond.pop('input_q_id')
        q_mask = (q_ids != 0).long().to(q_ids.device)

        seq_len = q_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=q_ids.device).unsqueeze(0)
        token_type_ids = torch.zeros_like(q_ids)
        token_emb = self.language_encoder(q_ids)
        position_emb = self.position_embeddings(position_ids)
        token_type_emb = self.token_type_embeddings(token_type_ids)
        question_emb = self.bert_embedding_layer_norm(token_emb + position_emb + token_type_emb)
        question_emb = self.bert_embedding_dropout(question_emb)

        encoder_out = self.bert.encoder(
            question_emb,
            attention_mask=q_mask,
            output_hidden_states=False,
        )
        question_feats = encoder_out.last_hidden_state
        question_feats = self.question_feature_proj(question_feats)

        # --- Image encoding ---
        image_feats = self.vision_encoder(image)
        image_feats = image_feats.transpose(1, 2)       # [B, C, L] → [B, L, C]
        image_feats = self.image_MLP(image_feats)
        image_feats = self.image_feature_proj(image_feats)
        image_masks = torch.ones((image_feats.size(0), image_feats.size(1)),
                                 dtype=torch.long, device=image_feats.device)

        question_feats = question_feats + self.modality_type_embeddings(torch.zeros_like(q_mask))
        image_feats = image_feats + self.modality_type_embeddings(torch.full_like(image_masks, 1))

        # Pool image to question length for CVAE input
        if question_feats.size(1) != image_feats.size(1):
            img_for_text = image_feats.mean(dim=1, keepdim=True).expand(-1, question_feats.size(1), -1)
        else:
            img_for_text = image_feats

        pre_simu_answer_feats = self.cvae(question_feats + img_for_text)

        # --- Cross/self-attention fusion ---
        f1 = self.cross_attention(pre_simu_answer_feats, question_feats, question_feats)
        f2 = self.cross_attention(f1, image_feats, image_feats)
        f3 = self.multi_attention(f2, f2, f2)
        f3 = self.layer_norm(f3)
        f4 = self.feature_proj(f3)

        # Align all tensors to question token length (seq_len)
        target_len = question_feats.size(1)
        if image_feats.size(1) != target_len:
            image_feats = image_feats.mean(dim=1, keepdim=True).expand(-1, target_len, -1)
        if f4.size(1) != target_len:
            f4 = f4.mean(dim=1, keepdim=True).expand(-1, target_len, -1)
        if question_emb.size(1) != target_len:
            question_emb = question_emb.mean(dim=1, keepdim=True).expand(-1, target_len, -1)

        assert f4.size(1) == image_feats.size(1) == question_feats.size(1) == target_len, \
            f"feature_fusion length mismatch: f4={f4.size(1)}, image={image_feats.size(1)}, q={question_feats.size(1)}"

        # Weighted sum: attended features + image summary + (cross-attended question + raw token embeddings)
        f = self.alpha * f4 + self.beta * image_feats + self.theta * (question_feats + question_emb)
        return f, pre_simu_answer_feats

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class TransformerNetModel(nn.Module):
    """
    Denoising transformer for diffusion-based text generation.

    Accepts noisy answer embeddings concatenated with fusion conditioning [fuse | ans_noise]
    and predicts the clean answer embedding x_0.

    logits_mode=1: dot-product logits (lm_head linear)
    logits_mode=2: L2-distance logits — consistent with get_efficient_knn used at inference
    """

    def __init__(
            self,
            input_dims,
            output_dims,
            hidden_t_dim,
            dropout=0,
            config=None,
            config_name='bert-base-uncased',
            vocab_size=None,
            init_pretrained='no',
            logits_mode=1,
            args=None,
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout
        self.args = args
        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        self.lm_head = nn.Linear(output_dims, vocab_size, bias=False)
        # Initial tie — valid only when dims match; re-tied after pretrained load below.
        if output_dims == self.input_dims:
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(
                nn.Linear(input_dims, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )

        if init_pretrained == 'bert':
            print('initializing from pretrained bert...')
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            self.word_embedding = temp_bert.embeddings.word_embeddings
            try:
                bert_emb_dim = self.word_embedding.weight.size(1)
                if bert_emb_dim != self.hidden_size:
                    self.word_embedding_proj = nn.Linear(bert_emb_dim, self.hidden_size)
                    self.word_embedding_proj.weight.data.normal_(mean=0.0, std=0.02)
                    if self.word_embedding_proj.bias is not None:
                        self.word_embedding_proj.bias.data.zero_()
            except Exception:
                pass
            self.fuse = feature_fusion(self.word_embedding, temp_bert, args)
            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm
            try:
                pos_dim = self.position_embeddings.weight.size(1)
                if pos_dim != self.hidden_size:
                    self.position_embeddings_proj = nn.Linear(pos_dim, self.hidden_size)
                    self.position_embeddings_proj.weight.data.normal_(mean=0.0, std=0.02)
                    if self.position_embeddings_proj.bias is not None:
                        self.position_embeddings_proj.bias.data.zero_()
            except Exception:
                pass
            # Re-tie after word_embedding is replaced with pretrained weights.
            self.lm_head.weight = self.word_embedding.weight
            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'pubmedbert':
            print('initializing from pretrained PubMedBERT (medical domain)...')
            temp_bert = BertModel.from_pretrained('NeuML/pubmedbert-base-embeddings', config=config)
            self.word_embedding = temp_bert.embeddings.word_embeddings
            try:
                bert_emb_dim = self.word_embedding.weight.size(1)
                if bert_emb_dim != self.hidden_size:
                    self.word_embedding_proj = nn.Linear(bert_emb_dim, self.hidden_size)
                    self.word_embedding_proj.weight.data.normal_(mean=0.0, std=0.02)
                    if self.word_embedding_proj.bias is not None:
                        self.word_embedding_proj.bias.data.zero_()
            except Exception:
                pass
            self.fuse = feature_fusion(self.word_embedding, temp_bert, args)
            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm
            try:
                pos_dim = self.position_embeddings.weight.size(1)
                if pos_dim != self.hidden_size:
                    self.position_embeddings_proj = nn.Linear(pos_dim, self.hidden_size)
                    self.position_embeddings_proj.weight.data.normal_(mean=0.0, std=0.02)
                    if self.position_embeddings_proj.bias is not None:
                        self.position_embeddings_proj.bias.data.zero_()
            except Exception:
                pass
            self.lm_head.weight = self.word_embedding.weight
            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'roberta':
            print('initializing from pretrained RoBERTa...')
            temp_roberta = RobertaModel.from_pretrained(config_name, config=config)
            self.word_embedding = temp_roberta.embeddings.word_embeddings
            try:
                roberta_emb_dim = self.word_embedding.weight.size(1)
                if roberta_emb_dim != self.hidden_size:
                    self.word_embedding_proj = nn.Linear(roberta_emb_dim, self.hidden_size)
                    self.word_embedding_proj.weight.data.normal_(mean=0.0, std=0.02)
                    if self.word_embedding_proj.bias is not None:
                        self.word_embedding_proj.bias.data.zero_()
            except Exception:
                pass
            self.fuse = feature_fusion(self.word_embedding, temp_roberta, args)
            self.input_transformers = temp_roberta.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_roberta.embeddings.position_embeddings
            self.LayerNorm = temp_roberta.embeddings.LayerNorm
            try:
                pos_dim = self.position_embeddings.weight.size(1)
                if pos_dim != self.hidden_size:
                    self.position_embeddings_proj = nn.Linear(pos_dim, self.hidden_size)
                    self.position_embeddings_proj.weight.data.normal_(mean=0.0, std=0.02)
                    if self.position_embeddings_proj.bias is not None:
                        self.position_embeddings_proj.bias.data.zero_()
            except Exception:
                pass
            self.lm_head.weight = self.word_embedding.weight
            del temp_roberta.embeddings
            del temp_roberta.pooler

        elif init_pretrained == 'no':
            self.input_transformers = BertEncoder(config)
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        else:
            assert False, "invalid type of init_pretrained"

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, self.output_dims)
            )

        # Project fusion output to transformer hidden size if dimensions differ.
        try:
            fuse_dim = args.hidden_dim if args is not None else self.hidden_size
            if fuse_dim != self.hidden_size:
                self.fuse_output_proj = nn.Linear(fuse_dim, self.hidden_size)
                self.fuse_output_proj.weight.data.normal_(mean=0.0, std=0.02)
                if self.fuse_output_proj.bias is not None:
                    self.fuse_output_proj.bias.data.zero_()
        except Exception:
            pass

    def get_embeds(self, input_ids):
        emb = self.word_embedding(input_ids)
        if hasattr(self, 'word_embedding_proj'):
            emb = self.word_embedding_proj(emb)
        return emb

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            # L2-distance logits — same metric as get_efficient_knn at inference.
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).reshape(-1, 1)
            text_emb_t = th.transpose(text_emb.reshape(-1, text_emb.size(-1)), 0, 1)
            arr_norm = (text_emb ** 2).sum(-1).reshape(-1, 1)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight, text_emb_t)
            scores = th.sqrt(th.clamp(dist, 1e-12, np.inf)).view(
                emb_norm.size(0), hidden_repr.size(0), hidden_repr.size(1))
            return -scores.permute(1, 2, 0).contiguous()
        else:
            raise NotImplementedError

    def get_ddpm_input(self, image, cond):
        ddpm_input, ans_emb = self.fuse(image, cond)
        if hasattr(self, 'fuse_output_proj'):
            ddpm_input = self.fuse_output_proj(ddpm_input)
            ans_emb = self.fuse_output_proj(ans_emb)
        return ddpm_input, ans_emb

    def forward(self, x, timesteps):
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, :seq_length]

        pos_emb_module = self.position_embeddings
        try:
            pos_len = pos_emb_module.weight.size(0)
        except Exception:
            pos_len = position_ids.size(1)

        if seq_length <= pos_len:
            pos_emb = pos_emb_module(position_ids)
            if hasattr(self, 'position_embeddings_proj'):
                pos_emb = self.position_embeddings_proj(pos_emb)
        else:
            # Tile positional embeddings when seq_length exceeds the trained max length.
            pos_weight = pos_emb_module.weight.data
            repeats = (seq_length + pos_len - 1) // pos_len
            expanded = pos_weight.repeat(repeats, 1)[:seq_length].unsqueeze(0)
            pos_emb = expanded.to(emb_x.device)
            if hasattr(self, 'position_embeddings_proj'):
                pos_emb = self.position_embeddings_proj(pos_emb)

        emb_inputs = pos_emb + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state

        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        return h.type(x.dtype)
