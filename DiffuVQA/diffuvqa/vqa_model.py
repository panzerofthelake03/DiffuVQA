from transformers import AutoConfig
# from transformers import BertEncoder
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


##############################################################
###########  vision and text feature fusion     #############
#############################################################
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CVAE(nn.Module):
    def __init__(self, embedding_dim):
        super(CVAE, self).__init__()
        self.process_f = nn.Sequential(nn.Linear(embedding_dim, embedding_dim // 2),
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
        mean = x.mean(-1, keepdim=True)  # bs,98,512 => 16,98,1
        std = x.std(-1, keepdim=True)  # # bs,98,512 => 16,98,1
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class feature_fusion(nn.Module):
    def __init__(self, language_encoder, bert, args):
        super().__init__()

        # self.alpha = torch.tensor(0.1).to("cuda")
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.theta = nn.Parameter(torch.tensor(1.0))
        ###########  load vision and language encoder ###############

        self.multi_attention = MultiHeadedAttention(args.num_heads, args.d_model)
        self.cross_attention = cross_attention(args.feature_size)
        self.multi_attention.apply(self.init_weights)
        self.cross_attention.apply(self.init_weights)

        self.language_encoder = language_encoder
        self.bert = bert
        self.vision_encoder = build_model(args.image_encoder, resolution_after=args.image_resolution)

        self.image_MLP = nn.Linear(145, 32)
        self.image_MLP.apply(self.init_weights)

        self.modality_type_embeddings = nn.Embedding(2, args.hidden_dim)
        self.modality_type_embeddings.apply(self.init_weights)

        ###########  feature proj ###############
        self.image_feature_proj = nn.Sequential(
            nn.Linear(args.input_image_embed_size, args.extend_hidden_size),
            nn.GELU(),
            nn.Linear(args.extend_hidden_size, args.hidden_dim),
        )
        self.image_feature_proj.apply(self.init_weights)

        self.question_feature_proj = nn.Sequential(
            nn.Linear(args.hidden_dim, args.extend_hidden_size),
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
        # == Text Encoding ==
        q_ids = cond.pop('input_q_id')
        q_mask = (q_ids != 0).long().to(q_ids.device)
        q_input_shape = q_mask.size()
        question_emb = self.language_encoder(q_ids)
        extended_q_masks = self.bert.get_extended_attention_mask(q_mask, q_input_shape, device=q_ids.device)
        for layer in self.bert.encoder.layer:
            question_feats = layer(question_emb, extended_q_masks)[0]
        question_feats = self.question_feature_proj(question_feats)  # B 32 768

        # print("q:", question_feats.shape)

        # question_feats = self.question_feature_proj(question_feats)

        # == Image Encoding ==
        image_feats = self.vision_encoder(image)
        image_feats = image_feats.transpose(1, 2)
        image_feats = self.image_MLP(image_feats).transpose(1, 2)
        image_feats = self.image_feature_proj(image_feats)
        image_masks = torch.ones((image_feats.size(0), image_feats.size(1)), dtype=torch.long,
                                 device=image_feats.device)

        question_feats, image_feats = (
            question_feats + self.modality_type_embeddings(torch.zeros_like(q_mask)),
            image_feats + self.modality_type_embeddings(torch.full_like(image_masks, 1)),
        )

        pre_simu_answer_feats = self.cvae(question_emb + image_feats)

        f1 = self.cross_attention(pre_simu_answer_feats, question_feats, question_feats)
        f2 = self.cross_attention(f1, image_feats, image_feats)
        f3 = self.multi_attention(f2, f2, f2)
        f3 = self.layer_norm(f3)
        f4 = self.feature_proj(f3)

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
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
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
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                               nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))

        if init_pretrained == 'bert':
            print('initializing from pretrained bert...')
            print(config)

            bert_config = BertConfig(
                vocab_size=args.vocab_size,
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_layers,
                num_attention_heads=args.num_heads,
                intermediate_size=args.hidden_size * args.mlp_ratio,
                max_position_embeddings=args.seq_len,
                hidden_dropout_prob=args.dropout,
                attention_probs_dropout_prob=args.dropout,
            )

            temp_bert = BertModel.from_pretrained(config_name, config=config)

            self.word_embedding = temp_bert.embeddings.word_embeddings
            self.fuse = feature_fusion(self.word_embedding, temp_bert, args)

            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            # self.lm_head.weight.requires_grad = False
            # self.word_embedding.weight.requires_grad = False

            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no':
            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        else:
            assert False, "invalid type of init_pretrained"

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                  nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:  # standard cosine similarity
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1))  # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def get_ddpm_input(self, image, cond):
        ddpm_input, ans_emb = self.fuse(image, cond)
        return ddpm_input, ans_emb

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """

        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]
        # print(emb_x.shape, emb_t.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state

        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h
