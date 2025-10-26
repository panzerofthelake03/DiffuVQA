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

        # Use the configured hidden_dim as the attention feature size so
        # attention modules operate in the same latent space as the
        # projections (this prevents matmul shape mismatches when
        # args.hidden_dim != pretrained BERT hidden size).
        assert args.hidden_dim % args.num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.multi_attention = MultiHeadedAttention(args.num_heads, args.hidden_dim)
        self.cross_attention = cross_attention(args.hidden_dim, head=args.num_heads)
        self.multi_attention.apply(self.init_weights)
        self.cross_attention.apply(self.init_weights) 

        self.language_encoder = language_encoder
        self.bert = bert
        self.modality_type_embeddings = nn.Embedding(2, args.hidden_dim)
        self.modality_type_embeddings.apply(self.init_weights)
        self.vision_encoder = build_model(args.image_encoder, resolution_after=args.image_resolution)

        # Project raw vision encoder channel features to the configured image embedding size.
        # Historically this code used 145 as the number of channels produced by
        # the pretrained CLIP/resnet visual backbone; keep that here for
        # compatibility with pre-trained weights. If you use a different visual
        # backbone adjust this number accordingly.
        self.image_MLP = nn.Linear(145, args.input_image_embed_size)
        self.image_MLP.apply(self.init_weights)

        # modality type embeddings (0=text,1=image)
        self.modality_type_embeddings = nn.Embedding(2, args.hidden_dim)
        self.modality_type_embeddings.apply(self.init_weights)

        ###########  feature proj ###############
        self.image_feature_proj = nn.Sequential(
            nn.Linear(args.input_image_embed_size, args.extend_hidden_size),
            nn.GELU(),
            nn.Linear(args.extend_hidden_size, args.hidden_dim),
        )
        self.image_feature_proj.apply(self.init_weights)

        # The language encoder (BERT) may have a different hidden size than
        # the configured `args.hidden_dim`. Use the encoder's hidden size as
        # the input dimension for the first projection so the code works when
        # args.hidden_dim != bert_hidden_size (e.g., using a smaller model
        # latent while still loading BERT-base as the language encoder).
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
        # vision_encoder -> [B, C, L]; transpose to [B, L, C] so Linear
        # operates on the channel/features dimension (C).
        image_feats = image_feats.transpose(1, 2)
        # Apply MLP to per-patch/channel features: result [B, L, input_image_embed_size]
        image_feats = self.image_MLP(image_feats)
        # Project per-patch embeddings to the hidden_dim: still [B, L, hidden_dim]
        image_feats = self.image_feature_proj(image_feats)
        image_masks = torch.ones((image_feats.size(0), image_feats.size(1)), dtype=torch.long,
                                 device=image_feats.device)

        # Add modality type embeddings
        question_feats = question_feats + self.modality_type_embeddings(torch.zeros_like(q_mask))
        image_feats = image_feats + self.modality_type_embeddings(torch.full_like(image_masks, 1))

        # If sequence lengths differ, pool image features and expand them to
        # match the question token length. We want `pre_simu_answer_feats` to
        # have the same length as the token-level question/answer embeddings
        # so it can be compared to `ans_emb` later during loss computation.
        if question_feats.size(1) != image_feats.size(1):
            img_pooled = image_feats.mean(dim=1, keepdim=True)  # [B,1,H]
            img_for_text = img_pooled.expand(-1, question_feats.size(1), -1)  # [B,QT,H]
        else:
            img_for_text = image_feats

        # Fuse question token features with the image summary per-token
        fused_for_cvae = question_feats + img_for_text
        pre_simu_answer_feats = self.cvae(fused_for_cvae)

        # Create attention masks
        q_mask = cond['input_mask']  # question mask (batch, q_len) 
        img_mask = torch.ones((image_feats.size(0), image_feats.size(1)), device=image_feats.device)
        
        # Make masks for cross attention (batch, seq_q, seq_k)
        qa_mask = q_mask.unsqueeze(1) & q_mask.unsqueeze(2)  # for answer->question
        qi_mask = q_mask.unsqueeze(1) & img_mask.unsqueeze(2)  # for fused->image

        # Apply masked cross attention
        f1 = self.cross_attention(pre_simu_answer_feats, question_feats, question_feats, mask=qa_mask)
        f2 = self.cross_attention(f1, image_feats, image_feats, mask=qi_mask)
        f3 = self.multi_attention(f2, f2, f2)
        f3 = self.layer_norm(f3)
        f4 = self.feature_proj(f3)

        # Ensure q_for_image is defined and all tensors have the same sequence length
        # by pooling-and-expanding the shorter sequences to match image_feats.
        if 'q_for_image' not in locals():
            # Create q_for_image by pooling question tokens and expanding to image length
            if question_feats.size(1) != image_feats.size(1):
                q_for_image = question_feats.mean(dim=1, keepdim=True).expand(-1, image_feats.size(1), -1)
            else:
                q_for_image = question_feats

        # If f4 has a different sequence length, pool-and-expand it as well so
        # the final element-wise combination works without size mismatch.
        if f4.size(1) != image_feats.size(1):
            f4 = f4.mean(dim=1, keepdim=True).expand(-1, image_feats.size(1), -1)

        # Debug print (optional) controlled by environment variable DVQA_DEBUG
        try:
            if os.environ.get('DVQA_DEBUG', '0') == '1':
                print(f"DEBUG feature_fusion shapes: f4 {tuple(f4.shape)}, image_feats {tuple(image_feats.shape)}, q_for_image {tuple(q_for_image.shape)}")
        except Exception:
            pass

        f = self.alpha * f4 + self.beta * image_feats + self.theta * q_for_image
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
            # If the pretrained token embeddings size differs from the
            # configured hidden_dim, add a small projection to map them
            # into the model latent space used by the rest of the network.
            try:
                bert_emb_dim = self.word_embedding.weight.size(1)
                # Project token embeddings into the transformer's hidden size
                # (config.hidden_size) so downstream components see the same
                # feature dimensionality.
                target_dim = self.hidden_size
                if bert_emb_dim != target_dim:
                    self.word_embedding_proj = nn.Linear(bert_emb_dim, target_dim)
                    # initialize similarly to other linears
                    self.word_embedding_proj.weight.data.normal_(mean=0.0, std=0.02)
                    if self.word_embedding_proj.bias is not None:
                        self.word_embedding_proj.bias.data.zero_()
            except Exception:
                pass
            self.fuse = feature_fusion(self.word_embedding, temp_bert, args)

            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            # self.lm_head.weight.requires_grad = False
            # self.word_embedding.weight.requires_grad = False

            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            # Project pretrained position embeddings to args.hidden_dim if needed
            try:
                pos_dim = self.position_embeddings.weight.size(1)
                # Project positional embeddings into the transformer's hidden
                # size so they can be added to other transformer inputs.
                target_dim = self.hidden_size
                if pos_dim != target_dim:
                    self.position_embeddings_proj = nn.Linear(pos_dim, target_dim)
                    self.position_embeddings_proj.weight.data.normal_(mean=0.0, std=0.02)
                    if self.position_embeddings_proj.bias is not None:
                        self.position_embeddings_proj.bias.data.zero_()
            except Exception:
                pass

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

        # If the fusion module produces features in a different latent
        # dimensionality (args.hidden_dim) than the transformer's
        # internal hidden size (config.hidden_size), add a projection to
        # map fusion outputs into the transformer's expected feature size.
        try:
            target_dim = self.hidden_size
            fuse_dim = args.hidden_dim if args is not None else target_dim
            if fuse_dim != target_dim:
                self.fuse_output_proj = nn.Linear(fuse_dim, target_dim)
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
        # If necessary, project fusion outputs into the transformer's
        # hidden size so downstream transformer layers receive the
        # expected dimensionality.
        if hasattr(self, 'fuse_output_proj'):
            ddpm_input = self.fuse_output_proj(ddpm_input)
            ans_emb = self.fuse_output_proj(ans_emb)
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

        # Positional embeddings may have been trained for a shorter max length
        # (e.g. 512). If the current sequence length is larger (because we
        # concatenated image patches and text tokens), tile/repeat the
        # positional embedding weights to cover the requested length.
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
            # Tile the learned position embeddings to the required length and
            # take the first seq_length entries.
            pos_weight = pos_emb_module.weight.data
            repeats = (seq_length + pos_len - 1) // pos_len
            expanded = pos_weight.repeat(repeats, 1)[:seq_length].unsqueeze(0)
            pos_emb = expanded.to(emb_x.device)
            if hasattr(self, 'position_embeddings_proj'):
                pos_emb = self.position_embeddings_proj(pos_emb)

        # print(emb_x.shape, emb_t.shape, self.position_embeddings)
        emb_inputs = pos_emb + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state

        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h
