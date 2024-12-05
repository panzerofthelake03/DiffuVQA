import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)

class cross_attention(nn.Module):
    def __init__(self, feature_size, head=8):
        super().__init__()
        self.cross_attn = MultiHeadedAttention(h=head, d_model=feature_size)
        self.layer_norm = LayerNorm(feature_size)

    def forward(self, q, k, v):
        return self.layer_norm(q + self.cross_attn(q, k, v))


class cross_attention_without_residual(nn.Module):
    def __init__(self, feature_size, head=8):
        super().__init__()
        self.cross_attn = MultiHeadedAttention(h=head, d_model=feature_size)
        self.layer_norm = LayerNorm(feature_size)

    def forward(self, q, k, v):
        return self.layer_norm(self.cross_attn(q, k, v))
    

if __name__ == '__main__':

    d_model = 768
    num_heads = 4
    dropout_rate = 0.1

    multi_head_attn = MultiHeadedAttention(h=num_heads, d_model=d_model, dropout=dropout_rate)
    cross_attn = cross_attention(d_model, num_heads)

    batch_size = 64
    seq_length = 20

    query = torch.rand(batch_size, seq_length, d_model)
    key = torch.rand(batch_size, seq_length, d_model)
    value = torch.rand(batch_size, seq_length, d_model)

    mask = torch.randint(0, 2, (batch_size, 1, 1, seq_length))

    output_1 = multi_head_attn(query, key, value, mask=mask)
    output_2 = cross_attn(query, key, value)

    print("Output shape:", output_1.shape)  
    print("Output shape:", output_2.shape)
