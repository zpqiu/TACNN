import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None, softmax=True):
        a_norm = query / (query.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        b_norm = key / (key.norm(p=2, dim=-1, keepdim=True) + 1e-13)

        scores = torch.bmm(a_norm, b_norm.transpose(-1, -2))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        if softmax:
            p_attn = F.softmax(scores, dim=-1)
        else:
            p_attn = scores

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
