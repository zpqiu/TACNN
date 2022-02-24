# encoding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention.single import Attention


class AttendLayer(nn.Module):
    """detecting difficulty attention representations for each question"""
    def __init__(self, batch_size, sentence_count, embedding_size,
                 dropout=None, mask=False):
        super(AttendLayer, self).__init__()
        self.attention = Attention()
        self.softmax = nn.Softmax()

        self.batch_size, self.sentence_count, self.embedding_size = \
            batch_size, sentence_count, embedding_size

        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.mask = mask

    def forward(self, text_emb, text_lens, question_emb):
        """

        :param text_emb: shape [batch_size, sentence_count, embedding_size]
        :param text_lens: shape [batch_size, ]
        :param question_emb: shape [batch_size, embedding_size]
        :return: shape [batch_size, embedding_size]
        """
        question_emb = question_emb.unsqueeze(1)

        if not self.mask:
            masks = None
        else:
            # shape: [batch_size, sentence_count]
            masks = self._length_to_mask(text_lens, max_len=self.sentence_count)
            # shape: [batch_size, 1, sentence_count]
            masks = masks.view(self.batch_size, 1, self.sentence_count)

        # shape: [batch_size, 1, embedding_size]
        attend, _ = self.attention(query=question_emb,
                                   key=text_emb,
                                   value=text_emb,
                                   mask=masks,
                                   dropout=self.dropout)
        attend = attend.squeeze()
        return attend

    def _length_to_mask(self, length, max_len=None, dtype=None):
        """length: B.
        return B x max_len.
        If max_len is None, then max of length will be used.
        """
        assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
        max_len = max_len or length.max().item()
        mask = torch.arange(max_len, device=length.device,
                            dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        return mask