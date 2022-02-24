#encoding: utf-8
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceCNNLayer(nn.Module):
    """CNN sentence to a fixed-length vector"""
    def __init__(self, batch_size, seq_length, embedding_size,
                 kernel_size, channels, pool_size_list, dropout=None):
        super(SentenceCNNLayer, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      padding=kernel_size-1)
            for in_channel, out_channel in zip(channels[:-1], channels[1:])
        ])

        self.pools = nn.ModuleList([
            nn.MaxPool1d(kernel_size=kernel_size) for kernel_size in pool_size_list
        ])

        self.batch_size, self.seq_length, self.embedding_size = \
            batch_size, seq_length, embedding_size

        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

    def conv_and_pooling(self, x, conv, pooling):
        x = conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = F.relu(x)
        x = pooling(x)

        return x

    def forward(self, emb):
        """
        多个 CNN+Pooling 层
        :param emb: shape: [batch_size, sentence_count, sequence_length, embedding_size]
        :return: shape: [batch_size, sentence_count, (last_channel_size * last_seq_length)]
        """
        sentence_count = emb.shape[1]
        # shape: [(batch_size * sentence_count), sequence_length, embedding_size]
        embeddings = emb.view(-1, self.seq_length, self.embedding_size)

        # shape: [(batch_size * sentence_count), embedding_size, sequence_length]
        embeddings = embeddings.permute(0, 2, 1)
        for conv, pooling in zip(self.convs, self.pools):
            embeddings = self.conv_and_pooling(embeddings, conv, pooling)

        embeddings = embeddings.view(self.batch_size, sentence_count, -1)

        return embeddings
