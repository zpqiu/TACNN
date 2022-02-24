# encoding:utf-8

import torch
import torch.nn as nn

from .attend import AttendLayer
from .encode import SentenceCNNLayer
from .embedding import token


class Model(nn.Module):
    def __init__(self, vocab_size, args):
        super().__init__()
        # self.embed = TokenEmbedding(vocab_size, embed_size=args.embed_size)
        self.embed = token.create_embedding_layer(weight_path=args.embedding,
                                                  vocab_size=vocab_size,
                                                  embed_size=args.embed_size)

        channels = [int(c) for c in args.channels.split(",")]
        pool_size_list = [int(c) for c in args.pool_size_list.split(",")]
        self.encode = SentenceCNNLayer(batch_size=args.batch_size,
                                       seq_length=args.seq_length,
                                       embedding_size=args.embed_size,
                                       kernel_size=args.kernel_size,
                                       channels=channels,
                                       pool_size_list=pool_size_list,
                                       dropout=args.dropout)

        self.doc_attend = AttendLayer(batch_size=args.batch_size,
                                      sentence_count=args.sentence_count,
                                      embedding_size=args.embed_size,
                                      mask=args.use_mask)

        self.option_attend = AttendLayer(batch_size=args.batch_size,
                                         sentence_count=5,
                                         embedding_size=args.embed_size,
                                         mask=False)

        self.prediction = nn.Sequential(
            nn.Linear(args.encode_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1),
            nn.Sigmoid()
        )

        self.batch_size, self.seq_length = args.batch_size, args.seq_length

    def forward(self, q, doc, options):
        """
        :param q: shape: [batch_size, seq_length]
        :param doc: Tuple, 1. [batch_size, sentence_count, seq_length], 2. [batch_size,]
        :param options: [batch_size, 5, seq_length]
        :return: [batch_size, ]
        """
        q = q.view(self.batch_size, 1, self.seq_length)
        qa_embed = self.embed(q)
        doc_embed = self.embed(doc[0])
        options_embed = self.embed(options)

        qa_encode = self.encode(qa_embed)
        doc_encode = self.encode(doc_embed)
        options_encode = self.encode(options_embed)

        # shape: [batch_size, encode_size]
        qa_encode = qa_encode.squeeze()

        doc_attn = self.doc_attend(doc_encode, doc[1], qa_encode)
        options_attn = self.option_attend(options_encode, None, qa_encode)

        h = torch.cat([doc_attn, qa_encode, options_attn], dim=1)

        output = self.prediction(h)

        return output.squeeze()
