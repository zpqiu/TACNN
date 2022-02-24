from torch.utils.data import Dataset
import tqdm
import torch
import random
import json


class ModelDataset(Dataset):
    def __init__(self, corpus_path, vocab, sentence_count, seq_len, encoding="utf-8", corpus_lines=None):
        self.vocab = vocab
        self.sentence_count = sentence_count
        self.seq_len = seq_len

        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            self.lines = [line.strip()
                          for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        q, doc, options, difficulty = self.parse_line(item)

        # shape: [seq_len, ]
        q_seq = self.vocab.to_seq(q, seq_len=self.seq_len, with_len=False)

        # shape: [sentence_count, seq_len]
        doc_seqs = []
        for sentence in doc:
            sentence_seq = self.vocab.to_seq(sentence, seq_len=self.seq_len, with_len=False)
            doc_seqs.append(sentence_seq)
        doc_len = len(doc)

        for i in range(self.sentence_count-doc_len):
            sentence_seq = self.vocab.to_seq("<pad>", seq_len=self.seq_len, with_len=False)
            doc_seqs.append(sentence_seq)

        # shape: [5, seq_len]
        option_seqs = []
        for option in options:
            option_seq = self.vocab.to_seq(option, seq_len=self.seq_len, with_len=False)
            option_seqs.append(option_seq)

        output = {"q": q_seq,
                  "doc": doc_seqs,
                  "doc_lens": doc_len,
                  "options": option_seqs,
                  "difficulty": difficulty
                  }

        return {key: torch.tensor(value) for key, value in output.items()}

    def parse_line(self, item):
        line = self.lines[item]
        j = json.loads(line)

        qa = j["q"]
        doc = j["doc"]
        options = j["options"]
        difficulty = j["difficulty"]

        return qa, doc, options, difficulty

