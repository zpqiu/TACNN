# encoding: utf-8
"""
Author: zhaopeng qiu
Date: 12 Feb, 2019
"""
import toml
import argparse
import logging

import torch
from torch.utils.data import DataLoader

from model.difficulty_model import Model
from trainer.train import Trainer
from dataset.dataset import ModelDataset
from dataset.vocab import WordVocab
from tester.test import Tester

from utils import util
import random


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-cf", "--config", type=str, required=True, help="config file path")
    parser.add_argument("--mode", type=int, default=0, help="0 for training, 1 for testing")
    parser.add_argument("--epoch_for_test", type=int, default=0, help="the test model's epoch")

    pargs = parser.parse_args()

    args = util.Params(pargs.config)
    util.set_logger(args.model_output_path + ".log")

    logging.info("Loading Vocab, {0}".format(args.vocab_path))
    vocab = WordVocab.load_vocab(args.vocab_path)
    logging.info("Vocab Size: {0}".format(len(vocab)))

    logging.info("Loading Train Dataset, {0}".format(args.train_dataset))
    train_dataset = ModelDataset(args.train_dataset, vocab, sentence_count=args.sentence_count, seq_len=args.seq_length)

    logging.info("Loading Test Dataset, {0}".format(args.test_dataset))
    test_dataset = ModelDataset(args.test_dataset, vocab, sentence_count=args.sentence_count, seq_len=args.seq_length) \
        if args.test_dataset is not None else None

    logging.info("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1,
                                   shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=1,
                                  drop_last=True) if test_dataset is not None else None

    logging.info("Building model")
    model = Model(len(vocab), args)

    if pargs.mode not in [0, 1]:
        logging.warning("Mode should be 0 or 1.")
        return

    if pargs.mode == 0:
        logging.info("Creating Trainer")
        trainer = Trainer(model, len(vocab), args,
                          train_dataloader=train_data_loader, test_dataloader=test_data_loader)

        logging.info("Training Start")
        for epoch in range(args.epochs):
            trainer.train(epoch)
            trainer.save(epoch)

            if test_data_loader is not None:
                trainer.test(epoch)
    else:
        logging.info("Creating Tester")
        tester = Tester(model, args, test_dataloader=test_data_loader)
        predict_result_path = args.predict_result_path
        test(tester, pargs, predict_result_path)


def test(tester, pargs, predict_result_path):
    tester.load(pargs.epoch_for_test)
    predict_list, real_list = tester.predict()

    rmse, doa, pcc = tester.evaluate(predict_list, real_list)
    logging.info("RMSE: {0}, DOA: {1}, PCC: {2}".format(rmse, doa, pcc))

    with open(predict_result_path, "w", encoding="utf8") as fw:
        for p, r in zip(predict_list, real_list):
            fw.write("{0}\t{1}\n".format(p, r))


if __name__ == '__main__':
    random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    train()
