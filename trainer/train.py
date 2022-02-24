# encoding: utf-8
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.difficulty_model import Model
from utils.optim import AdamW

import argparse
import tqdm
import logging
import numpy as np
from scipy.stats import pearsonr


class Trainer:
    def __init__(self, model: Model, vocab_size: int, args: argparse.Namespace,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None):
        """
        :param model: model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param config: config dict
        """
        self.args = args

        # Setup cuda device for model training
        cuda_condition = torch.cuda.is_available() and args.with_cuda
        self.device = torch.device(("cuda:"+args.gpu) if cuda_condition else "cpu")

        # TODO: need check
        self.model = model.to(self.device)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
        # self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using MSE Loss function for predicting the difficulty
        self.criterion = nn.MSELoss()

        self.log_freq = args.log_freq

        logging.info("Total Parameters: {0}".format(sum([p.nelement() for p in self.model.parameters()])))

    def train(self, epoch):
        self.model.train()
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every epoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_loss = 0.0
        predict_list = []
        real_list = []

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the qa and doc_set
            predict_difficulty = self.model.forward(data["q"],
                                                    (data["doc"], data["doc_lens"]),
                                                    data["options"])

            # 2. MSE loss
            mse_loss = self.criterion(predict_difficulty, data["difficulty"])

            # 3. backward and optimization only in train
            if train:
                self.optim.zero_grad()
                mse_loss.backward()
                # self.optim_schedule.step_and_update_lr()
                self.optim.step()
            else:
                predict_list += predict_difficulty.data.cpu().numpy().tolist()
                real_list += data["difficulty"].data.cpu().numpy().tolist()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_mse_loss": mse_loss,
            }

            avg_loss += mse_loss.data.cpu().numpy()

            if (i+1) % self.log_freq == 0:
                post_fix["avg_mse_loss"] = avg_loss / self.log_freq
                data_iter.write(str(post_fix))
                total_loss += avg_loss
                avg_loss = 0.0

        logging.info("EP{0}_{1}, avg_loss={2}".format(epoch, str_code, total_loss / len(data_iter)))

        if not train:
            rmse, doa, pcc = self.evaluate(predict_list, real_list)
            logging.info("RMSE: {0}, DOA: {1}, PCC: {2}".format(rmse, doa, pcc))

    def evaluate(self, predict, real):
        """
        多个metrics
        :param predict:
        :param real:
        :return: Tuple(float, float, float), 1. RMSE, 2. degree of agreement, 3. pearson
        """
        predict = np.array(predict)
        real = np.array(real)

        rmse = np.sqrt(((predict-real)**2).mean())
        doa = 0
        N = len(predict)
        for x, y in zip(real.argsort(), predict.argsort()):
            doa += (N-max(x, y))
        doa = doa/(N*(N+1))*2
        pcc = pearsonr(real, predict)

        return rmse, doa, pcc

    def save(self, epoch):
        """
        Saving the current model on file_path

        :param epoch: current epoch number
        :return: final_output_path
        """
        file_path = self.args.model_output_path
        output_path = file_path + ".ep%d" % epoch
        self.model.cpu()
        torch.save(self.model.state_dict(), output_path)
        self.model.to(self.device)
        logging.info("EP:{0} Model Saved on: {1}".format(epoch, output_path))
        return output_path
