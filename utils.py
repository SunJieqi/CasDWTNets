# -*-coding:utf-8-*-
import logging
import math
import os
import shutil

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from config import Config


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class Logger(object):
    def __init__(self, log_file_name, log_level, logger_name):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] - [%(filename)s line:%(lineno)3d] : %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mixup_data(x, y, alpha):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(optimizer)
    if Config.type == "STEP":
        if epoch in Config.lr_epochs:
            lr = lr * Config.lr_mults
    elif Config.type == "COSINE":
        ratio = epoch / Config.epochs
        lr = (
            Config.min_lr
            + (Config.base_lr - Config.min_lr)
            * (1.0 + math.cos(math.pi * ratio))
            / 2.0
        )
    elif Config.type == "HTD":
        ratio = epoch / Config.epochs
        lr = (
            Config.min_lr
            + (Config.base_lr - Config.min_lr)
            * (
                1.0
                - math.tanh(
                    Config.lower_bound
                    + (
                        Config.upper_bound
                        - Config.lower_bound
                    )
                    * ratio
                )
            )
            / 2.0
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr
