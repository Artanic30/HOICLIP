from typing import Union, List
import logging
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import os
import torch.distributed as dist
import warnings
import math


class MultiStepLRWarmup(MultiStepLR):

    def __init__(self, *args, **kwargs):
        self.warmup_iter = kwargs['warmup_iter']
        self.cur_iter = 0
        self.warmup_ratio = kwargs['warmup_ratio']
        self.init_lr = None
        del kwargs['warmup_iter']
        del kwargs['warmup_ratio']
        super(MultiStepLRWarmup, self).__init__(*args, **kwargs)
        self.init_lr = [group['lr'] for group in self.optimizer.param_groups]

    def iter_step(self):
        self.cur_iter += 1
        if self.cur_iter <= self.warmup_iter and self.init_lr:
            values = [lr * (self.warmup_ratio + (1 - self.warmup_ratio) * (self.cur_iter / self.warmup_iter))
                      for lr in self.init_lr]
            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, 0)

            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class CosineAnnealingLRWarmup(CosineAnnealingLR):

    def __init__(self, *args, **kwargs):
        self.warmup_iter = kwargs['warmup_iter']
        self.cur_iter = 0
        self.warmup_ratio = kwargs['warmup_ratio']
        self.init_lr = None
        del kwargs['warmup_iter']
        del kwargs['warmup_ratio']
        super(CosineAnnealingLRWarmup, self).__init__(*args, **kwargs)
        self.init_lr = [group['lr'] for group in self.optimizer.param_groups]

    def iter_step(self):
        self.cur_iter += 1
        if self.cur_iter <= self.warmup_iter and self.init_lr:
            values = [lr * (self.warmup_ratio + (1 - self.warmup_ratio) * (self.cur_iter / self.warmup_iter))
                      for lr in self.init_lr]
            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, 0)

            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - init_lr * self.eta_min) + init_lr * self.eta_min
                for init_lr, group in zip(self.init_lr, self.optimizer.param_groups)]
