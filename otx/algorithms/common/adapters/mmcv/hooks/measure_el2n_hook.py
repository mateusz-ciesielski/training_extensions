import os
from typing import Callable
import torch
from torch import nn
import numpy as np

from mmcv.runner import BaseRunner, EpochBasedRunner
from mmcv.runner.hooks import HOOKS, Hook
from torch.utils.data import DataLoader

from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()

def get_lord_error_fn(X,Y):
    errors = torch.tensor(X) - nn.functional.one_hot(Y, num_classes=20).squeeze()
    scores = np.linalg.norm(errors, ord=2, axis=-1)
    return np.array(scores)
    
@HOOKS.register_module()
class MeasureEL2NHook(Hook):
    
    def after_train_epoch(self, runner):
        cur_epoch = runner.epoch
        
        if cur_epoch % 20 == 0:
            dataset = runner.data_loader.dataset
            collate_fn = runner.data_loader.collate_fn
            batch_size=runner.data_loader.batch_size

            new_dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=False,
                collate_fn=collate_fn,

            )
            for i,data in enumerate(new_dataloader):
                with torch.no_grad():
                    gt_label = data.pop("gt_label")
                    result = runner.model(return_loss=False, **data)
                    scores = get_lord_error_fn(result, gt_label)
                    
                    for j, score in enumerate(scores):
                        dataset.el2n[i*batch_size + j] = score
