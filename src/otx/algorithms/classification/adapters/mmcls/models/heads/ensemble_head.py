"""Module defining for OTX ensemble classification head."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from collections import defaultdict
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.builder import HEADS
from .cls_head import ClsHead

__all__ = ["LinearEnsembleClsHead"]


class LinearBN(nn.Module):
    """Layer combining Linear and BatchNorm layers for the structural re-parametrization.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        num_bn_branches (int): Number of batch normalization branches to make
            ensembles of non-linear heads. Default is 3,
        dropout_prob (float): Drop-out probability in front of the final FC layer.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_bn_branches: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()

        self.bn_path = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_channels, in_channels),
                    nn.BatchNorm1d(in_channels),
                )
                for _ in range(num_bn_branches)
            ]
        )
        self.id_path = nn.Identity()
        self.out_head = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.id_path(x)

        for bn_path in self.bn_path:
            out = out + bn_path(x)

        return self.out_head(out)


@HEADS.register_module()
class LinearEnsembleClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        num_bn_branches (int): Number of batch normalization branches to make
            ensembles of non-linear heads. Default is 3,
        num_ensembles (int): Number of heads to ensemble. Default is 5,
        dropout_prob (float): Drop-out probability in front of the final FC layer.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        num_bn_branches: int = 3,
        num_ensembles: int = 5,
        dropout_prob: float = 0.0,
        init_cfg=dict(type="Normal", layer="Linear", std=0.01),
        *args,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_bn_branches = num_bn_branches
        self.num_ensembles = num_ensembles

        if self.num_classes <= 0:
            raise ValueError(f"num_classes={num_classes} must be a positive integer")

        self.linear_bns = nn.ModuleList(
            LinearBN(
                in_channels=in_channels,
                num_classes=num_classes,
                num_bn_branches=num_bn_branches,
                dropout_prob=dropout_prob,
            )
            for _ in range(self.num_ensembles)
        )

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        scale = 1 / len(self.linear_bns)
        cls_score = sum(scale * linear_bn(x) for linear_bn in self.linear_bns)

        if softmax:
            pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)

        scale = 1 / len(self.linear_bns)
        losses = defaultdict(list)
        for linear_bn in self.linear_bns:
            for k, v in self.loss(linear_bn(x), gt_label, **kwargs).items():
                losses[k].append(scale * v)
        return {k: sum(v) for k, v in losses.items()}
