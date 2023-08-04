"""Module defining Classification Head for MMOV inference."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmpretrain.models.builder import HEADS
from mmpretrain.models.heads import ClsHead as OriginClsHead


@HEADS.register_module(force=True)
class ClsHead(OriginClsHead):
    """Classification Head for MMOV inference."""

    def __init__(self, *args, **kwargs) -> None:
        do_squeeze = kwargs.pop("do_squeeze", False)
        super().__init__(*args, **kwargs)
        self._do_squeeze = do_squeeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward fuction of ClsHead class."""
        return self.simple_test(x)

    def forward_train(self, cls_score: torch.Tensor, gt_label: torch.Tensor) -> dict:
        """Forward_train fuction of ClsHead class."""
        if self._do_squeeze:
            cls_score = cls_score.unsqueeze(0).squeeze()
        return super().forward_train(cls_score, gt_label)

    def simple_test(self, cls_score: torch.Tensor) -> torch.Tensor:
        """Test without augmentation."""
        if self._do_squeeze:
            cls_score = cls_score.unsqueeze(0).squeeze()
        return super().simple_test(cls_score)
