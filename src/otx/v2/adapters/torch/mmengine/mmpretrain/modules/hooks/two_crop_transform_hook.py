"""Two crop transform hook."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from torch.utils.data import Dataset

from otx.v2.adapters.torch.mmengine.mmpretrain.modules.datasets.pipelines.transforms import TwoCropTransform
from otx.v2.api.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class TwoCropTransformHook(Hook):
    """TwoCropTransformHook with every specific interval.

    This hook decides whether using single pipeline or two pipelines
    implemented in `TwoCropTransform` for the current iteration.

    Args:
        interval (int): If `interval` == 1, both pipelines is used.
            If `interval` > 1, the first pipeline is used and then
            both pipelines are used every `interval`. Defaults to 1.
        by_epoch (bool): (TODO) Use `interval` by epoch. Defaults to False.
    """

    def __init__(self, interval: int = 1, by_epoch: bool = False) -> None:
        if by_epoch:
            raise NotImplementedError("by_epoch is not implemented.")

        self.interval = interval
        self.cnt = 0

    def _get_dataset(self, runner: Runner) -> Dataset:
        """Get dataset to handle `is_both`."""
        if hasattr(runner.data_loader.dataset, "dataset"):
            # for RepeatDataset
            dataset = runner.data_loader.dataset.dataset
        else:
            dataset = runner.data_loader.dataset

        return dataset

    def _find_two_crop_transform(self, transforms: list) -> Optional[TwoCropTransform]:
        """Find TwoCropTransform among transforms."""
        for transform in transforms:
            if transform.__class__.__name__ == "TwoCropTransform":
                return transform
        return None

    def before_train_epoch(self, runner: Runner) -> None:
        """Called before_train_epoch in TwoCropTransformHook."""
        # Always keep `TwoCropTransform` enabled.
        if self.interval == 1:
            return

        dataset = self._get_dataset(runner)
        two_crop_transform = self._find_two_crop_transform(dataset.pipeline.transforms)
        if two_crop_transform is None:
            return
        if self.cnt == self.interval - 1:
            # start using both pipelines
            two_crop_transform.is_both = True
        else:
            two_crop_transform.is_both = False

    def after_train_iter(self, runner: Runner) -> None:
        """Called after_train_iter in TwoCropTransformHook."""
        # Always keep `TwoCropTransform` enabled.
        if self.interval == 1:
            return

        if self.cnt < self.interval - 1:
            # Instead of using `runner.every_n_iters` or `runner.every_n_inner_iters`,
            # this condition is used to compare `self.cnt` with `self.interval` throughout the entire epochs.
            self.cnt += 1

        if self.cnt == self.interval - 1:
            dataset = self._get_dataset(runner)
            two_crop_transform = self._find_two_crop_transform(dataset.pipeline.transforms)
            if two_crop_transform is None:
                return
            if not two_crop_transform.is_both:
                # If `self.cnt` == `self.interval`-1, there are two cases,
                # 1. `self.cnt` was updated in L709, so `is_both` must be on for the next iter.
                # 2. if the current iter was already conducted, `is_both` must be off.
                two_crop_transform.is_both = True
            else:
                two_crop_transform.is_both = False
                self.cnt = 0
