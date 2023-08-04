"""Hook module to track loss dynamics during training and export these statistics to Datumaro format."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os.path as osp

from mmengine.config import Config, ConfigDict
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from torch.nn.parallel import DataParallel

from otx.v2.adapters.datumaro.noisy_label_detection.base import LossDynamicsTracker, LossDynamicsTrackingMixin
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import (
    update_or_add_custom_hook,
)
from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.utils.logger import get_logger

logger = get_logger()

__all__ = ["LossDynamicsTrackingHook"]


@HOOKS.register_module()
class LossDynamicsTrackingHook(Hook):
    """Tracking loss dynamics during training and export it to Datumaro dataset format."""

    def __init__(self, output_path: str, alpha: float = 0.001) -> None:
        self._output_fpath = osp.join(output_path, "noisy_label_detection")

    def before_run(self, runner: Runner) -> None:
        """Before run, check the type of model for safe running."""
        if not isinstance(runner.model, DataParallel):
            raise NotImplementedError(f"Except DataParallel, runner.model={type(runner.model)} is not supported now.")

    def before_train_epoch(self, runner: Runner) -> None:
        """Initialize the tracker for training loss dynamics tracking.

        Tracker needs the training dataset for initialization.
        However, there is no way to access to dataloader until the beginning of training epoch.
        """
        self._init_tracker(runner, runner.data_loader.dataset.otx_dataset)

    def _get_tracker(self, runner: Runner) -> LossDynamicsTracker:
        model = runner.model.module

        if not isinstance(model, LossDynamicsTrackingMixin):
            raise RuntimeError(
                f"The model should be an instance of LossDynamicsTrackingMixin, but type(model)={type(model)}.",
            )
        return model.loss_dyns_tracker

    def _init_tracker(self, runner: Runner, otx_dataset: DatasetEntity) -> None:
        tracker = self._get_tracker(runner)
        if tracker.initialized:
            return

        logger.info("Initialize training loss dynamics tracker.")
        tracker.init_with_otx_dataset(otx_dataset)

    def after_train_iter(self, runner: Runner) -> None:
        """Accumulate training loss dynamics.

        It should be here because it needs to access the training iteration.
        """
        tracker = self._get_tracker(runner)
        tracker.accumulate(runner.outputs, runner.iter)

    def after_run(self, runner: Runner) -> None:
        """Export loss dynamics statistics to Datumaro format."""

        tracker = self._get_tracker(runner)

        if tracker.initialized:
            logger.info(f"Export training loss dynamics to {self._output_fpath}")
            tracker.export(self._output_fpath)

    @classmethod
    def configure_recipe(cls: Hook, recipe_cfg: Config, output_path: str) -> None:
        """Configure recipe to enable loss dynamics tracking."""
        recipe_cfg.model["track_loss_dynamics"] = True

        update_or_add_custom_hook(
            recipe_cfg,
            ConfigDict(
                type="LossDynamicsTrackingHook",
                priority="LOWEST",
                output_path=output_path,
            ),
        )
