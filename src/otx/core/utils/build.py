# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for mmX build function."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.core.utils.config import convert_conf_to_mmconfig_dict
import torch
if TYPE_CHECKING:
    from mmengine.registry import Registry
    from omegaconf import DictConfig
    from torch import nn

def build_mm_model(config: DictConfig,
                   model_registry: Registry,
                   load_from: str,
                   load_backbone_only: bool = False) -> nn.Module:
    """Build a model by using the registry."""
    from mmengine.runner import load_checkpoint

    from otx import algo  # noqa: F401


    try:
        model = model_registry.build(convert_conf_to_mmconfig_dict(config, to="tuple"))
    except AssertionError:
        model = model_registry.build(convert_conf_to_mmconfig_dict(config, to="list"))
    model_device = next(model.parameters()).device

    if load_backbone_only:
        load_checkpoint(model.backbone, load_from, map_location=model_device)
    elif load_from is not None:
        load_checkpoint(model, load_from, map_location=model_device)

    return model
