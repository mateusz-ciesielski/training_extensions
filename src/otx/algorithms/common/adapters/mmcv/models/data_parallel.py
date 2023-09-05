"""Data parallel adapter."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mmcv.parallel import MMDataParallel


class XPUDataParallel(MMDataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def scatter(self, inputs, kwargs,
                device_ids):
        for x in inputs:
            if isinstance(x, dict):
                for k in x:
                    if isinstance(x[k], torch.Tensor):
                        x[k] = x[k].to("xpu")

        return (inputs,), (kwargs, )
