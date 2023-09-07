"""Data parallel adapter."""
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Union

from torch import Tensor

from mmcv.parallel import MMDataParallel, DataContainer


ScatterInputs = Union[Tensor, DataContainer, tuple, list, dict]


def scatter(inputs: ScatterInputs,
            target_gpus: List[int],
            dim: int = 0) -> list:
    """Scatter inputs to target gpus.

    The only difference from original :func:`scatter` is to add support for
    :type:`~mmcv.parallel.DataContainer`.
    """
    def scatter_container_data(data):
        if isinstance(data, list):
            #print("------> DataContainer: scatter ~ list", len(data))
            return list(map(scatter_container_data, data))
        else:
            #print("------> DataContainer: scatter ~ tensor", type(data))
            return data.to("xpu")

    def scatter_map(obj):
        if isinstance(obj, Tensor):
            #print("------> to xpu", obj.size())
            return (obj.to("xpu"),)
        if isinstance(obj, DataContainer):
            if obj.cpu_only:
                return obj.data
            out = scatter_container_data(obj.data)
            return tuple(out) if isinstance(out, list) else (out,)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj for _ in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None  # type: ignore


def scatter_kwargs(inputs: ScatterInputs,
                   kwargs: ScatterInputs,
                   target_gpus: List[int],
                   dim: int = 0) -> Tuple[tuple, tuple]:
    """Scatter with support for kwargs dictionary."""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        length = len(kwargs) - len(inputs)
        inputs.extend([() for _ in range(length)])  # type: ignore
    elif len(kwargs) < len(inputs):
        length = len(inputs) - len(kwargs)
        kwargs.extend([{} for _ in range(length)])  # type: ignore
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class XPUDataParallel(MMDataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
