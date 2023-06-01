"""This module define a module to adapt dataset entity from a local dataset files."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label_schema import LabelSchemaEntity


class DatasetAdapter(metaclass=abc.ABCMeta):
    """Dataset adapter to extract dataset entity from a local dataset files."""

    @abc.abstractmethod
    def get_otx_dataset(self) -> DatasetEntity:
        """Get DatasetEntity."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_label_schema(self) -> LabelSchemaEntity:
        """Get Label Schema."""
        raise NotImplementedError
