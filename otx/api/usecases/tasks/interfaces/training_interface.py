"""This module contains the interface class for tasks that can perform training."""


# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
from typing import Optional

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.adapters.dataset_adapter import DatasetAdapter


class ITrainingTask(metaclass=abc.ABCMeta):
    """A base interface class for tasks which can perform training."""

    @abc.abstractmethod
    def save_model(self, output_model: ModelEntity):
        """Save the model currently loaded by the task to `output_model`.

        This method is for instance used to save the pre-trained weights before training
        when the task has been initialised with pre-trained weights rather than an existing model.

        Args:
            output_model (ModelEntity): Output model where the weights should be stored
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: TrainParameters,
    ):
        """Train a new model using the model currently loaded by the task.

        If training was successful, the new model should be used for subsequent calls (e.g. `optimize` or `infer`).

        The new model weights should be saved in the object `output_model`.

        The task has two choices:

         - Set the output model weights, if the task was able to improve itself (according to own measures)
         - Set the model state as failed if it failed to improve itself (according to own measures)

        Args:
            dataset (DatasetEntity): Dataset containing the training and validation splits to use for training.
            output_model (ModelEntity): Output model where the weights should be stored
            train_parameters (TrainParameters): Training parameters
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cancel_training(self):
        """Cancels the currently running training process.

        If training is not running, do nothing.
        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_dataset_adapter(
        task_type: TaskType,
        train_data_roots: Optional[str] = None,
        train_ann_files: Optional[str] = None,
        val_data_roots: Optional[str] = None,
        val_ann_files: Optional[str] = None,
        test_data_roots: Optional[str] = None,
        test_ann_files: Optional[str] = None,
        unlabeled_data_roots: Optional[str] = None,
        unlabeled_file_list: Optional[str] = None,
    ) -> DatasetAdapter:
        """Get dataset adapter corresponded to the task type.

        Dataset adapter adapts DatasetEntity from local dataset files.

        Args:
            task_type (TaskType): type of the task
            train_data_roots (Optional[str]): Path for training data
            train_ann_files (Optional[str]): Path for training annotation file
            val_data_roots (Optional[str]): Path for validation data
            val_ann_files (Optional[str]): Path for validation annotation file
            test_data_roots (Optional[str]): Path for test data
            test_ann_files (Optional[str]): Path for test annotation file
            unlabeled_data_roots (Optional[str]): Path for unlabeled data
            unlabeled_file_list (Optional[str]): Path of unlabeled file list

        Since all adapters can be used for training and validation,
        the default value of train/val/test_data_roots was set to None.

        i.e)
        For the training/validation phase, test_data_roots is not used.
        For the test phase, train_data_roots and val_data_root are not used.
        """

        # TODO: Move TrainType to otx.api and make it configurable here

        raise NotImplementedError
