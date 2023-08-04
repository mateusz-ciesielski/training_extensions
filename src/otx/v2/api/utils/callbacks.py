"""Time monitor callback module."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import logging
import math
import time
from copy import deepcopy
from typing import TypeVar

import dill

from otx.v2.api.entities.train_parameters import (
    UpdateProgressCallback,
    default_progress_callback,
)

logger = logging.getLogger(__name__)


class Callback:
    """Abstract base class used to build new callbacks.

    Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    """

    def set_params(self, params: dict) -> None:
        """Set callback parameters."""
        self.params = params

    def set_model(self, model: TypeVar) -> None:
        """Set callback model."""
        self.model = model

    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """Call on epoch begin event."""

    def on_epoch_end(self, epoch: int, logs: str, **kwargs) -> None:
        """Call on epoch end event."""

    def on_batch_begin(self, batch: int, **kwargs) -> None:
        """Call on batch begin event."""

    def on_batch_end(self, batch: int, **kwargs) -> None:
        """Call on batch end event."""

    def on_train_begin(self, **kwargs) -> None:
        """Call on train begin event."""

    def on_train_end(self, batch: int, **kwargs) -> None:
        """Call on train end event."""

    def on_train_batch_begin(self, batch: int, **kwargs) -> None:
        """Call on train batch begin event."""

    def on_train_batch_end(self, batch: int, **kwargs) -> None:
        """Call on train batch end event."""

    def on_test_begin(self, **kwargs) -> None:
        """Call on test begin event."""

    def on_test_end(self, **kwargs) -> None:
        """Call on test end event."""

    def on_test_batch_begin(self, batch: int, logger: logging.Logger, **kwargs) -> None:
        """Call on test batch begin event."""

    def on_test_batch_end(self, batch: int, logger: logging.Logger, **kwargs) -> None:
        """Call on test batch end event."""


class TimeMonitorCallback(Callback):
    """A callback to monitor the progress of training.

    Args:
        num_epoch (int): Amount of epochs
        num_train_steps (int): amount of training steps per epoch
        num_val_steps (int): amount of validation steps per epoch
        num_test_steps (int): amount of testing steps
        epoch_history (int): Amount of previous epochs to calculate average epoch time over
        step_history (int): Amount of previous steps to calculate average steps time over
        update_progress_callback (UpdateProgressCallback): Callback to update progress
    """

    def __init__(
        self,
        num_epoch: int = 0,
        num_train_steps: int = 0,
        num_val_steps: int = 0,
        num_test_steps: int = 0,
        epoch_history: int = 5,
        step_history: int = 50,
        update_progress_callback: UpdateProgressCallback = default_progress_callback,
    ) -> None:
        """Initialize a Callbacks object with the given parameters.

        Args:
            num_epoch (int): The number of epochs to train the model for.
            num_train_steps (int): The number of steps to train the model for.
            num_val_steps (int): The number of steps to validate the model for.
            num_test_steps (int): The number of steps to test the model for.
            epoch_history (int): The number of past epochs to keep track of for calculating average epoch duration.
            step_history (int): The number of past steps to keep track of for calculating average step duration.
            update_progress_callback (UpdateProgressCallback): The callback function to update the progress of training.

        Returns:
            None
        """
        self.total_epochs = num_epoch
        self.train_steps = num_train_steps
        self.val_steps = num_val_steps
        self.test_steps = num_test_steps
        self.steps_per_epoch = self.train_steps + self.val_steps
        self.total_steps = math.ceil(self.steps_per_epoch * self.total_epochs + num_test_steps)
        self.current_step = 0
        self.current_epoch = 0

        # Step time calculation
        self.start_step_time = time.time()
        self.past_step_duration: list[float] = []
        self.average_step: int | float = 0
        self.step_history = step_history

        # epoch time calculation
        self.start_epoch_time = time.time()
        self.past_epoch_duration: list[float] = []
        self.average_epoch: int | float = 0
        self.epoch_history = epoch_history

        # whether model is training flag
        self.is_training = False

        self.update_progress_callback = update_progress_callback

    def __getstate__(self) -> dict:
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # update_progress_callback is not always pickable object
        # if it is not, replace it with default callback
        if not dill.pickles(state["update_progress_callback"]):
            state["update_progress_callback"] = default_progress_callback
        return state

    def __deepcopy__(self, memo: dict) -> TimeMonitorCallback:
        """Return deepcopy object."""
        update_progress_callback = self.update_progress_callback
        self.update_progress_callback = None
        self.__dict__["__deepcopy__"] = None

        result = deepcopy(self, memo)

        self.__dict__.pop("__deepcopy__")
        result.__dict__.pop("__deepcopy__")
        result.update_progress_callback = update_progress_callback
        self.update_progress_callback = update_progress_callback

        memo[id(self)] = result
        return result

    def on_train_batch_begin(self, batch: int, **kwargs) -> None:
        """Set the value of current step and start the timer."""
        super().on_train_batch_begin(batch=batch, **kwargs)
        self.current_step += 1
        self.start_step_time = time.time()

    def on_train_batch_end(self, batch: int, **kwargs) -> None:
        """Compute average time taken to complete a step."""
        super().on_train_batch_end(batch=batch, **kwargs)
        self.__calculate_average_step()

    def is_stalling(self) -> bool:
        """Return True if the training is stalling.

        Returns True if the current step has taken more than 30 seconds and
        at least 20x more than the average step duration
        """
        factor = 20
        min_abs_threshold = 30  # seconds
        if self.is_training and self.current_step > 2:
            step_duration = time.time() - self.start_step_time
            if step_duration > min_abs_threshold and step_duration > factor * self.average_step:
                logger.error(
                    f"Step {self.current_step} has taken {step_duration}s which is "
                    f">{min_abs_threshold}s and  {factor} times "
                    f"more than the expected {self.average_step}s",
                )
                return True
        return False

    def __calculate_average_step(self) -> None:
        """Compute average duration taken to complete a step."""
        self.past_step_duration.append(time.time() - self.start_step_time)
        if len(self.past_step_duration) > self.step_history:
            self.past_step_duration.remove(self.past_step_duration[0])
        self.average_step = sum(self.past_step_duration) / len(self.past_step_duration)

    def on_test_batch_begin(self, batch: int, logger: logging.Logger, **kwargs) -> None:
        """Set the number of current epoch and start the timer."""
        super().on_test_batch_begin(batch=batch, logger=logger, **kwargs)
        self.current_step += 1
        self.start_step_time = time.time()

    def on_test_batch_end(self, batch: int, logger: logging.Logger, **kwargs) -> None:
        """Compute average time taken to complete a step based on a running average of `step_history` steps."""
        super().on_test_batch_end(batch=batch, logger=logger, **kwargs)
        self.__calculate_average_step()

    def on_train_begin(self, **kwargs) -> None:
        """Set training to true."""
        super().on_train_begin(**kwargs)
        self.is_training = True

    def on_train_end(self, batch: int, **kwargs) -> None:
        """Handle early stopping when the total_steps is greater than the current_step."""
        # To handle cases where early stopping stops the task the progress will still be accurate
        super().on_train_end(batch=batch, **kwargs)
        self.current_step = self.total_steps - self.test_steps
        self.current_epoch = self.total_epochs
        self.is_training = False

    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """Set the number of current epoch and start the timer."""
        super().on_epoch_begin(epoch=epoch, **kwargs)
        self.current_epoch = epoch + 1
        self.start_epoch_time = time.time()

    def on_epoch_end(self, epoch: int, logs: str, **kwargs) -> None:
        """Compute the average time taken to complete an epoch based on a running average of `epoch_history` epochs."""
        super().on_epoch_end(epoch=epoch, logs=logs, **kwargs)
        self.past_epoch_duration.append(time.time() - self.start_epoch_time)
        self._calculate_average_epoch()
        self.update_progress_callback(self.get_progress())

    def _calculate_average_epoch(self) -> None:
        if len(self.past_epoch_duration) > self.epoch_history:
            del self.past_epoch_duration[0]
        self.average_epoch = sum(self.past_epoch_duration) / len(self.past_epoch_duration)

    def get_progress(self) -> float:
        """Return current progress as a percentage."""
        return (self.current_step / self.total_steps) * 100
