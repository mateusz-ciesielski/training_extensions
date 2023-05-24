"""Balanced sampler for imbalanced data."""
import math
import random
import numpy as np
from torch.utils.data.sampler import Sampler

from otx.algorithms.common.utils.logger import get_logger

logger = get_logger()


class EL2NSampler(Sampler):  # pylint: disable=too-many-instance-attributes
    """Balanced sampler for imbalanced data for class-incremental task.

    This sampler is a sampler that creates an effective batch
    In reduce mode,
    reduce the iteration size by estimating the trials
    that all samples in the tail class are selected more than once with probability 0.999

    Args:
        dataset (Dataset): A built-up dataset
        samples_per_gpu (int): batch size of Sampling
        efficient_mode (bool): Flag about using efficient mode
    """

    def __init__(self, dataset, batch_size, efficient_mode=True, num_replicas=1, rank=0, drop_last=False, budget_size=100):
        self.batch_size = batch_size
        self.repeat = 1
        if hasattr(dataset, "times"):
            self.repeat = dataset.times
        if hasattr(dataset, "dataset"):
            self.dataset = dataset.dataset
        else:
            self.dataset = dataset
        el2n = np.array(dataset.el2n)
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        
        if el2n[0] != -1:
            indices = el2n.argsort()[-budget_size:][::-1]
            self.indices = indices.tolist()
            random.shuffle(self.indices)
        
        else:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            self.indices = indices[:budget_size*3]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        """Return length of selected samples."""
        return len(self.indices)
