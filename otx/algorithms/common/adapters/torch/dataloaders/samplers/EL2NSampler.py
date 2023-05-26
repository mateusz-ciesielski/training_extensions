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
        
        self.img_indices = self.dataset.img_indices
        
        
        if dataset.is_el2n_updated > 0:
            start = 0.1
            increase = 0 # 0.01 * (dataset.is_el2n_updated - 1)
            hard_samples = int(budget_size*(start + increase))
            el2n_sorted = el2n.argsort()
            hard_indices = el2n_sorted[-hard_samples:][::-1]
            easy_indices = el2n_sorted[:-hard_samples].tolist()
            random.shuffle(easy_indices)
            easy_indices = easy_indices[:budget_size-hard_samples]

            self.indices = []
            self.indices.extend(hard_indices.tolist())
            self.indices.extend(easy_indices)
            random.shuffle(self.indices)
            logger.info(f"Finding new hard samples... dataset size to {len(self.indices)}")

            """
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            self.indices = indices[:budget_size]
            
            logger.info(f"Finding new hard samples... dataset size to {len(self.indices)}")
            """
        else:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            self.indices = indices[:budget_size*3]

            logger.info(f"Random sampling without replacement... dataset size to {len(self.indices)}")

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        """Return length of selected samples."""
        return len(self.indices)
