from abc import ABC, abstractmethod
import numpy as np


class Sampler(ABC):
    @abstractmethod
    def query(self, *args, **kwargs):
        pass


class RandomSampler(Sampler):
    name = "random"

    def query(self, al_batch_size, unlabeled_inds, **kwargs):
        return np.random.choice(unlabeled_inds, size=al_batch_size, replace=False)
