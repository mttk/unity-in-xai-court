from abc import ABC, abstractmethod
import numpy as np
import torch

from ..datasets import make_iterable


class Sampler(ABC):
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def query(self, query_size, *args, **kwargs):
        pass

    def _forward_iter(self, indices, forward_fn):
        iter = make_iterable(
            self.dataset, self.device, batch_size=self.batch_size, indices=indices
        )
        out_list = []
        for batch in iter:
            x, *_ = batch.text
            out, *_ = forward_fn(x)
            out_list.append(out)

        res = torch.stack(out_list)
        return res


class RandomSampler(Sampler):
    name = "random"

    def query(self, al_batch_size, unlabeled_inds, **kwargs):
        return np.random.choice(unlabeled_inds, size=al_batch_size, replace=False)
