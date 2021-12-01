from abc import ABC, abstractmethod
import numpy as np
import torch

from ..datasets import make_iterable


class Sampler(ABC):
    def __init__(self, dataset, batch_size, device, meta):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.meta = meta

    @abstractmethod
    def query(self, query_size, *args, **kwargs):
        pass

    def _forward_iter(self, indices, forward_fn):
        iter = make_iterable(
            self.dataset, self.device, batch_size=self.batch_size, indices=indices
        )
        out_list = []
        for batch in iter:
            ret_val = batch.text
            if type(ret_val) is tuple:
                # Unpack inputs and lengths.
                x, lengths = ret_val
            else:
                x = ret_val
                # Set lengths to None to match the method's signature.
                lengths = None
            out = forward_fn(x, lengths=lengths)
            out_list.append(out)

        res = torch.stack(out_list)
        return res


class RandomSampler(Sampler):
    name = "random"

    def query(self, query_size, unlab_inds, **kwargs):
        return np.random.choice(unlab_inds, size=query_size, replace=False)
