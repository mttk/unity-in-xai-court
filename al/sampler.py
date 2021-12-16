from abc import ABC, abstractmethod
import numpy as np
import torch

from dataloaders import make_iterable


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

        res = torch.cat(out_list)
        return res

    def _predict_probs_dropout(self, model, n_drop, indices, num_labels):
        model.train()

        probs = torch.zeros([len(indices), num_labels]).to(self.device)

        iter = make_iterable(
            self.dataset, self.device, batch_size=self.batch_size, indices=indices
        )

        # Dropout approximation for output probs.
        for _ in range(n_drop):
            index = 0
            for batch in iter:
                x, lengths = batch.text
                probs_i = model.predict_probs(x, lengths=lengths)
                start = index
                end = start + x.shape[0]
                probs[start:end] += probs_i
                index = end

        probs /= n_drop

        return probs


class RandomSampler(Sampler):
    name = "random"

    def query(self, query_size, unlab_inds, **kwargs):
        return np.random.choice(unlab_inds, size=query_size, replace=False)
