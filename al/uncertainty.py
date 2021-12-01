from .sampler import Sampler


import numpy as np


class LeastConfidentSampler(Sampler):
    name = "least_confident"

    def query(self, query_size, unlab_inds, model, **kwargs):
        probs = self._forward_iter(unlab_inds, model.predict_probs).cpu().numpy()
        max_probs = np.max(probs, axis=1)

        # Retrieve `query_size` instances with highest posterior probabilities.
        top_n = np.argpartition(max_probs, query_size)[:query_size]
        return unlab_inds[top_n]


class MarginSampler(Sampler):
    name = "margin"

    def query(self, query_size, unlab_inds, model, **kwargs):
        probs = self._forward_iter(unlab_inds, model.predict_probs).cpu().numpy()

        sort_probs = np.sort(probs, 1)[:, -2:]
        min_margin = sort_probs[:, 1] - sort_probs[:, 0]

        # Retrieve `query_size` instances with smallest margins.
        top_n = np.argpartition(min_margin, query_size)[:query_size]
        return unlab_inds[top_n]


class EntropySampler(Sampler):
    name = "entropy"

    def query(self, query_size, unlabeled_inds, model, **kwargs):
        probs = self._forward_iter(unlabeled_inds, model.predict_probs).cpu().numpy()

        # Clip for numerical stability.
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)

        # Retrieve `query_size` instances with highest entropies.
        top_n = np.argpartition(entropies, -query_size)[-query_size:]
        return unlabeled_inds[top_n]
