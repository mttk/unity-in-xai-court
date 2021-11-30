from .sampler import Sampler
import numpy as np


class LeastConfidentSampler(Sampler):
    name = "least_confident"

    def select_batch(self, X, al_batch_size, unlabeled_inds, model, **kwargs):
        X_unlab = X[unlabeled_inds]
        # TODO: Retrieve probs for the entire unlabeled set.
        # This should be done in batches to avoid memory problems.
        # Up for discussion: where should we situate this method?
        probs = model.predict_probs(X_unlab)
        probs = probs.cpu().numpy()
        max_probs = np.max(probs, axis=1)

        # Retrieve `batch_size` instances with highest posterior probabilities.
        top_n = np.argpartition(max_probs, al_batch_size)[:al_batch_size]
        return unlabeled_inds[top_n]


class MarginSampler(Sampler):
    name = "margin"

    def query(self, X, al_batch_size, unlabeled_inds, model, **kwargs):
        X_unlab = X[unlabeled_inds]
        probs = model.predict_probs(X_unlab)
        probs = probs.cpu().numpy()
        sort_probs = np.sort(probs, 1)[:, -2:]
        min_margin = sort_probs[:, 1] - sort_probs[:, 0]
        min_margin = self._margin(X_unlab, model)

        # Retrieve `batch_size` instances with smallest margins.
        top_n = np.argpartition(min_margin, al_batch_size)[:al_batch_size]
        return unlabeled_inds[top_n]


class EntropySampler(Sampler):
    name = "entropy"

    def query(self, X, al_batch_size, unlabeled_inds, model, **kwargs):
        X_unlab = X[unlabeled_inds]
        probs = model.predict_probs(X_unlab)
        probs = probs.cpu().numpy()

        # Clip for numerical stability.
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        entropies = np.sum(-probs * np.log(probs), axis=1)

        # Retrieve `batch_size` instances with highest entropies.
        top_n = np.argpartition(entropies, -al_batch_size)[-al_batch_size:]
        return unlabeled_inds[top_n]
