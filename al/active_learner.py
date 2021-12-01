import numpy as np


class ActiveLearner:
    # TODO: AL loop abstraction
    def __init__(self, dataset, sampler):
        self.dataset = dataset
        self.sampler = sampler
        self.lab_mask = np.full(len(dataset), False)

    def al_loop(self, model, n_warm_start, query_size):
        # TODO: stratified warm start
        random_inds = np.random.choice(len(self.dataset), n_warm_start, replace=False)
        self.lab_mask[random_inds] = True

        stopping_criterion = True
        while not stopping_criterion:
            # TODO
            # 1. train model with labeled data: fine-tune vs. re-train
            # ...
            # 2. evaluate model (test set)
            # ...
            # 3. Retrieve active sample.
            unlab_inds, *_ = np.where(self.lab_mask)
            selected_inds = self.sampler.query(
                query_size=query_size, unlab_inds=unlab_inds, model=model
            )
            self.lab_mask[selected_inds] = True
            # 4. calculate intepretability metrics
            # ...
            pass
