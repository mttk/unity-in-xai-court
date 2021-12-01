import numpy as np
import torch
import time

from datasets import make_iterable
dataset, device, batch_size=32, train=False, indices=None

class ActiveLearner:
    # TODO: AL loop abstraction
    def __init__(self, sampler, train_set, test_set, device, args, meta):
        self.sampler = sampler
        self.train_set = train_set
        self.test_set = test_set
        self.batch_size = args.batch_size
        self.device = device
        self.args = args
        self.meta = meta

    def al_loop(self, model, n_warm_start, query_size):
        # Initialize label mask.
        lab_mask = np.full(len(self.train_set), False)
        # TODO: stratified warm start
        random_inds = np.random.choice(len(self.train_set), n_warm_start, replace=False)
        lab_mask[random_inds] = True

        stopping_criterion = True
        while not stopping_criterion:
            # TODO
            # 1. train model with labeled data: fine-tune vs. re-train
            self.train_model(model)
            # ...
            # 2. evaluate model (test set)
            self.evaluate_model(model)
            # ...
            # 3. Retrieve active sample.
            unlab_inds, *_ = np.where(lab_mask)
            selected_inds = self.sampler.query(
                query_size=query_size, unlab_inds=unlab_inds, model=model
            )
            lab_mask[selected_inds] = True
            # 4. calculate intepretability metrics
            # ...
            pass

    def _train_model(self, model, lab_mask):
        model.train()
        data = make_iterable(
            self.train_set, self.device, batch_size=self.batch_size, indices=lab_mask
        )

        accuracy, confusion_matrix = 0, np.zeros((self.meta.num_labels, self.meta.num_labels), dtype=int)
        total_loss = 0.

        for batch_num, batch in enumerate(data):

            t = time.time()
            # Unpack batch & cast to device
            (x, lengths), y = batch.text, batch.label

            y = y.squeeze() # y needs to be a 1D tensor for xent(batch_size)

            logits, return_dict = model(x, lengths)

            # Bookkeeping and cast label to float
            accuracy, confusion_matrix = ActiveLearner.update_stats(accuracy, confusion_matrix, logits, y)
            if logits.shape[-1] == 1:
                # binary cross entropy, cast labels to float
                y = y.type(torch.float)

            loss = self.criterion(logits.view(-1, self.meta.num_targets).squeeze(), y)

            total_loss += float(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
            self.optimizer.step()

            print("[Batch]: {}/{} in {:.5f} seconds".format(
                batch_num, len(data), time.time() - t), end='\r', flush=True)
        result_dict = {
            'loss' : total_loss / len(data) / data.batch_size
        }
        return result_dict

    def _evaluate_model(self, model):
        make_iterable()
        model.eval()
        pass

    @staticmethod
    def update_stats(accuracy, confusion_matrix, logits, y):
        if logits.shape[-1] == 1:
            # BCE, need to check ge 0 (or gt 0?)
            max_ind = torch.ge(logits, 0).type(torch.long).squeeze()
        else:
            _, max_ind = torch.max(logits, 1)

        equal = torch.eq(max_ind, y)
        correct = int(torch.sum(equal))
        if len(max_ind.shape) == 0:
            # only one element here? is this even possible?
            confusion_matrix[y, max_ind] += 1
        else:
            for j, i in zip(max_ind, y):
                confusion_matrix[int(i),int(j)] += 1

        return accuracy + correct, confusion_matrix