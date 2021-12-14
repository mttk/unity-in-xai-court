import numpy as np
import torch
import time

from train import *
from dataloaders import *
from al.uncertainty import MarginSampler


class ActiveLearner:
    def __init__(self, sampler, train_set, test_set, device, args, meta):
        self.sampler = sampler
        self.train_set = train_set
        self.test_set = test_set
        self.batch_size = args.batch_size
        self.device = device
        self.args = args
        self.meta = meta

        self.test_iter = make_iterable(
            self.test_set,
            self.device,
            batch_size=self.batch_size,
            train=False,
        )

    def al_loop(
        self,
        create_model_fn,
        optimizer,
        criterion,
        warm_start_size,
        query_size,
        interpreters,
        correlations,
    ):
        score_list = []

        # Initialize label mask.
        lab_mask = np.full(len(self.train_set), False)
        # TODO: stratified warm start
        random_inds = np.random.choice(
            len(self.train_set), warm_start_size, replace=False
        )
        lab_mask[random_inds] = True

        al_epoch = 0
        # Loop until all of the labels are used up.
        while not lab_mask.sum() == lab_mask.size:
            print(f"AL epoch: {al_epoch}")

            # 1. train model with labeled data: fine-tune vs. re-train
            print(f"Training on {lab_mask.sum()}/{lab_mask.size} labeled data...")
            model = create_model_fn(self.args, self.meta)
            model.to(self.device)
            result_dict_train = self._train_model(model, lab_mask, optimizer, criterion)

            # 2. evaluate model (test set)
            eval_result_dict = self._evaluate_model(model)

            # 3. Retrieve active sample.
            print("Retrieving AL sample...")
            unlab_inds, *_ = np.where(~lab_mask)
            selected_inds = self.sampler.query(
                query_size=query_size,
                unlab_inds=unlab_inds,
                model=model,
                lab_mask=lab_mask,
            )
            lab_mask[selected_inds] = True

            # 4. calculate intepretability metrics
            print("Calculating intepretability metrics...")
            intepret_result_dict = interpret_evaluate(
                interpreters, model, self.test_iter, self.args, self.meta
            )
            scores = pairwise_correlation(
                intepret_result_dict["attributions"], correlations
            )
            score_list.append(scores)
            print("Interpretability scores", scores)

            al_epoch += 1

        return score_list

    def _train_model(self, model, lab_mask, optimizer, criterion):
        model.train()

        data = make_iterable(
            self.train_set,
            self.device,
            batch_size=self.batch_size,
            train=True,
            indices=lab_mask,
        )
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )
        total_loss = 0.0

        for batch_num, batch in enumerate(data):

            t = time.time()
            # Unpack batch & cast to device
            (x, lengths), y = batch.text, batch.label

            y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)

            logits, _ = model(x, lengths)

            # Bookkeeping and cast label to float
            accuracy, confusion_matrix = ActiveLearner.update_stats(
                accuracy, confusion_matrix, logits, y
            )
            if logits.shape[-1] == 1:
                # binary cross entropy, cast labels to float
                y = y.type(torch.float)

            loss = criterion(logits.view(-1, self.meta.num_targets).squeeze(), y)

            total_loss += float(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
            optimizer.step()

            print(
                "[Batch]: {}/{} in {:.5f} seconds".format(
                    batch_num, len(data), time.time() - t
                ),
                end="\r",
                flush=True,
            )
        result_dict = {"loss": total_loss / len(data) / data.batch_size}
        return result_dict

    def _evaluate_model(self, model):
        model.eval()

        data = self.test_iter
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )

        with torch.inference_mode():
            for batch_num, batch in enumerate(data):
                # if batch_num > 100: break # checking beer imitation

                t = time.time()
                model.zero_grad()

                # Unpack batch & cast to device
                (x, lengths), y = batch.text, batch.label

                y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)

                logits, _ = model(x, lengths)
                # attn = return_dict['attn'].squeeze()

                # Bookkeeping and cast label to float
                accuracy, confusion_matrix = ActiveLearner.update_stats(
                    accuracy, confusion_matrix, logits, y
                )
                if logits.shape[-1] == 1:
                    # binary cross entropy, cast labels to float
                    y = y.type(torch.float)

                print(
                    "[Batch]: {}/{} in {:.5f} seconds".format(
                        batch_num, len(data), time.time() - t
                    ),
                    end="\r",
                    flush=True,
                )

        print(
            "[Accuracy]: {}/{} : {:.3f}%".format(
                accuracy,
                len(data) * data.batch_size,
                accuracy / len(data) / data.batch_size * 100,
            )
        )
        print(confusion_matrix)
        result_dict = {"loss": 0.0}
        return result_dict

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
                confusion_matrix[int(i), int(j)] += 1

        return accuracy + correct, confusion_matrix
