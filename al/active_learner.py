import numpy as np
import logging
import torch
import time

from train import *
from dataloaders import *


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
        criterion,
        warm_start_size,
        query_size,
        correlations,
    ):
        # Initialize label mask.
        lab_mask = np.full(len(self.train_set), False)
        # TODO: stratified warm start
        random_inds = np.random.choice(
            len(self.train_set), warm_start_size, replace=False
        )
        lab_mask[random_inds] = True

        al_epochs = self.args.al_epochs
        if al_epochs == -1:
            unlab_size = (~lab_mask).sum()
            al_epochs = np.int(np.ceil(unlab_size / query_size)) + 1

        results = {"train": [], "eval": [], "agreement": [], "labeled": []}
        for al_epoch in range(1, al_epochs + 1):
            logging.info(f"AL epoch: {al_epoch}/{al_epochs}")
            results["labeled"].append(lab_mask.sum())

            # 1) Train model with labeled data: fine-tune vs. re-train
            logging.info(
                f"Training on {lab_mask.sum()}/{lab_mask.size} labeled data..."
            )
            # Create new model: re-train scenario.
            model = create_model_fn(self.args, self.meta)
            model.to(self.device)
            optimizer = torch.optim.Adam(
                model.parameters(), self.args.lr, weight_decay=self.args.l2
            )
            # Prepare interpreters.
            interpreters = {
                i: get_interpreter(i)(model) for i in sorted(self.args.interpreters)
            }
            train_results = []
            eval_results = []
            agreement_results = []
            for epoch in range(1, self.args.epochs + 1):
                logging.info(f"Training epoch: {epoch}/{self.args.epochs}")
                # a) Train for one epoch
                result_dict_train = self._train_model(
                    model, lab_mask, optimizer, criterion
                )
                train_results.append(result_dict_train)

                # b) Evaluate model (test set)
                eval_result_dict = self._evaluate_model(model)
                eval_results.append(eval_result_dict)

                # c) Calculate intepretability metrics
                logging.info("Calculating intepretability metrics...")
                intepret_result_dict = interpret_evaluate(
                    interpreters,
                    model,
                    self.test_iter,
                    self.args,
                    self.meta,
                    use_rationales=False,
                )
                scores = pairwise_correlation(
                    intepret_result_dict["attributions"], correlations
                )
                agreement_results.append(scores)
                logging.info("Interpretability scores", scores)

            # 2) Retrieve active sample.
            if not lab_mask.all():
                logging.info("Retrieving AL sample...")
                lab_inds, *_ = np.where(lab_mask)
                unlab_inds, *_ = np.where(~lab_mask)
                if len(unlab_inds) <= query_size:
                    selected_inds = unlab_inds
                else:
                    model.eval()
                    selected_inds = self.sampler.query(
                        query_size=query_size,
                        unlab_inds=unlab_inds,
                        lab_inds=lab_inds,
                        model=model,
                        lab_mask=lab_mask,
                        num_labels=self.meta.num_labels,
                        num_targets=self.meta.num_targets,
                        criterion=criterion,
                    )

                lab_mask[selected_inds] = True
                logging.info(f"{len(selected_inds)} data points selected.")

            # 3) Store results.
            results["train"].append(train_results)
            results["eval"].append(eval_results)
            results["agreement"].append(agreement_results)

        return results

    def _train_model(self, model, lab_mask, optimizer, criterion):
        model.train()

        indices, *_ = np.where(lab_mask)
        data = make_iterable(
            self.train_set,
            self.device,
            batch_size=self.batch_size,
            train=True,
            indices=indices,
        )
        total_loss = 0.0
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )

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

        loss = total_loss / len(data) / data.batch_size
        result_dict = {"loss": loss}
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

        logging.info(
            "[Accuracy]: {}/{} : {:.3f}%".format(
                accuracy,
                len(self.test_set),
                accuracy / len(self.test_set) * 100,
            )
        )
        logging.info(confusion_matrix)
        result_dict = {"accuracy": accuracy / len(self.test_set)}
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
