import numpy as np
import logging
import torch
import time

from train import *
from dataloaders import *

from util import logits_to_probs


def compute_forgetfulness(epochwise_tensor):
    """
    Given a epoch-wise trend of train predictions, compute frequency with which
    an example is forgotten, i.e. predicted incorrectly _after_ being predicted correctly.
    Based on: https://arxiv.org/abs/1812.05159
    """

    out = []

    datawise_tensor = epochwise_tensor.transpose(0, 1)
    for correctness_trend in datawise_tensor:
        if not any(
            correctness_trend
        ):  # Example is never predicted correctly, or learnt!
            out.append(torch.tensor(1000))
            continue
        learnt = False  # Predicted correctly in the current epoch.
        times_forgotten = 0
        for is_correct in correctness_trend:
            if (not learnt and not is_correct) or (learnt and is_correct):
                # Nothing changed.
                continue
            elif learnt and not is_correct:
                # Forgot after learning at some point!
                learnt = False
                times_forgotten += 1
            elif not learnt and is_correct:
                # Learnt!
                learnt = True
        out.append(torch.tensor(times_forgotten))

    return torch.stack(out)


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
        if warm_start_size == -1:
            lab_mask = np.full(len(self.train_set), True)
        else:
            # Initialize label mask.
            lab_mask = np.full(len(self.train_set), False)
            # TODO: stratified warm start
            random_inds = np.random.choice(
                len(self.train_set), warm_start_size, replace=False
            )
            lab_mask[random_inds] = True

        al_epochs = self.args.al_epochs
        if al_epochs == -1:
            unlab_size = self.args.max_train_size - lab_mask.sum()
            al_epochs = np.int(np.ceil(unlab_size / query_size)) + 1

        results = {
            "train": [],
            "eval": [],
            "agreement": [],
            "attributions": [],
            "correlation": [],
            "labeled": [],
            "cartography": {"train": [], "test": []},
        }
        for al_epoch in range(1, al_epochs + 1):
            logging.info(f"AL epoch: {al_epoch}/{al_epochs}")
            results["labeled"].append(lab_mask.sum())

            # 1) Train model on labeled data.
            logging.info(
                f"Training on {lab_mask.sum()}/{lab_mask.size} labeled data..."
            )
            # Create new model: re-train scenario.
            model = create_model_fn(self.args, self.meta)
            model.to(self.device)

            indices, *_ = np.where(lab_mask)
            train_iter = make_iterable(
                self.train_set,
                self.device,
                batch_size=self.batch_size,
                train=False,
                indices=indices,
            )
            optimizer = torch.optim.Adam(
                model.parameters(), self.args.lr, weight_decay=self.args.l2
            )
            # Prepare interpreters
            interpreters = {
                i: get_interpreter(i)(model) for i in sorted(self.args.interpreters)
            }
            train_results = []
            eval_results = []
            agreement_results = []
            attributions_results = []
            correlation_results = []
            cartography_trends = {
                "train": {"is_correct": [], "true_probs": []},
                "test": {"is_correct": [], "true_probs": []},
            }
            for epoch in range(1, self.args.epochs + 1):
                logging.info(f"Training epoch: {epoch}/{self.args.epochs}")
                # a) Train for one epoch
                result_dict_train, logits, y_true = self._train_model(
                    model, optimizer, criterion, train_iter
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
                scores, raw_correlations = pairwise_correlation(
                    intepret_result_dict["attributions"], correlations
                )
                agreement_results.append(scores)
                correlation_results.append(raw_correlations)
                attributions_results.append(intepret_result_dict["attributions"])
                logging.info("Interpretability scores", scores)

                # d) Calculate epoch cartography
                logging.info("Calculating cartography...")
                #   i) train set
                is_correct, true_probs = self._cartography_epoch_train(logits, y_true)
                cartography_trends["train"]["is_correct"].append(is_correct)
                cartography_trends["train"]["true_probs"].append(true_probs)

                #   ii) test set
                is_correct, true_probs = self._cartography_epoch_test(model)
                cartography_trends["test"]["is_correct"].append(is_correct)
                cartography_trends["test"]["true_probs"].append(true_probs)

            # 2) Dataset cartography
            logging.info("Computing dataset cartography...")
            cartography_results = {}
            cartography_results["train"] = self._compute_cartography(
                cartography_trends["train"]
            )
            cartography_results["test"] = self._compute_cartography(
                cartography_trends["test"]
            )

            # 3) Retrieve active sample.
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
            results["attributions"].append(attributions_results)
            results["correlation"].append(correlation_results)
            results["cartography"]["train"].append(cartography_results["train"])
            results["cartography"]["test"].append(cartography_results["test"])

        return results

    def _train_model(self, model, optimizer, criterion, train_iter):
        model.train()

        total_loss = 0.0
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )

        logit_list = []
        y_true_list = []
        for batch_num, batch in enumerate(train_iter):
            t = time.time()

            # Unpack batch & cast to device
            (x, lengths), y = batch.text, batch.label

            y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)
            y_true_list.append(y)

            logits, _ = model(x, lengths)
            logit_list.append(logits)

            # Bookkeeping and cast label to float
            accuracy, confusion_matrix = ActiveLearner.update_stats(
                accuracy, confusion_matrix, logits, y
            )
            if logits.shape[-1] == 1:
                # binary cross entropy, cast labels to float
                y = y.type(torch.float)

            loss = criterion(logits.view(-1, self.meta.num_targets).squeeze(), y)

            total_loss += float(loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
            optimizer.step()

            print(
                "[Batch]: {}/{} in {:.5f} seconds".format(
                    batch_num, len(train_iter), time.time() - t
                ),
                end="\r",
                flush=True,
            )

        loss = total_loss / len(train_iter)
        result_dict = {"loss": loss}
        logit_tensor = torch.cat(logit_list)
        y_true = torch.cat(y_true_list)
        return result_dict, logit_tensor, y_true

    def _evaluate_model(self, model):
        model.eval()

        data = self.test_iter
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )

        with torch.inference_mode():
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
        if len(y.shape) == 0:
            # only one element here? is this even possible?
            confusion_matrix[y, max_ind] += 1
        else:
            for j, i in zip(max_ind, y):
                confusion_matrix[int(i), int(j)] += 1

        return accuracy + correct, confusion_matrix

    def _cartography_epoch_train(self, logits, y_true):
        logits = logits.cpu()
        y_true = y_true.cpu()
        probs = logits_to_probs(logits)
        true_probs = probs.gather(dim=1, index=y_true.unsqueeze(dim=1)).squeeze()
        y_pred = torch.argmax(probs, dim=1)
        is_correct = y_pred == y_true

        return is_correct, true_probs

    def _cartography_epoch_test(self, model):
        model.train()

        data = self.test_iter

        logit_list = []
        y_true_list = []
        with torch.inference_mode():
            for batch_num, batch in enumerate(data):
                # Unpack batch & cast to device
                (x, lengths), y = batch.text, batch.label

                y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)
                y_true_list.append(y.cpu())

                logits, _ = model(x, lengths)
                logit_list.append(logits.cpu())

        logit_tensor = torch.cat(logit_list)
        y_true = torch.cat(y_true_list)
        probs = logits_to_probs(logit_tensor)
        true_probs = probs.gather(dim=1, index=y_true.unsqueeze(dim=1)).squeeze()
        y_pred = torch.argmax(probs, dim=1)
        is_correct = y_pred == y_true

        return is_correct, true_probs

    def _compute_cartography(self, trends):
        cartography_results = {}

        is_correct = torch.stack(trends["is_correct"])
        true_probs = torch.stack(trends["true_probs"])

        cartography_results["correctness"] = (
            is_correct.sum(dim=0).squeeze().detach().numpy()
        )
        cartography_results["confidence"] = (
            true_probs.mean(dim=0).squeeze().detach().numpy()
        )
        cartography_results["variability"] = (
            true_probs.std(dim=0).squeeze().detach().numpy()
        )
        cartography_results["forgetfulness"] = compute_forgetfulness(is_correct).numpy()
        conf = cartography_results["confidence"]
        cartography_results["threshold_closeness"] = conf * (1 - conf)

        return cartography_results
