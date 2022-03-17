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


class Experiment:
    def __init__(self, train_set, test_set, device, args, meta):
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

        self.test_lengths = self.extract_lengths()
        self.id_mapping = self.get_id_mapping()

    def run(
        self,
        create_model_fn,
        criterion,
        warm_start_size,
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

        # 1) Train model on labeled data.
        logging.info(f"Training on {lab_mask.sum()}/{lab_mask.size} labeled data...")
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
        interpreters = {i: get_interpreter(i)(model) for i in self.args.interpreters}

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

        # 3) Store results.
        results = {}
        results["train"] = train_results
        results["eval"] = eval_results
        results["agreement"] = agreement_results
        results["attributions"] = attributions_results
        results["correlation"] = correlation_results
        results["cartography"] = {}
        results["cartography"]["train"] = cartography_results["train"]
        results["cartography"]["test"] = cartography_results["test"]

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

            logits, return_dict = model(x, lengths)
            logit_list.append(logits)

            # Bookkeeping and cast label to float
            accuracy, confusion_matrix = Experiment.update_stats(
                accuracy, confusion_matrix, logits, y
            )
            if logits.shape[-1] == 1:
                # binary cross entropy, cast labels to float
                y = y.type(torch.float)

            loss = criterion(logits.view(-1, self.meta.num_targets).squeeze(), y)

            # Perform weight tying if required
            if self.args.tying > 0.0 and self.args.model_name == "JWA":
                e = return_dict["embeddings"].transpose(0, 1)  # BxTxH -> TxBxH
                h = return_dict["hiddens"]  # TxBxH

                # print(h.shape, e.shape)
                reg = (h - e).norm(2, dim=-1).mean()
                loss += self.args.tying * reg

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
                accuracy, confusion_matrix = Experiment.update_stats(
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

    def extract_lengths(self):
        len_list = []
        for batch in self.test_iter:
            _, lenghts = batch.text
            len_list.append(lenghts.cpu())

        return torch.cat(len_list)

    def get_id_mapping(self):
        mapping_list = []
        for batch in self.test_iter:
            mapping_list.extend(batch.id)

        return [int(id) for ids in mapping_list for id in ids]
