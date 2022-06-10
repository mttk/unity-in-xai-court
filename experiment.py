import time
import torch
import logging
import numpy as np
from sklearn.metrics import f1_score

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
        self.N_samples = args.perturb_samples

        test_batch_size = 4 if args.model_name == "DBERT" else 32

        self.test_iter = make_iterable(
            self.test_set,
            self.device,
            batch_size=test_batch_size,
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
            train=True,  # [Debug] was False
            # indices=indices,
        )
        # optimizer = torch.optim.AdamW(
        #     filter(lambda p: p.requires_grad, model.parameters()),
        #     self.args.lr,
        #     weight_decay=self.args.l2,
        # )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            self.args.lr,
            weight_decay=self.args.l2,
        )
        # Prepare interpreters
        if self.args.model_name != "vanilla-DBERT":
            interpreters = {
                i: get_interpreter(i)(model) for i in self.args.interpreters
            }
        else:
            # Not needed as variable is not used for van-dbert, but just in case
            interpreters = None

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
            if self.args.model_name != "vanilla-DBERT":
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
                attributions_results.append(intepret_result_dict["attributions"])
                logging.info("Interpretability scores", scores)
            else:
                # Make dummy scores and correlations sos code deosn't break
                scores = []
                raw_correlations = []
                attributions_results.append([])

            agreement_results.append(scores)
            correlation_results.append(raw_correlations)

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

        # 3) MC-Evaluate local smoothness
        smoothness_results = self._evaluate_smoothness(model)

        # 4) Store results.
        results = {}
        results["train"] = train_results
        results["eval"] = eval_results
        results["agreement"] = agreement_results
        results["attributions"] = attributions_results
        results["correlation"] = correlation_results
        results["cartography"] = {}
        results["cartography"]["train"] = cartography_results["train"]
        results["cartography"]["test"] = cartography_results["test"]
        results["smoothness"] = {}
        results["smoothness"]["logit_variance"] = smoothness_results["logit_variance"]

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

            optimizer.zero_grad()

            # Unpack batch & cast to device
            (x, lengths), y = batch.text, batch.label

            y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)
            y_true_list.append(y)

            if self.args.model_name != "vanilla-DBERT":
                # Vanilla distilBert returns only logits
                logits, return_dict = model(x, lengths)
            else:
                maxlen = lengths.max()
                mask = (
                    torch.arange(maxlen, device=lengths.device)[None, :]
                    >= lengths[:, None]
                )
                logits = model(x, attention_mask=mask).logits

            logit_list.append(logits)

            # Bookkeeping and cast label to float
            accuracy, confusion_matrix = Experiment.update_stats(
                accuracy, confusion_matrix, logits, y
            )
            if logits.shape[-1] == 1:
                # binary cross entropy, cast labels to float
                y = y.type(torch.float)

            # loss = criterion(logits, y)
            loss = criterion(logits.view(-1, self.meta.num_targets).squeeze(), y)

            # Perform weight tying if required
            if self.args.tying > 0.0:  #  and self.args.model_name == "JWA"
                e = return_dict["embeddings"].transpose(0, 1)  # BxTxH -> TxBxH
                h = return_dict["hiddens"]  # TxBxH

                # print(h.shape, e.shape)
                reg = (h - e).norm(2, dim=-1).mean()
                loss += self.args.tying * reg

            # Perform conicity regularization if required
            if self.args.conicity > 0.0:  #  and self.args.model_name == "JWA"
                h = return_dict["hiddens"].transpose(0, 1)  # [BxTxH]
                # Compute mean hidden across T
                h_mu = h.mean(1, keepdim=True)  # [Bx1xH]
                # Compute ATM
                cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(h, h_mu)  # [BxT]
                conicity = cosine.mean()  # Conicity = average ATM, dim=[1]

                loss += self.args.conicity * conicity

            total_loss += float(loss)

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

        logging.info(f"[Train accuracy]: {accuracy/len(self.train_set)*100:.3f}%")

        return result_dict, logit_tensor, y_true

    def _evaluate_model(self, model):
        model.eval()

        data = self.test_iter
        accuracy, confusion_matrix = 0, np.zeros(
            (self.meta.num_labels, self.meta.num_labels), dtype=int
        )

        logit_list = []
        y_true_list = []
        with torch.inference_mode():
            for batch_num, batch in enumerate(data):
                t = time.time()

                # Unpack batch & cast to device
                (x, lengths), y = batch.text, batch.label

                y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)
                y_true_list.append(y.reshape(1).cpu() if y.dim() == 0 else y.cpu())

                if self.args.model_name != "vanilla-DBERT":
                    logits, _ = model(x, lengths)
                else:
                    maxlen = lengths.max()
                    mask = (
                        torch.arange(maxlen, device=lengths.device)[None, :]
                        >= lengths[:, None]
                    )
                    logits = model(x, attention_mask=mask).logits
                logit_list.append(logits.cpu())

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

        logits = torch.cat(logit_list)
        y_true = torch.cat(y_true_list)
        probs = logits_to_probs(logits)
        y_pred = torch.argmax(probs, dim=1)
        f1_average = "binary" if torch.unique(y_true).numel() == 2 else "micro"
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average=f1_average)

        logging.info(
            "[Test accuracy]: {}/{} : {:.3f}%".format(
                accuracy,
                len(self.test_set),
                accuracy / len(self.test_set) * 100,
            )
        )
        logging.info(f"[F1]: {f1:.3f}")
        logging.info(confusion_matrix)

        result_dict = {"accuracy": accuracy / len(self.test_set), "f1": f1}
        return result_dict

    def _evaluate_smoothness(self, model):
        model.eval()

        data = self.test_iter

        logit_list = []
        d_logit_list = []
        y_true_list = []
        with torch.inference_mode():
            for batch_num, batch in enumerate(data):
                t = time.time()

                # Unpack batch & cast to device
                (x, lengths), y = batch.text, batch.label

                y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)
                y_true_list.append(y.reshape(1).cpu() if y.dim() == 0 else y.cpu())

                hidden = model.get_encoded(x, lengths) # [BxH]
                true_logits = model.decode(hidden, output_dict={})
                # Perturbation experiment: explore local space around hidden representation
                hidden_l2 = torch.norm(hidden, p=2, dim=-1) # [Bx1]
                noise_scale = torch.mean(hidden_l2) / 4.

                d_logits = []
                for _ in range(self.N_samples):
                    # Consider increasing norm gradually
                    d_hidden = torch.randn(hidden.shape, device=noise_scale.device) * noise_scale
                    d_logit = model.decode(hidden + d_hidden, output_dict={})
                    d_logits.append(d_logit.squeeze().detach().cpu()) # 32 x 1

                print("Logits shape", torch.cat(d_logits).shape)
                print("Logits shape", torch.stack(d_logits).shape)
                
                d_logit_list.append(
                        torch.std(torch.stack(d_logits), axis=0).mean()
                    )
                print(d_logit_list[-1].shape)
                logit_list.append(true_logits.cpu())

                print(
                    "[Batch]: {}/{} in {:.5f} seconds".format(
                        batch_num, len(data), time.time() - t
                    ),
                    end="\r",
                    flush=True,
                )

        logits = torch.cat(logit_list)
        d_logits = torch.stack(d_logit_list)
        y_true = torch.cat(y_true_list)

        logging.info(
            "[Logit variance wrt noise]: {:.3f}".format(
                torch.mean(d_logits) # Average variance over all instances over N_samples
            )
        )

        result_dict = {"logit_variance": d_logits}
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

                if self.args.model_name != "vanilla-DBERT":
                    logits, _ = model(x, lengths)
                else:
                    maxlen = lengths.max()
                    mask = (
                        torch.arange(maxlen, device=lengths.device)[None, :]
                        >= lengths[:, None]
                    )
                    logits = model(x, attention_mask=mask).logits
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
            _, lengths = batch.text
            len_list.append(lengths.cpu())

        return torch.cat(len_list)

    def get_id_mapping(self):
        mapping_list = []
        for batch in self.test_iter:
            mapping_list.extend(batch.id)

        return [int(id) for ids in mapping_list for id in ids]
