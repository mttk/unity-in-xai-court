import argparse
from opcode import hasconst
import os, sys
import time
import copy
import itertools

from datetime import datetime
from pprint import pprint
from attr import attr

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from podium import BucketIterator
from transformers import DistilBertTokenizer

from util import Config
from dataloaders import *
from model import *
from distillbert import *
from interpret import *
from correlation_measures import *


from sklearn.metrics import average_precision_score

word_vector_files = {"glove": os.path.expanduser("~/data/vectors/glove.840B.300d.txt")}

dataset_loaders = {
    "IMDB": load_imdb,
    "IMDB-rationale": load_imdb_rationale,
    "IMDB-sentences": load_imdb_sentences,
    "TSE": load_tse,
    "TREC": load_trec,
    "SST": load_sst,
    "SUBJ": load_subj,
    "JWA-SST": load_jwa_sst,
    "POL": load_polarity,
    "COLA": load_cola,
}

models = {
    "JWA": JWAttentionClassifier,
    "MLP": MLP,
    "DBERT": DistilBertForSequenceClassification.from_huggingface_model_name,
    "vanilla-DBERT": make_vanilla_distilbert,
}

TRANSFORMERS = ["DBERT", "vanilla-DBERT"]


def make_parser():
    parser = argparse.ArgumentParser(description="PyTorch RNN Classifier w/ attention")
    parser.add_argument(
        "--data",
        type=str,
        default="IMDB",
        help="Data corpus: [IMDB, IMDB-rationale, TSE, TREC, SST]",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="JWA",
        help="Model: [JWA, MLP, DBERT, vanilla-DBERT]",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained transformer model to load",
    )

    # Representation tying arguments
    parser.add_argument(
        "--tying", type=float, default=0.0, help="Weight tying lambda (if applicable)"
    )
    # Conicity regularization arguments
    parser.add_argument(
        "--conicity",
        type=float,
        default=0.0,
        help="Conicity regularization lambda (if applicable)",
    )
    # JWA arguments
    parser.add_argument(
        "--rnn_type",
        type=str,
        default="LSTM",
        help="type of recurrent net [LSTM, GRU, MHA]",
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="nqadd",
        help="attention type [dot, add, nqdot, nqadd], default = nqadd",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=300,
        help="size of word embeddings [Uses pretrained on 300]",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=150,
        help="number of hidden units for the encoder",
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, help="number of layers of the encoder"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument(
        "--vectors",
        type=str,
        default="glove",
        help="Pretrained vectors to use [glove, fasttext]",
    )
    parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=5, help="upper epoch limit")
    parser.add_argument(
        "--batch_size", type=int, default=32, metavar="N", help="batch size"
    )
    parser.add_argument("--dropout", type=float, default=0, help="dropout")
    parser.add_argument(
        "--l2", type=float, default=0, help="l2 regularization (weight decay)"
    )
    parser.add_argument("--bi", action="store_true", help="Bidirectional encoder")
    parser.add_argument("--freeze", action="store_true", help="Freeze embeddings")

    # DistillBERT arguments
    parser.add_argument(
        "--seq_classif_dropout",
        type=float,
        default=0.1,
        help="Decoder dropout after *BERT encoding",
    )

    # Interpreters & corr measures
    parser.add_argument(
        "--interpreters",
        nargs="+",
        default=["deeplift", "grad-shap", "deeplift-shap", "int-grad"],
        choices=["deeplift", "grad-shap", "deeplift-shap", "int-grad", "lime"],
        help="Specify a list of interpreters.",
    )
    parser.add_argument(
        "--correlation_measures",
        nargs="+",
        default=["kendall-tau", "pearson", "jsd"],
        choices=["kendall-tau", "pearson", "jsd"],
        help="Specify a list of correlation metrics.",
    )

    # Vocab specific arguments
    parser.add_argument(
        "--max_vocab", type=int, default=10000, help="maximum size of vocabulary"
    )
    parser.add_argument(
        "--min_freq", type=int, default=5, help="minimum word frequency"
    )
    parser.add_argument(
        "--max_len", type=int, default=200, help="maximum length of input sequence"
    )

    # Repeat experiments
    parser.add_argument(
        "--repeat", type=int, default=5, help="number of times to repeat training"
    )

    # Save directory
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Directory for storing results.",
    )

    # Gpu based arguments
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="Gpu to use for experiments (-1 means no GPU)",
    )

    # Storing & loading arguments
    parser.add_argument(
        "--save",
        type=str,
        default="chkp/",
        help="Folder to store final model (or model with best valid perf) in",
    )
    parser.add_argument(
        "--log", type=str, default="tb_log/", help="Folder to store tensorboard logs in"
    )
    parser.add_argument(
        "--restore", type=str, default="", help="File to restore model from"
    )

    # Model "unlearning" -- removing train instances with largest disagreement
    parser.add_argument(
        "--ul-epochs",
        type=int,
        default=-1,
        help="Number of UL epochs (-1 uses the whole train set)",
    )

    # Active learning arguments
    parser.add_argument(
        "--warm-start-size", type=int, default=-1, help="Initial AL batch size."
    )

    # Number of samples for perturbation experiment (smoothness approx)
    parser.add_argument(
        "--perturb-samples", type=int, default=1000, help="Perturbation samples."
    )

    return parser.parse_args()


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


def correct_for_missing(indices, mask):
    # Since some instances are missing from the dataset, we need
    # to align the indices wrt masked positions
    offset = np.cumsum(~mask)
    aligned_indices = [i - offset[i] for i in indices]
    return aligned_indices


def initialize_model(args, meta):
    # 1. Construct encoder (shared in any case)
    # 2. Construct decoder / decoders
    if not hasattr(meta, "embeddings") and args.model_name not in [
        "DBERT",
        "vanilla-DBERT",
    ]:
        # Cache embeddings
        meta.embeddings = torch.tensor(load_embeddings(meta.vocab, name="glove"))
    model_cls = models[args.model_name]
    model = model_cls(args, meta)

    return model


def rationale_correlation(importance_dictionary, rationales):
    # Mean Average Precisions
    importance_rationale_maps = {}

    for method_name, importances in importance_dictionary.items():
        aps = []
        for inst_importance, inst_rationale in zip(importances, rationales):
            aps.append(average_precision_score(inst_rationale, inst_importance))
        importance_rationale_maps[method_name] = np.mean(aps)
    return importance_rationale_maps


def pairwise_correlation(importance_dictionary, correlation_measures):
    # importance_dictionary -> [method_name: list_of_values_for_instances]

    N = len(importance_dictionary)
    K = len(correlation_measures)
    all_scores = {}
    all_raw_correlations = {}

    for corr_idx, corr in enumerate(correlation_measures):
        scores = {}  # pairwise for each correlation
        raw_correlations = {}
        for i, k_i in enumerate(importance_dictionary):
            for j, k_j in enumerate(importance_dictionary):
                corrs = []

                if k_i == k_j or (k_i, k_j) in scores or (k_j, k_i) in scores:
                    # Account for same & symmetry
                    continue

                for inst_i, inst_j in zip(
                    importance_dictionary[k_i], importance_dictionary[k_j]
                ):
                    r = corr.correlation(inst_i, inst_j)
                    corrs.append(r[corr.id].correlation)
                scores[(k_i, k_j)] = np.mean(corrs)
                raw_correlations[(k_i, k_j)] = corrs
        all_scores[corr._id] = scores
        all_raw_correlations[corr._id] = raw_correlations

    pprint(all_scores)
    return all_scores, all_raw_correlations


def evaluate(model, data, args, meta):
    model.eval()

    accuracy, confusion_matrix = 0, np.zeros(
        (meta.num_labels, meta.num_labels), dtype=int
    )

    with torch.inference_mode():
        for batch_num, batch in enumerate(data):

            t = time.time()

            # Unpack batch & cast to device
            (x, lengths), y = batch.text, batch.label

            y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)

            logits, return_dict = model(x, lengths)
            # attn = return_dict['attn'].squeeze()

            # Bookkeeping and cast label to float
            accuracy, confusion_matrix = update_stats(
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


# For regression & classification
def train(model, data, optimizer, criterion, args, meta):
    model.train()

    accuracy, confusion_matrix = 0, np.zeros(
        (meta.num_labels, meta.num_labels), dtype=int
    )
    total_loss = 0.0

    for batch_num, batch in enumerate(data):

        t = time.time()
        # Unpack batch & cast to device
        (x, lengths), y = batch.text, batch.label
        # print("Lens", lengths)

        y = y.squeeze()  # y needs to be a 1D tensor for xent(batch_size)

        logits, return_dict = model(x, lengths)

        # Bookkeeping and cast label to float
        accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
        if logits.shape[-1] == 1:
            # binary cross entropy, cast labels to float
            y = y.type(torch.float)

        # print(logits.shape, y.shape)
        loss = criterion(logits.view(-1, meta.num_targets).squeeze(), y)

        # Perform weight tying if required
        if args.tying > 0.0:  #  and args.model_name == "JWA"
            e = return_dict["embeddings"].transpose(0, 1)  # BxTxH -> TxBxH
            h = return_dict["hiddens"]  # TxBxH

            # print(h.shape, e.shape)
            reg = (h - e).norm(2, dim=-1).mean()
            loss += args.tying * reg

        if args.conicity > 0.0:  #  and args.model_name == "JWA"
            h = return_dict["hiddens"].transpose(0, 1)  # [BxTxH]
            # Compute mean hidden across T
            h_mu = h.mean(1, keepdim=True)  # [Bx1xH]
            # Compute ATM
            cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(h, h_mu)  # [BxT]
            # print(cosine.shape)
            conicity = cosine.mean()  # Conicity = average ATM, dim=[1]
            # print(conicity)
            loss += args.conicity * conicity

        total_loss += float(loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
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


def interpret_evaluate(interpreters, model, data, args, meta, use_rationales=True):
    model.train()

    attributions = {k: [] for k in interpreters}
    rationales = []  # Will be of variable length (slice based on lengths)

    for batch_num, batch in enumerate(data):

        t = time.time()

        # Unpack batch & cast to device
        (x, lengths), y = batch.text, batch.label
        # print(x.shape)
        if use_rationales:
            rationale = (
                batch.rationale.detach().cpu().numpy()
            )  # These are padded to the batch length
            rationales.extend([r[:l] for r, l in zip(rationale, lengths)])

        for k, interpreter in interpreters.items():
            # print(lengths.shape)
            labels = (
                None
                if meta.num_targets == 1 and args.model_name != "DBERT"
                else y.squeeze()
            )
            batch_attributions = interpreter.interpret(x, lengths, labels=labels)
            batch_attributions = batch_attributions.detach().cpu().numpy()
            torch.cuda.empty_cache()
            # print(batch_attributions)

            # attributions[k].extend(batch_attributions)
            # Select only non-padding attributions
            attributions[k].extend([a[:l] for a, l in zip(batch_attributions, lengths)])

        print(
            "[Batch]: {}/{} in {:.5f} seconds".format(
                batch_num, len(data), time.time() - t
            ),
            end="\r",
            flush=True,
        )

        model.zero_grad()

    result_dict = {"attributions": attributions, "rationales": rationales}

    # for k, v in attributions.items():
    #  print(k, v[0].shape)

    return result_dict


def faithfulness(model, attributions, data):
    faithfulness = {k: [] for k, v in attributions.items()}
    with torch.no_grad():
        for batch in data:
            for k, v in attributions.items():
                attr = attributions[k]
                print(attr[0].shape, attr[1].shape)
                for p in range(10, 100 + 10, 10):
                    inputs, _ = batch.text
                    tokens = inputs.clone()
                    print(tokens)
                    model(tokens)


def experiment(args, meta, train_dataset, val_dataset, test_dataset, restore=None):
    # Input: model arguments and dataset splits, whether to restore the model
    # Constructor delegated to args selector of attention

    # Just to be safe
    args = copy.deepcopy(args)

    cuda = torch.cuda.is_available() and args.gpu != -1
    device = torch.device("cpu") if not cuda else torch.device(f"cuda:{args.gpu}")
    # Setup the loss fn
    if meta.num_labels == 2:
        # Binary classification
        criterion = nn.BCEWithLogitsLoss()
        meta.num_targets = 1
    else:
        # Multiclass classification
        criterion = nn.CrossEntropyLoss()
        meta.num_targets = meta.num_labels

    # Initialize model
    model = initialize_model(args, meta)
    model.to(device)

    # TODO: use AdamW for DBERT as default
    if args.model_name in TRANSFORMERS:
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.l2)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.l2)

    # train_iter_noshuf = make_iterable(train_dataset, device, batch_size=args.batch_size)
    train_iter = make_iterable(
        train_dataset, device, batch_size=args.batch_size, train=True
    )
    val_iter = make_iterable(val_dataset, device, batch_size=args.batch_size)
    test_iter = make_iterable(test_dataset, device, batch_size=args.batch_size)

    if restore is not None:
        m, o, c = restore
        print(f"Loading model from {args.restore}", flush=True)
        # Only restore the main model
        model.load_state_dict(m)
        optimizer.load_state_dict(o)
        criterion.load_state_dict(c)

    # Construct interpreters
    interpreters = {}
    for i in args.interpreters:
        if i == "int-grad":
            get_interpreter(i)(model, internal_batch_size=32)
        else:
            get_interpreter(i)(model)
    print(f"Interpreters: {' '.join(list(interpreters.keys()))}")

    # Construct correlation metrics
    correlations = [get_corr(key)() for key in args.correlation_measures]
    print(f"Correlation measures: {correlations}")

    # TODO: check if rationales exist in the dataset
    use_rationales = True if args.data in ["IMDB-rationale"] else False

    loss = 0.0
    # The actual training loop
    try:
        best_valid_loss = None
        best_valid_epoch = 0
        best_model = copy.deepcopy(model)

        for epoch in range(1, args.epochs + 1):

            train(model, train_iter, optimizer, criterion, args, meta)

            # Compute importance scores for tokens on all batches of validation split

            result_dict = interpret_evaluate(
                interpreters, model, val_iter, args, meta, use_rationales=use_rationales
            )
            # print(result_dict['rationales'])
            # Compute pairwise correlations between interpretability methods
            scores, raw_correlations = pairwise_correlation(
                result_dict["attributions"], correlations
            )

            if use_rationales:
                rationale_scores = rationale_correlation(
                    result_dict["attributions"], result_dict["rationales"]
                )
                pprint(rationale_scores)

            print(f"Epoch={epoch}, evaluating on validation set:")
            result_dict = evaluate(model, val_iter, args, meta)
            loss = result_dict["loss"]

            if best_valid_loss is None or loss < best_valid_loss:
                best_valid_loss = loss
                best_valid_epoch = epoch
                best_model = copy.deepcopy(
                    model
                )  # clone params of model, this might be slow, maybe dump?

            # Run on train set without shuffling so instance indices are preserved
            # train_interpret_scores = interpret_evaluate(interpreters, model, train_iter_noshuf, args, meta, use_rationales=use_rationales)
            # train_scores, train_raw_correlations = pairwise_correlation(train_interpret_scores['attributions'], correlations)

            # for k in train_raw_correlations:
            #   per_instance_agreement = train_raw_correlations[k]

            # min_agreement_indices = np.argsort(per_instance_agreement) # sorted ascending
            # worst_agreement = min_agreement_indices[:args.query_size] # Worst query_size instances
            # worst_agreement = correct_for_missing(worst_agreement, inst_mask)
            # Mask out the worst indices
            # inst_mask[worst_agreement] = False

            # sys.exit(-1)

    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")

    results = {}
    results["loss"] = loss

    if args.model_name == "JWA":
        best_model.rnn.flatten_parameters()

    best_model_pack = (best_model, criterion, optimizer)
    return results, best_model_pack


def main():
    args = make_parser()
    dataloader = dataset_loaders[args.data]

    tokenizer = None
    # If we're using bert, use the pretrained tokenizer instead
    if args.model_name in TRANSFORMERS:
        tokenizer = DistilBertTokenizer.from_pretrained(args.pretrained_model)
        splits, _ = dataloader(tokenizer=tokenizer)
        vocab = TokenizerVocabWrapper(tokenizer)
        # print(vocab.numericalize("A sample sentence"))
    else:
        splits, vocab = dataloader(tokenizer=tokenizer)

    if len(splits) == 3:
        train, val, test = splits
    else:
        train, test = splits
        val = test  # Change sometime later

    meta = Config()
    meta.num_labels = 2
    meta.num_tokens = len(vocab)
    meta.padding_idx = vocab.get_padding_index()
    meta.vocab = vocab

    experiment(args, meta, train, val, test)


if __name__ == "__main__":
    main()
