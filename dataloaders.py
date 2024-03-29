from cgi import test
import os, sys
from random import random
from re import M
from sklearn.utils import shuffle
import torch

import numpy as np
from podium import Vocab, Field, LabelField, Iterator  # , BucketIterator
from podium.datasets import TabularDataset, Dataset, ExampleFactory
from podium.datasets.hf import HFDatasetConverter
from podium.vectorizers import GloVe
from podium.datasets.impl import SST
from podium.utils.general_utils import repr_type_and_attrs

from typing import Callable
from typing import Iterator as PythonIterator
from typing import List, NamedTuple, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
from random import Random

from podium.vocab import Special, UNK, PAD


from transformers import BertTokenizer

from datasets import load_dataset

from eraser.eraser_utils import (
    load_documents,
    load_datasets,
    annotations_from_jsonl,
    Annotation,
)


class MASK(Special):
    token = "<MASK>"


class BucketIterator(Iterator):
    """
    Creates a bucket iterator which uses a look-ahead heuristic to batch
    examples in a way that minimizes the amount of necessary padding.

    Uses a bucket of size N x batch_size, and sorts instances within the bucket
    before splitting into batches, minimizing necessary padding.
    """

    def __init__(
        self,
        dataset=None,
        batch_size=32,
        sort_key=None,
        shuffle=True,
        seed=1,
        matrix_class=np.array,
        internal_random_state=None,
        look_ahead_multiplier=100,
        bucket_sort_key=None,
    ):
        """
        Creates a BucketIterator with the given bucket sort key and look-ahead
        multiplier (how many batch_sizes to look ahead when sorting examples for
        batches).

        Parameters
        ----------
        look_ahead_multiplier : int
            Multiplier of ``batch_size`` which determines the size of the
            look-ahead bucket.
            If ``look_ahead_multiplier == 1``, then the BucketIterator behaves
            like a normal Iterator.
            If ``look_ahead_multiplier >= (num_examples / batch_size)``, then
            the BucketIterator behaves like a normal iterator that sorts the
            whole dataset.
            Default is ``100``.
            The callable object used to sort examples in the bucket.
            If ``bucket_sort_key=None``, then the ``sort_key`` must not be ``None``,
            otherwise a ``ValueError`` is raised.
            Default is ``None``.

        Raises
        ------
        ValueError
            If sort_key and bucket_sort_key are both None.
        """

        if sort_key is None and bucket_sort_key is None:
            raise ValueError(
                "For BucketIterator to work, either sort_key or "
                "bucket_sort_key must be != None."
            )

        super().__init__(
            dataset,
            batch_size,
            sort_key=sort_key,
            shuffle=shuffle,
            seed=seed,
            matrix_class=matrix_class,
            internal_random_state=internal_random_state,
        )

        self.bucket_sort_key = bucket_sort_key
        self.look_ahead_multiplier = look_ahead_multiplier

    def __iter__(self) -> PythonIterator[Tuple[NamedTuple, NamedTuple]]:
        step = self.batch_size * self.look_ahead_multiplier
        dataset = self._dataset

        # Fix: Shuffle dataset if the shuffle is turned on, only IF sort key is not none
        if self._shuffle and self._sort_key is None:
            indices = list(range(len(dataset)))
            # Cache state prior to shuffle so we can use it when unpickling
            self._shuffler_state = self.get_internal_random_state()
            self._shuffler.shuffle(indices)
            # dataset.shuffle_examples(random_state=self._shuffler_state)
            dataset = dataset[indices]

        # Determine the step where iteration was stopped for lookahead & within bucket
        lookahead_start = (
            self.iterations // self.look_ahead_multiplier * self.look_ahead_multiplier
        )
        batch_start = self.iterations % self.look_ahead_multiplier

        if self._sort_key is not None:
            dataset = dataset.sorted(key=self._sort_key)
        for i in range(lookahead_start, len(dataset), step):
            bucket = dataset[i : i + step]

            if self.bucket_sort_key is not None:
                bucket = bucket.sorted(key=self.bucket_sort_key)

            for j in range(batch_start, len(bucket), self.batch_size):
                batch_dataset = bucket[j : j + self.batch_size]
                batch = self._create_batch(batch_dataset)

                yield batch
                self._iterations += 1

        # prepare for new epoch
        self._iterations = 0
        self._epoch += 1

    def __repr__(self) -> str:
        attrs = {
            "batch_size": self._batch_size,
            "epoch": self._epoch,
            "iteration": self._iterations,
            "shuffle": self._shuffle,
            "look_ahead_multiplier": self.look_ahead_multiplier,
        }
        return repr_type_and_attrs(self, attrs, with_newlines=True)


class TokenizerVocabWrapper:
    def __init__(self, tokenizer):
        # wrap BertTokenizer so the method signatures align with podium
        self.tokenizer = tokenizer

    def get_padding_index(self):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

    def __len__(self):
        return len(self.tokenizer)

    def numericalize(self, instance):
        # Equivalent to .encode, but I want to delineate the steps
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(instance))


def load_embeddings(vocab, name="glove"):
    if name == "glove":
        glove = GloVe()
        embeddings = glove.load_vocab(vocab)
        return embeddings
    else:
        raise ValueError(f"Wrong embedding key provided {name}")
        # return None


def make_iterable(dataset, device, batch_size=32, train=False, indices=None):
    """
    Construct a DataLoader from a podium Dataset
    """

    def instance_length(instance):
        raw, tokenized = instance.text
        return -len(tokenized)

    def cast_to_device(data):
        return torch.tensor(np.array(data), device=device)

    # Selects examples at given indices to support subset iteration.
    if indices is not None:
        dataset = dataset[indices]

    iterator = BucketIterator(
        dataset,
        batch_size=batch_size,
        sort_key=instance_length,
        shuffle=train,
        matrix_class=cast_to_device,
        look_ahead_multiplier=20,
    )

    # iterator = Iterator(
    #    dataset,
    #    batch_size=batch_size,
    #   bucket_sort_key=instance_length,
    #    shuffle=train,
    #    matrix_class=cast_to_device,
    # )

    return iterator


class Instance:
    def __init__(self, index, text, label, extras=None):
        self.index = index
        self.text = text
        self.label = label
        self.extras = extras
        self.length = len(text)  # text is already tokenized & filtered

    def set_mask(self, masked_text, masked_labels):
        # Set the masking as an attribute
        self.masked_text = masked_text
        self.masked_labels = masked_labels

    def set_numericalized(self, indices, target):
        self.numericalized_text = indices
        self.numericalized_label = target
        self.length = len(indices)

    def __repr__(self):
        return f"{self.index}: {self.length}, {self.label}"


def generate_eraser_rationale_mask(tokens, evidences):
    mask = torch.zeros(len(tokens))  # zeros for where you can attend to

    any_evidence_left = False
    for ev in evidences:
        if ev.start_token > len(tokens) or ev.end_token > len(tokens):
            continue  # evidence out of span

        if not any_evidence_left:
            any_evidence_left = True
        # 1. Validate

        assert ev.text == " ".join(
            tokens[ev.start_token : ev.end_token]
        ), "Texts dont match; did you filter some tokens?"

        mask[ev.start_token : ev.end_token] = 1
    return mask


# Map this dataloader to podium Dataset classes ?
def eraser_reader(data_root, conflate=False):
    documents = load_documents(data_root)
    train, val, test = load_datasets(data_root)

    # Tokenizer is always str.split
    # Maybe remove max_len

    splits = []

    for split in [train, val, test]:
        examples = []
        freqs = {}
        class_map = {}
        for idx, row in enumerate(split):
            evidences = row.all_evidences()
            if not evidences:
                # Skip document that doesn't have any evidence; TODO: how many of those exist?
                continue
            # Get document id for the annotation
            # If there are multiple document ids, this isn't the case we
            #  are looking for here
            (docid,) = set(ev.docid for ev in evidences)

            document = documents[docid]
            # obtain whole document tokens; flatten nested list
            document_tokens = [token for sentence in document for token in sentence]

            label = row.classification
            # build frequencies
            for token in document_tokens:
                if token not in freqs:
                    freqs[token] = 0
                freqs[token] += 1

            if label not in class_map:
                class_map[label] = len(class_map)

            rationale_mask = generate_eraser_rationale_mask(document_tokens, evidences)

            extras = {"evidence": row, "rationale_mask": rationale_mask}

            instance = Instance(idx, document_tokens, label, extras)
            examples.append(instance)
        if not conflate:
            splits.append((examples, class_map, freqs))
        else:
            splits.extend(examples)
    return splits


class IMDBRationale(Dataset):
    @staticmethod
    def load_dataset_splits(fields, data_root="data/movies"):
        documents = load_documents(data_root)
        train, val, test = load_datasets(data_root)

        fact = ExampleFactory(fields)

        dataset_splits = {}

        for name, split in zip(["train", "val", "test"], [train, val, test]):
            split_examples = []
            for idx, row in enumerate(split):
                evidences = row.all_evidences()
                if not evidences:
                    # Skip document that doesn't have any evidence; TODO: how many of those exist?
                    continue

                (docid,) = set(ev.docid for ev in evidences)

                document = documents[docid]
                # obtain whole document tokens; flatten nested list
                document_tokens = [token for sentence in document for token in sentence]

                label = row.classification
                rationale_mask = generate_eraser_rationale_mask(
                    document_tokens, evidences
                )

                example_dict = {
                    "id": docid,
                    "text": " ".join(document_tokens),
                    "label": label,
                    "rationale": rationale_mask.numpy(),
                }

                example = fact.from_dict(example_dict)
                split_examples.append(example)
            dataset_split = Dataset(**{"examples": split_examples, "fields": fields})
            dataset_splits[name] = dataset_split
        return dataset_splits

    @staticmethod
    def get_default_fields(tokenizer=None):
        if tokenizer is None:
            vocab = Vocab(max_size=20000)
            text = Field("text", numericalizer=vocab, include_lengths=True)
        else:
            vocab = None
            text = Field(
                "text",
                tokenizer=tokenizer.tokenize,
                padding_token=pad_index,
                numericalizer=tokenizer.convert_tokens_to_ids,
                include_lengths=True,
            )

        fields = {
            "id": Field("id", numericalizer=None),
            "text": text,
            "rationale": Field(
                "rationale", tokenizer=None, numericalizer=None
            ),  # shd be a boolean mask of same length as text
            "label": LabelField("label"),
        }

        return fields, vocab


def load_tse(
    train_path="data/TSE/train.csv", test_path="data/TSE/test.csv", max_size=20000
):

    vocab = Vocab(max_size=max_size)
    fields = [
        Field("id", numericalizer=None),
        Field("text", numericalizer=vocab, include_lengths=True),
        Field("rationale", numericalizer=vocab),
        LabelField("label"),
    ]
    train_dataset = TabularDataset(
        train_path, format="csv", fields=fields, skip_header=True
    )
    test_dataset = TabularDataset(
        test_path, format="csv", fields=fields, skip_header=True
    )
    train_dataset.finalize_fields()
    return (train_dataset, test_dataset), vocab


def load_imdb_rationale(
    tokenizer=None,
    train_path="data/movies/train.csv",
    valid_path="data/movies/dev.csv",
    test_path="data/movies/test.csv",
    max_size=20000,
):
    fields, vocab = IMDBRationale.get_default_fields(tokenizer)
    splits = IMDBRationale.load_dataset_splits(fields)
    splits["train"].finalize_fields()
    return list(splits.values()), vocab


class MaxLenHook:
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, raw, tokenized):
        return raw, tokenized[: self.max_len]


def lowercase_hook(raw, tokenized):
    return raw, [tok.lower() for tok in tokenized]


def isalnum(token):
    return any(c.isalnum() for c in token)


def remove_nonalnum(raw, tokenized):
    # Remove non alphanumeric tokens
    return raw, [tok for tok in tokenized if isalnum(tok)]


def load_imdb(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/IMDB",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_imdb_sentences(
    tokenizer=None,
    train_path="data/IMDB_sentences/IMDB_nlptown_bert_train.csv",
    test_path="data/IMDB_sentences/IMDB_nlptown_bert_test.csv",
    max_vocab_size=20000,
    max_len=100,
    clip=True,
):

    post_hooks = []
    if max_len:
        post_hooks.append(MaxLenHook(max_len))

    if tokenizer is None:
        vocab = Vocab(max_size=max_vocab_size)
        fields = [
            Field("id"),
            Field(
                "text",
                numericalizer=vocab,
                include_lengths=True,
                posttokenize_hooks=post_hooks,
                keep_raw=True,
            ),
            LabelField("label"),
        ]
    else:
        # Use BERT subword tokenization
        vocab = None
        pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        fields = [
            Field("id"),
            Field(
                "text",
                tokenizer=tokenizer.tokenize,
                padding_token=pad_index,
                numericalizer=tokenizer.convert_tokens_to_ids,
                include_lengths=True,
                posttokenize_hooks=post_hooks,
                keep_raw=True,
            ),
            LabelField("label"),
        ]

    train_dataset, valid_dataset = TabularDataset(
        train_path, format="csv", fields=fields
    ).split(split_ratio=0.7, random_state=42)
    test_dataset = TabularDataset(test_path, format="csv", fields=fields)
    test_dataset.shuffle_examples(random_state=42)

    train_dataset.finalize_fields()
    test_dataset = test_dataset[:10_000] if clip else test_dataset
    return (
        train_dataset[:50_000],
        valid_dataset[:10_000],
        test_dataset,
    ), vocab


def test_load_imdb():
    splits, vocab = load_imdb(
        None, "data/IMDB/train.csv", "data/IMDB/dev.csv", "data/IMDB/test.csv"
    )
    print(vocab)
    train, valid, test = splits
    print(len(train), len(valid), len(test))

    print(train)
    print(train[0])

    device = torch.device("cpu")
    train_iter = make_iterable(train, device, batch_size=2)
    batch = next(iter(train_iter))

    print(batch)
    text, length = batch.text
    print(vocab.reverse_numericalize(text[0]))
    print(length[0])
    print(vocab.get_padding_index())


def test_load_imdb_sentences():
    splits, vocab = load_imdb_sentences()
    print(vocab)
    train, valid, test = splits
    print(len(train), len(valid), len(test))

    print(train)

    device = torch.device("cpu")
    train_iter = make_iterable(test, device, batch_size=16)
    batch = next(iter(train_iter))

    print(batch)
    text, length = batch.text
    for i in range(len(text)):
        print(vocab.reverse_numericalize(text[i]))
        print(length[i])


def test_load_imdb_rationale(conflate=True):
    if not conflate:
        dataset_splits = eraser_reader("data/movies", conflate=conflate)
        instances, _, _ = dataset_splits[0]
    else:
        instances = eraser_reader("data/movies", conflate=conflate)
    print(len(instances))
    print(instances[0])
    print(instances[0].extras["rationale_mask"])


def test_load_tse_rationale():
    (tse_train, tse_test), vocab = load_tse()
    print(tse_train[0])

    device = torch.device("cpu")
    train_iter = make_iterable(tse_train, device, batch_size=2)
    batch = next(iter(train_iter))

    print(batch)
    text, length = batch.text

    print(vocab.reverse_numericalize(text[0]))
    print(length[0])
    print(vocab.get_padding_index())


def load_dataset(
    data_dir, meta, tokenizer=None, max_vocab_size=20_000, max_seq_len=200
):
    if tokenizer is None:
        vocab = Vocab(max_size=max_vocab_size, specials=(UNK(), PAD(), MASK()))
        fields = [
            Field("id", disable_batch_matrix=True),
            Field(
                "text",
                numericalizer=vocab,
                include_lengths=True,
                posttokenize_hooks=[
                    remove_nonalnum,
                    MaxLenHook(max_seq_len),
                    lowercase_hook,
                ],
            ),
            LabelField("label"),
        ]
    else:
        # Use BERT subword tokenization
        vocab = TokenizerVocabWrapper(tokenizer)
        pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        fields = [
            Field("id", disable_batch_matrix=True),
            Field(
                "text",
                tokenizer=tokenizer.tokenize,
                padding_token=pad_index,
                numericalizer=tokenizer.convert_tokens_to_ids,
                include_lengths=True,
                posttokenize_hooks=[
                    remove_nonalnum,
                    MaxLenHook(max_seq_len),
                    lowercase_hook,
                ],
            ),
            LabelField("label"),
        ]

    train = TabularDataset(
        os.path.join(data_dir, "train.csv"), format="csv", fields=fields
    )
    val = TabularDataset(os.path.join(data_dir, "dev.csv"), format="csv", fields=fields)
    test = TabularDataset(
        os.path.join(data_dir, "test.csv"), format="csv", fields=fields
    )

    train.finalize_fields()

    meta.vocab = vocab
    meta.num_tokens = len(vocab)
    meta.padding_idx = vocab.get_padding_index()
    meta.num_labels = len(train.field("label").vocab)

    return (train, val, test), vocab


def load_sst(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):
    return load_dataset(
        "data/SST",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def test_load_sst(max_vocab_size=20_000, max_seq_len=200):
    splits, vocab = load_sst()
    print(vocab)
    train, valid, test = splits
    print(len(train), len(valid), len(test))

    print(train)
    print(train[0])

    device = torch.device("cpu")
    train_iter = make_iterable(train, device, batch_size=2)
    batch = next(iter(train_iter))

    print(batch)
    text, length = batch.text
    print(vocab.reverse_numericalize(text[0]))
    print(length[0])
    print(vocab.get_padding_index())


def load_jwa_sst(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/JWA-SST",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_trec(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/TREC",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_cola(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/COLA",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_polarity(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/POL",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_subj(
    meta,
    tokenizer=None,
    max_vocab_size=20_000,
    max_seq_len=200,
):

    return load_dataset(
        "data/SUBJ",
        meta=meta,
        tokenizer=tokenizer,
        max_vocab_size=max_vocab_size,
        max_seq_len=max_seq_len,
    )


def load_trec_hf(label="label-coarse", max_vocab_size=20_000, max_seq_len=200):
    vocab = Vocab(max_size=max_vocab_size)
    fields = [
        Field(
            "text",
            numericalizer=vocab,
            include_lengths=True,
            posttokenize_hooks=[MaxLenHook(max_seq_len)],
            keep_raw=True,
        ),
        LabelField("label"),
    ]
    hf_dataset = load_dataset("trec")
    hf_dataset = hf_dataset.rename_column(label, "label")
    print(hf_dataset)
    hf_train_val, hf_test = (
        hf_dataset["train"],
        hf_dataset["test"],
    )
    train_val_conv = HFDatasetConverter(hf_train_val, fields=fields)
    test_conv = HFDatasetConverter(hf_test, fields=fields)
    train_val, test = (
        train_val_conv.as_dataset(),
        test_conv.as_dataset(),
    )
    train, val = train_val.split(split_ratio=0.8, random_state=0)
    train.finalize_fields()
    print(train)
    return (train, val, test), vocab


def test_load_trec():
    splits, vocab = load_trec()
    print(vocab)
    train, valid, test = splits
    print(len(train), len(valid), len(test))

    print(train)
    print(train[0])

    device = torch.device("cpu")
    train_iter = make_iterable(train, device, batch_size=2)
    batch = next(iter(train_iter))

    print(batch)
    text, length = batch.text
    print(vocab.reverse_numericalize(text[0]))
    print(length[0])
    print(vocab.get_padding_index())


def add_ids_to_files(root_folder):
    split_ins = ["train_old.csv", "dev_old.csv", "test_old.csv"]
    split_outs = ["train.csv", "dev.csv", "test.csv"]

    for split_in, split_out in zip(split_ins, split_outs):
        with open(os.path.join(root_folder, split_in), "r") as infile:
            with open(os.path.join(root_folder, split_out), "w") as outfile:
                for idx, line in enumerate(infile):
                    parts = line.strip().split(",")
                    if idx == 0:
                        continue
                    outfile.write(f"{idx-1},{parts[0]},{parts[1]}\n")


if __name__ == "__main__":
    add_ids_to_files("data/JWA-SST")
    # (train, dev, test), vocab = load_imdb_rationale()
    # print(len(train), len(dev), len(test))
    # print(train[0].keys())

    # device = torch.device("cpu")
    # train_iter = make_iterable(train, device, batch_size=2)
    # batch = next(iter(train_iter))

    # print(batch)
    # text, length = batch.text
    # rationale = batch.rationale
    # print(vocab.reverse_numericalize(text[0]))
    # print(length[0])
    # print(vocab.get_padding_index())
    # print(rationale)
