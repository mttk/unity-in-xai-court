import os, sys
from re import M
import torch

import numpy as np
from podium import Vocab, Field, LabelField, BucketIterator
from podium.datasets import TabularDataset, Dataset, ExampleFactory
from podium.datasets.hf import HFDatasetConverter
from podium.vectorizers import GloVe

from datasets import load_dataset

from eraser.eraser_utils import load_documents, load_datasets, annotations_from_jsonl, Annotation


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
        bucket_sort_key=instance_length,
        shuffle=train,
        matrix_class=cast_to_device,
    )

    return iterator


class Instance:
    def __init__(self, index, text, label, extras=None):
        self.index = index
        self.text = text
        self.label = label
        self.extras = extras
        self.length = len(text) # text is already tokenized & filtered

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
    mask = torch.zeros(len(tokens)) # zeros for where you can attend to

    any_evidence_left = False
    for ev in evidences:
        if ev.start_token > len(tokens) or ev.end_token > len(tokens): 
            continue  # evidence out of span

        if not any_evidence_left: any_evidence_left = True
        # 1. Validate

        assert ev.text == ' '.join(tokens[ev.start_token:ev.end_token]), "Texts dont match; did you filter some tokens?"

        mask[ev.start_token:ev.end_token] = 1
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
                if token not in freqs: freqs[token] = 0
                freqs[token] += 1

            if label not in class_map:
                class_map[label] = len(class_map)

            rationale_mask = generate_eraser_rationale_mask(document_tokens, evidences)

            extras = {
                'evidence': row,
                'rationale_mask': rationale_mask
            }

            instance = Instance(idx, document_tokens, label, extras)
            examples.append(instance)
        if not conflate:
            splits.append((examples, class_map, freqs))
        else:
            splits.extend(examples)
    return splits


class IMDBRationale(Dataset):

    @staticmethod
    def load_dataset_splits(fields, data_root='data/movies'):
        documents = load_documents(data_root)
        train, val, test = load_datasets(data_root)

        fact = ExampleFactory(fields)

        dataset_splits = {}

        for name, split in zip(['train', 'val', 'test'], [train, val, test]):
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
                rationale_mask = generate_eraser_rationale_mask(document_tokens, evidences)

                example_dict = {
                    'id': docid,
                    'text': ' '.join(document_tokens),
                    'label': label,
                    'rationale': rationale_mask.numpy()
                }

                example = fact.from_dict(example_dict)
                split_examples.append(example)
            dataset_split = Dataset(**{"examples":split_examples, "fields":fields})
            dataset_splits[name] = dataset_split
        return dataset_splits

    @staticmethod
    def get_default_fields():
        vocab = Vocab(max_size=20000)
        fields = {
            'id': Field("id", numericalizer=None),
            'text': Field("text", numericalizer=vocab, include_lengths=True),
            'rationale': Field("rationale", tokenizer=None, numericalizer=None), # shd be a boolean mask of same length as text
            'label': LabelField("label"),
        }
        return fields, vocab


def load_tse(train_path="data/TSE/train.csv", 
             test_path="data/TSE/test.csv",
             max_size=20000):

    vocab = Vocab(max_size=max_size)
    fields = [
        Field("id", numericalizer=None),
        Field("text", numericalizer=vocab, include_lengths=True),
        Field("rationale", numericalizer=vocab),
        LabelField("label"),
    ]
    train_dataset = TabularDataset(train_path, format="csv", fields=fields, skip_header=True)
    test_dataset = TabularDataset(test_path, format="csv", fields=fields, skip_header=True)
    train_dataset.finalize_fields()
    return (train_dataset, test_dataset), vocab

def load_imdb_rationale(
    train_path="data/movies/train.csv",
    valid_path="data/movies/dev.csv",
    test_path="data/movies/test.csv",
    max_size=20000
):
    fields, vocab = IMDBRationale.get_default_fields()
    splits = IMDBRationale.load_dataset_splits(fields)
    splits['train'].finalize_fields()
    return list(splits.values()), vocab

class MaxLenHook():
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, raw, tokenized):
        return raw, tokenized[:self.max_len]

def load_imdb(
    train_path="data/IMDB/train.csv",
    valid_path="data/IMDB/dev.csv",
    test_path="data/IMDB/test.csv",
    max_size=20000,
    max_len=200
):

    vocab = Vocab(max_size=max_size)
    fields = [
        Field("text", numericalizer=vocab, include_lengths=True,
               posttokenize_hooks=[MaxLenHook(max_len)]),
        LabelField("label"),
    ]

    train_dataset = TabularDataset(train_path, format="csv", fields=fields)
    valid_dataset = TabularDataset(valid_path, format="csv", fields=fields)
    test_dataset = TabularDataset(test_path, format="csv", fields=fields)

    train_dataset.finalize_fields()
    return (train_dataset, valid_dataset, test_dataset), vocab

def test_load_imdb():
    splits, vocab = load_imdb(
        "data/IMDB/train.csv", "data/IMDB/dev.csv", "data/IMDB/test.csv"
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


def test_load_imdb_rationale(conflate=True):
    if not conflate:
        dataset_splits = eraser_reader('data/movies', conflate=conflate)
        instances, _, _ = dataset_splits[0]
    else:
        instances = eraser_reader('data/movies', conflate=conflate)
    print(len(instances))
    print(instances[0])
    print(instances[0].extras['rationale_mask'])

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


def load_trec(label="label-coarse", max_vocab_size=20_000, max_seq_len=200):
    vocab = Vocab(max_size=max_vocab_size)
    fields = [
        Field(
            "text",
            numericalizer=vocab,
            include_lengths=True,
            posttokenize_hooks=[MaxLenHook(max_seq_len)]
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
    train, val = train_val.split(split_ratio=0.8)
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


if __name__ == "__main__":
    (train, dev, test), vocab = load_imdb_rationale()
    print(len(train), len(dev), len(test))
    print(train[0].keys())

    device = torch.device("cpu")
    train_iter = make_iterable(train, device, batch_size=2)
    batch = next(iter(train_iter))

    print(batch)
    text, length = batch.text
    rationale = batch.rationale
    print(vocab.reverse_numericalize(text[0]))
    print(length[0])
    print(vocab.get_padding_index())
    print(rationale)
