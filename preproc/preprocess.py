import os, sys
import numpy as np
from podium import Vocab, Field, LabelField, BucketIterator
from podium.datasets import TabularDataset, Dataset, ExampleFactory
from podium.datasets.hf import HFDatasetConverter
from podium.vectorizers import GloVe
from podium.datasets.impl import SST, IMDB

import spacy

# I can't think of a good way to create 3 data loading abstractions:
#  (1) for *BERT, (2) for JWA and (3) for preprocessing, so just copying code

def load_imdb_sentence_split(
  train_path="../data/IMDB/train.csv",
  valid_path="../data/IMDB/dev.csv",
  test_path="../data/IMDB/test.csv",
  max_size=20000,
  max_len=200
):
  
  nlp = spacy.load("en_core_web_sm", disable = ['ner', 'parser'])
  nlp.add_pipe("sentencizer")

  fields = {
      'text': Field(
          "text",
          tokenizer=None,
          numericalizer=None,
          pretokenize_hooks=nlp
      ),
      'label': LabelField("label"),
  }

  train, test = IMDB.get_dataset_splits(fields=fields)
  return train, test
