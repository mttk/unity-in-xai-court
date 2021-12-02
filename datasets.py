import os, sys
import torch

from podium import Vocab, Field, LabelField, BucketIterator
from podium.datasets import TabularDataset
from podium.vectorizers import GloVe

def load_embeddings(vocab, name='glove'):
  if name == 'glove':
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
      return torch.tensor(data, device=device)

    # Selects examples at given indices to support subset iteration.
    if indices:
      dataset = dataset[indices]

    iterator = BucketIterator(dataset, batch_size=batch_size,
                              bucket_sort_key=instance_length, shuffle=train,
                              matrix_class=cast_to_device)

    return iterator


def load_imdb(train_path='datasets/IMDB/train.csv',
              valid_path='datasets/IMDB/dev.csv',
              test_path='datasets/IMDB/test.csv',
              max_size=20000):
  vocab = Vocab(max_size=max_size)
  fields = [
      Field('text', numericalizer=vocab, include_lengths=True),
      LabelField('label')
    ]

  train_dataset = TabularDataset(train_path, format='csv', fields=fields)
  valid_dataset = TabularDataset(valid_path, format='csv', fields=fields)
  test_dataset = TabularDataset(test_path, format='csv', fields=fields)

  train_dataset.finalize_fields()
  return (train_dataset, valid_dataset, test_dataset), vocab

if __name__ == '__main__':
  splits, vocab = load_imdb('datasets/IMDB/train.csv','datasets/IMDB/dev.csv','datasets/IMDB/test.csv')
  print(vocab)
  train, valid, test = splits
  print(len(train), len(valid), len(test))

  print(train)
  print(train[0])

  device = torch.device('cpu')
  train_iter = make_iterable(train, device, batch_size=2)
  batch = next(iter(train_iter))

  print(batch)
  text, length = batch.text
  print(vocab.reverse_numericalize(text[0]))
  print(length[0])
