import os, sys
import matplotlib
matplotlib.use('Agg') # No display
from transformers import pipeline
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset

from podium.datasets import TabularDataset, Dataset, ExampleFactory
from podium import Vocab, Field, LabelField, BucketIterator

models_to_consider = ["siebert/sentiment-roberta-large-english"]

dataset_paths = {
  'train': '../data/IMDB_sentencesplit_nofilter/IMDB_sentencesplit_train.csv',
  'test': '../data/IMDB_sentencesplit_nofilter/IMDB_sentencesplit_test.csv'
}

fields = [
    Field(
        "text",
        numericalizer=None,
        tokenizer=None
    ),
    LabelField("label"),
]

l2i = {
    'positive': 1,
    'negative': 0
}

class SimpleDataset(Dataset):
    def __init__(self, podium_dataset):
        self.dataset = podium_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        # We need to pass only the text to hf pipeline
        instance = self.dataset[key]
        text = instance.text[1]
        return text

def load_dataset(data_path):
  dataset = TabularDataset(data_path, format="csv", fields=fields)
  simplified_dataset = SimpleDataset(dataset)
  return simplified_dataset

def process_dataset(model, dataset):
  stats = {}
  preds = []
  proba = []
  labels = []

  _, dataset_labels = zip(*list(dataset.label))
  dataset_labels = list(dataset_labels)
  for idx, result in tqdm(enumerate(model(simplified_dataset,
                                          batch_size=batch_size,
                                          truncation=True)),
                          total=len(simplified_dataset)):
    label = dataset_labels[idx]

    # Map string labels to indices
    pred = result['label'].lower()
    prob = result['score']

    # Store
    preds.append(l2i[pred])
    labels.append(l2i[label])
    proba.append(prob)

  stats[k]['preds'] = preds
  stats[k]['proba'] = proba
  stats[k]['labels'] = labels
  print(f"Stats on {k} dataset:")
  print("F1 score:", f1_score(labels, preds))
  print(confusion_matrix(labels, preds))
  return stats

def store_stats(stats, dest):
    with open(dest, 'w') as outfile:
      preds, proba, labels = stats['preds'], stats['proba'], stats['labels']
      for p, y_hat, y in zip(proba, preds, labels):
        outfile.write(str(y_hat) + "," + str(y) + "," + str(p) + "\n")

      png_dest = dest.replace('.csv', '_pred_dist.png')
      fig = plt.figure()
      plt.hist(proba)
      plt.savefig(png_dest)
      plt.close(fig)


def main():
  batch_size = 32
  cuda = 1

  for model_name in models_to_consider:
    model = pipeline("sentiment-analysis", model=model_name, device=cuda)

    for k,v in dataset_paths.items():
      dataset_dir, _ = os.path.split(v)
      # Make directory to store output files
      root_log_dir = os.path.join(dataset_dir, model_name.replace("/", "_"))
      os.makedirs(root_log_dir, exist_ok=True)

      simplified_dataset = load_dataset(v)
      stats = process_dataset(model, dataset)

      stats_dest = os.path.join(root_log_dir, f"{k}_outputs.csv")
      store_stats(stats, stats_dest)


if __name__ == '__main__':
  main()