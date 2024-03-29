import os, sys
import matplotlib
matplotlib.use('Agg') # No display
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset

from podium.datasets import TabularDataset, ExampleFactory
from podium import Vocab, Field, LabelField, BucketIterator

# "finiteautomata/beto-sentiment-analysis", # Spanish, disregard

models_to_consider = ["distilbert-base-uncased-finetuned-sst-2-english"
                      "siebert/sentiment-roberta-large-english",
                      "nlptown/bert-base-multilingual-uncased-sentiment",
                      "cardiffnlp/twitter-roberta-base-sentiment",
]

dataset_paths = {
  'train': '../data/IMDB_sentencesplit_filtered_lowercased/IMDB_sentencesplit_train.csv',
  'test': '../data/IMDB_sentencesplit_filtered_lowercased/IMDB_sentencesplit_test.csv'
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
    # siebert labels
    'positive': 1,
    'negative': 0,
    # Beto-sentiment-analysis labels
    'pos': 1,
    'neg': 0,
    'neu': 2,
    # nlptown labels
    '1 star': 0, # Negative
    '2 stars': 0, # Negative
    '3 stars': 2, # Neutral?
    '4 stars': 1, # Positive
    '5 stars': 1, # Positive
    # Cardiffnlp labels
    'label_0': 0, # Negative
    'label_1': 2, # Neutral
    'label_2': 1, # Positive
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

def process_dataset(model, dataset, batch_size=32, split='train'):
  stats = {}
  preds = []
  proba = []
  labels = []

  _, dataset_labels = zip(*list(dataset.dataset.label))
  dataset_labels = list(dataset_labels)
  for idx, result in tqdm(enumerate(model(dataset,
                                          batch_size=batch_size,
                                          truncation=True)),
                          total=len(dataset)):
    label = dataset_labels[idx]

    # Map string labels to indices
    pred = result['label'].lower()
    prob = result['score']

    # Store
    preds.append(l2i[pred])
    labels.append(l2i[label])
    proba.append(prob)

  stats['preds'] = preds
  stats['proba'] = proba
  stats['labels'] = labels

  confmat = confusion_matrix(labels, preds)
  if max(preds) >= 2:
    # Really ugly, detects multiclass
    fscore = f1_score(labels, preds, average='micro')
  else:
    # Binary
    fscore = f1_score(labels, preds)

  stats['fscore'] = fscore
  stats['confmat'] = confmat

  print(f"Stats on {split} dataset:")
  print("F1 score:", fscore)
  print(confmat)
  return stats

def store_stats(stats, dest):
  # This should be prettier
  # Store raw predictions and prediction probabilities
  preds_ext = "_outputs.csv"
  with open(dest + preds_ext, 'w') as outfile:
    preds, proba, labels = stats['preds'], stats['proba'], stats['labels']
    for p, y_hat, y in zip(proba, preds, labels):
      outfile.write(str(y_hat) + "," + str(y) + "," + str(p) + "\n")

  # Store f1 score and confusion matrix in raw txt
  stats_ext = "_stats.txt"
  with open(dest + stats_ext, 'w') as outfile:
    outfile.write("F1: " + str(stats['fscore']) + "\n")
    outfile.write(str(stats['confmat']))

  # Store prediction probability distributions
  png_dest = dest + '_pred_dist.png'
  fig = plt.figure()
  plt.hist(proba)
  plt.savefig(png_dest)
  plt.close(fig)

def main():
  batch_size = 32
  cuda = 1

  for model_name in models_to_consider:
    print(f"For model: {model_name}")
    if 'cardiffnlp' in model_name:
      # Hardcode max length because for some reason it isn't defined in model
      tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
      model = pipeline("sentiment-analysis", model=model_name, device=cuda,
                        tokenizer=tokenizer)
    else:
      model = pipeline("sentiment-analysis", model=model_name, device=cuda)

    for k,v in dataset_paths.items():
      dataset_dir, _ = os.path.split(v)
      # Make directory to store output files
      root_log_dir = os.path.join(dataset_dir, model_name.replace("/", "_"))
      os.makedirs(root_log_dir, exist_ok=True)

      dataset = load_dataset(v)
      stats = process_dataset(model, dataset, batch_size=batch_size, split=k)

      stats_dest = os.path.join(root_log_dir, k)
      store_stats(stats, stats_dest)


if __name__ == '__main__':
  main()