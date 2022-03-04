import os, sys

from dataclasses import dataclass

@dataclass
class Sample:
  pred: int
  true: int
  proba: float

  def agreed(self):
    return self.pred == self.true

# oracle_model = 'siebert_sentiment-roberta-large-english' # Binary model
oracle_model = 'nlptown_bert-base-multilingual-uncased-sentiment' # 3-class model
dataset_folder = '../data/IMDB_sentencesplit_filtered_lowercased'

dataset_names = {
  'train':'IMDB_sentencesplit_train.csv',
  'test': 'IMDB_sentencesplit_test.csv'
}

def load_dataset(data_path):
  # Just load the lines, no need for splitting / stripping
  dataset_lines = []
  for line in open(data_path, 'r'):
    dataset_lines.append(line.strip())
  return dataset_lines

def load_oracle_judgements(oracle_path):
  judgements = []
  for line in open(oracle_path, 'r'):
    pred, true, proba = line.strip().split(",")
    pred, true = int(pred), int(true)
    proba = float(proba)
    judgements.append(Sample(
        pred, true, proba
      ))
  return judgements


def main():

  for dataset_type in dataset_names:
    dataset_path = os.path.join(dataset_folder, dataset_names[dataset_type])
    oracle_path = os.path.join(dataset_folder, oracle_model, dataset_type + '_outputs.csv')

    oracle_short_name = oracle_model.split('-')[0]
    dataset_dest = os.path.join(dataset_folder, 'IMDB_' + oracle_short_name + '_' + dataset_type + '.csv')

    oracle_judgements = load_oracle_judgements(oracle_path)
    dataset_lines = load_dataset(dataset_path)

    assert len(oracle_judgements) == len(dataset_lines)

    with open(dataset_dest, 'w') as outfile:
      for line, judge in zip(dataset_lines, oracle_judgements):
        if judge.agreed():
          outfile.write(line + "\n")


if __name__ == '__main__':
  main()
