import argparse
import os, sys
import time
import copy
import itertools

from datetime import datetime
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from podium import BucketIterator

from util import Config
from dataloaders import *
from model import *
from interpret import *
from correlation_measures import *

from sklearn.metrics import average_precision_score

word_vector_files = {
  'glove' : os.path.expanduser('~/data/vectors/glove.840B.300d.txt')
}

dataset_loaders = {
  'IMDB': load_imdb,
  'IMDB-rationale': load_imdb_rationale,
  'TSE': load_tse
}

def make_parser():
  parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')
  parser.add_argument('--data', type=str, default='IMDB',
                        help='Data corpus: [IMDB, IMDB-rationale, TSE]')

  parser.add_argument('--rnn_type', type=str, default='LSTM',
                        help='type of recurrent net [LSTM, GRU, MHA]')
  parser.add_argument('--attention_type', type=str, default='nqadd',
                        help='attention type [dot, add, nqdot, nqadd], default = nqadd')
  parser.add_argument('--embedding_dim', type=int, default=300,
                        help='size of word embeddings [Uses pretrained on 300]')
  parser.add_argument('--hidden_dim', type=int, default=150,
                        help='number of hidden units for the encoder')
  parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers of the encoder')
  parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
  parser.add_argument('--vectors', type=str, default='glove',
                        help='Pretrained vectors to use [glove, fasttext]')
  parser.add_argument('--clip', type=float, default=5,
                        help='gradient clipping')
  parser.add_argument('--epochs', type=int, default=5,
                        help='upper epoch limit')
  parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
  parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout')
  parser.add_argument('--l2', type=float, default=1e-5,
                        help='l2 regularization (weight decay)')
  parser.add_argument('--bi', action='store_true',
                        help='[USE] bidirectional encoder')
  parser.add_argument('--freeze', action='store_true',
                        help='Freeze embeddings')

  # Interpreters & corr measures
  parser.add_argument("--interpreters", nargs="+",
                      default=["deeplift", "grad-shap"], choices=["deeplift", "grad-shap", "deeplift-shap", "int-grad", "lime"],
                      help="Specify a list of interpreters.")
  parser.add_argument("--correlation_measures", nargs="+",
                      default=["kendall-tau"], choices=["kendall-tau"],
                      help="Specify a list of correlation metrics.")

  # Vocab specific arguments
  parser.add_argument('--max_vocab', type=int, default=10000,
                        help='maximum size of vocabulary')
  parser.add_argument('--min_freq', type=int, default=5,
                        help='minimum word frequency')
  parser.add_argument('--max_len', type=int, default=200,
                        help='maximum length of input sequence')

  # Repeat experiments
  parser.add_argument('--repeat', type=int, default=5,
                        help='number of times to repeat training')

  # Gpu based arguments
  parser.add_argument('--gpu', type=int, default=-1,
                        help='Gpu to use for experiments (-1 means no GPU)')

  # Storing & loading arguments
  parser.add_argument('--save', type=str, default='chkp/',
                        help='Folder to store final model (or model with best valid perf) in')
  parser.add_argument('--log', type=str, default='tb_log/',
                        help='Folder to store tensorboard logs in')
  parser.add_argument('--restore', type=str, default='',
                        help='File to restore model from')

  # Model "unlearning" -- removing train instances with largest disagreement
  parser.add_argument('--ul-epochs', type=int, default=-1,
                        help='Number of UL epochs (-1 uses the whole train set)')

  # Active learning arguments
  parser.add_argument('--al-sampler', default="entropy",
                        choices=[
                          "random",
                          "least_confident",
                          "margin",
                          "entropy",
                          "kmeans",
                          "least_confident_dropout",
                          "margin_dropout",
                          "entropy_dropout",
                          "badge",
                          "core_set",
                          "batch_bald",
                        ],
                        help='Active learning sampler')
  parser.add_argument('--al-epochs', type=int, default=-1,
                        help='Number of AL epochs (-1 uses the whole train set)')
  parser.add_argument('--query-size', type=int, default=50,
                        help='Active learning query size.')
  parser.add_argument('--warm-start-size', type=int, default=50,
                        help='Initial AL batch size.')

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
    confusion_matrix[y, max_ind] +=1
  else:
    for j, i in zip(max_ind, y):
      confusion_matrix[int(i),int(j)]+=1

  return accuracy + correct, confusion_matrix


def initialize_model(args, meta):
  # 1. Construct encoder (shared in any case)
  # 2. Construct decoder / decoders
  meta.embeddings = torch.tensor(load_embeddings(meta.vocab, name='glove'))
  model = JWAttentionClassifier(args, meta)

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

  scores = {} # pairwise for each correlation
  raw_correlations = {}

  for corr_idx, corr in enumerate(correlation_measures):
    for i, k_i in enumerate(importance_dictionary):
      for j, k_j in enumerate(importance_dictionary):
        corrs = []

        if k_i == k_j or (k_i, k_j) in scores or (k_j, k_i) in scores:
          # Account for same & symmetry
          continue


        for inst_i, inst_j in zip(importance_dictionary[k_i], importance_dictionary[k_j]):
          r = corr.correlation(inst_i, inst_j)
          corrs.append(r[corr.id].correlation)
        scores[(k_i, k_j)] = np.mean(corrs)
        raw_correlations[(k_i, k_j)] = corrs

  pprint(scores)
  return scores, raw_correlations

def evaluate(model, data, args, meta):
  model.eval()

  accuracy, confusion_matrix = 0, np.zeros((meta.num_labels, meta.num_labels), dtype=int)

  with torch.inference_mode():
    for batch_num, batch in enumerate(data):

      t = time.time()
      model.zero_grad()

      # Unpack batch & cast to device
      (x, lengths), y = batch.text, batch.label

      y = y.squeeze() # y needs to be a 1D tensor for xent(batch_size)

      logits, return_dict = model(x, lengths)
      # attn = return_dict['attn'].squeeze()

      # Bookkeeping and cast label to float
      accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
      if logits.shape[-1] == 1:
        # binary cross entropy, cast labels to float
        y = y.type(torch.float)

      print("[Batch]: {}/{} in {:.5f} seconds".format(
            batch_num, len(data), time.time() - t), end='\r', flush=True)

  print("[Accuracy]: {}/{} : {:.3f}%".format(
        accuracy, len(data) * data.batch_size, accuracy / len(data) / data.batch_size * 100))
  print(confusion_matrix)
  result_dict = {'loss': 0.}
  return result_dict

# For regression & classification
def train(model, data, optimizer, criterion, args, meta):
  model.train()

  accuracy, confusion_matrix = 0, np.zeros((meta.num_labels, meta.num_labels), dtype=int)
  total_loss = 0.

  for batch_num, batch in enumerate(data):

    t = time.time()
    # Unpack batch & cast to device
    (x, lengths), y = batch.text, batch.label

    y = y.squeeze() # y needs to be a 1D tensor for xent(batch_size)

    logits, return_dict = model(x, lengths)

    # Bookkeeping and cast label to float
    accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
    if logits.shape[-1] == 1:
      # binary cross entropy, cast labels to float
      y = y.type(torch.float)

    loss = criterion(logits.view(-1, meta.num_targets).squeeze(), y)

    total_loss += float(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    print("[Batch]: {}/{} in {:.5f} seconds".format(
          batch_num, len(data), time.time() - t), end='\r', flush=True)
  result_dict = {
    'loss' : total_loss / len(data) / data.batch_size
  }
  return result_dict

def interpret_evaluate(interpreters, model, data, args, meta, use_rationales=True):
  model.train()

  attributions = {k:[] for k in interpreters}
  rationales = [] # Will be of variable length (slice based on lengths)

  for batch_num, batch in enumerate(data):

    t = time.time()

    # Unpack batch & cast to device
    (x, lengths), y = batch.text, batch.label
    # print(x.shape)
    if use_rationales:
      rationale = batch.rationale.detach().cpu().numpy() # These are padded to the batch length
      rationales.extend([r[:l] for r, l in zip(rationale, lengths)])

    for k, interpreter in interpreters.items():
      # print(lengths.shape)
      batch_attributions = interpreter.interpret(x, lengths)
      batch_attributions = batch_attributions.detach().cpu().numpy()

      # attributions[k].extend(batch_attributions)
      # Select only non-padding attributions
      attributions[k].extend([a[:l] for a, l in zip(batch_attributions,lengths)])

    print("[Batch]: {}/{} in {:.5f} seconds".format(
          batch_num, len(data), time.time() - t), end='\r', flush=True)

    model.zero_grad()

  result_dict = {
    'attributions': attributions,
    'rationales': rationales
  }

  #for k, v in attributions.items():
  #  print(k, v[0].shape)

  return result_dict

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

  optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr, weight_decay=args.l2)

  train_iter = make_iterable(train_dataset, device, batch_size=args.batch_size, train=True)
  train_iter_noshuf = make_iterable(train_dataset, device, batch_size=args.batch_size)
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
  interpreters = {i: get_interpreter(i)(model) for i in args.interpreters}
  print(f"Interpreters: {' '.join(list(interpreters.keys()))}")

  # Construct correlation metrics
  correlations = [get_corr(key)() for key in args.correlation_measures]
  print(f"Correlation measures: {correlations}")

  # TODO: check if rationales exist in the dataset
  use_rationales = True if args.data in ['IMDB-rationale'] else False

  if args.ul_epochs == -1:
    ul_epochs = len(train_iter) * args.batch_size // args.query_size - 1 # number of steps to reduce the entire dataset to a single query_size
  else:
    ul_epochs = args.ul_epochs

  loss = 0.
  # The actual training loop
  try:
    best_valid_loss = None
    best_valid_epoch = 0
    best_model = copy.deepcopy(model)

    for ul_epoch in range(1, ul_epochs + 1):
      # Reduce dataset post-train loop

      for epoch in range(1, args.epochs + 1):

        train(model, train_iter, optimizer, criterion, args, meta)

        # Compute importance scores for tokens on all batches of validation split

        result_dict = interpret_evaluate(interpreters, model, val_iter, args, meta, use_rationales=use_rationales)
        # print(result_dict['rationales'])
        # Compute pairwise correlations between interpretability methods
        scores, raw_correlations = pairwise_correlation(result_dict['attributions'], correlations)

        if use_rationales:
          rationale_scores = rationale_correlation(result_dict['attributions'], result_dict['rationales'])
          pprint(rationale_scores)

        print(f"Epoch={epoch}, evaluating on validation set:")
        result_dict = evaluate(model, val_iter, args, meta)
        loss = result_dict['loss']

        if best_valid_loss is None or loss < best_valid_loss:
          best_valid_loss = loss
          best_valid_epoch = epoch
          best_model = copy.deepcopy(model) # clone params of model, this might be slow, maybe dump?


      # Run on train set without shuffling so instance indices are preserved
      train_interpret_scores = interpret_evaluate(interpreters, model, train_iter_noshuf, args, meta, use_rationales=use_rationales)
      train_scores, train_raw_correlations = pairwise_correlation(train_interpret_scores['attributions'], correlations)

      for k in train_raw_correlations:
        per_instance_agreement = train_raw_correlations[k]

      print(len(train_iter) * args.batch_size, len(train_iter_noshuf) * args.batch_size)
      print(len(per_instance_agreement), per_instance_agreement[0])
      min_agreement_indices = np.argsort(per_instance_agreement) # sorted ascending
      worst_agreement = min_agreement_indices[:args.query_size] # Worst query_size instances
      print("Best agreement", per_instance_agreement[min_agreement_indices[-1]])
      print(worst_agreement)

      for instance_index in worst_agreement:
        print(instance_index)
        print(train_dataset[int(instance_index[0])])
        print(train_interpret_scores['attributions'])
        print(per_instance_agreement[instance_index[0]])
        print()
      sys.exit(-1)


  except KeyboardInterrupt:
    print("[Ctrl+C] Training stopped!")

  results = {}
  results['loss'] = loss

  best_model.rnn.flatten_parameters()

  best_model_pack = (best_model, criterion, optimizer)
  return results, best_model_pack

def main():
  args = make_parser()
  dataloader = dataset_loaders[args.data]
  splits, vocab = dataloader()
  if len(splits) == 3:
    train, val, test = splits
  else:
    train, test = splits
    val = test # Change sometime later

  meta = Config()
  meta.num_labels = 2
  meta.num_tokens = len(vocab)
  meta.padding_idx = vocab.get_padding_index()
  meta.vocab = vocab

  experiment(args, meta, train, val, test)

if __name__ == '__main__':
  main()
