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

from captum.attr import (
    GradientShap,
    Lime,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerIntegratedGradients,
    TokenReferenceBase,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

from captum._utils.models.linear_model import SkLearnRidge

from podium import BucketIterator

from util import Config
from datasets import *
from model import *
from interpret import visualize_attributions

word_vector_files = {
  'glove' : os.path.expanduser('~/data/vectors/glove.840B.300d.txt')
}

def make_parser():
  parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')
  parser.add_argument('--data', type=str, default='IMDB',
                        help='Data corpus: [IMDB]')

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
  model = JWAttentionClassifier(args, meta)

  return model

def evaluate(model, data, args, meta):
  model.eval()

  accuracy, confusion_matrix = 0, np.zeros((meta.num_labels, meta.num_labels), dtype=int)

  with torch.inference_mode():
    for batch_num, batch in enumerate(data):
      #if batch_num > 100: break # checking beer imitation

      t = time.time()
      model.zero_grad()

      # Unpack batch & cast to device
      (x, lengths), y = batch.text, batch.label

      y = y.squeeze() # y needs to be a 1D tensor for xent(batch_size)

      logits, return_dict = model(x, lengths, pred_only=False)
      attn = return_dict['attn'].squeeze()

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
  result_dict = {}
  return result_dict

def interpret_instance_lime(model, numericalized_instance):
  device = next(iter(model.parameters())).device
  linear_model = SkLearnRidge()
  lime = Lime(model)

  numericalized_instance = numericalized_instance.unsqueeze(0) # Add fake batch dim
  # Feature mask enumerates (word) features in each instance 
  bsz, seq_len = 1, len(numericalized_instance)
  feature_mask = torch.tensor(list(range(bsz*seq_len))).reshape([bsz, seq_len, 1])
  feature_mask = feature_mask.to(device)
  feature_mask = feature_mask.expand(-1, -1, model.encoder.embedding_dim)

  attributions = lime.attribute(numericalized_instance,
                                target=1, n_samples=1000,
                                feature_mask=feature_mask) # n samples arg taken from court of xai

  print(attributions.shape)
  print('Lime Attributions:', attributions)
  return attributions

def interpret_instance_lig(model, numericalized_instance):

  lig = LayerIntegratedGradients(model, model.encoder.embedding) # LIG uses embedding data

  numericalized_instance = numericalized_instance.unsqueeze(0) # Add fake batch dim
  lengths = torch.tensor(len(numericalized_instance)).unsqueeze(0)
  logits, return_dict = model(numericalized_instance, lengths, use_mask=False, pred_only=False)
  pred = logits.squeeze() # obtain prediction
  # print(pred)
  scaled_pred = nn.Sigmoid()(pred).item() # scale to probability

  # Reference indices are just a bunch of padding indices
  token_reference = TokenReferenceBase(reference_token_idx=0) # Padding index is the reference
  reference_indices = token_reference.generate_reference(len(numericalized_instance), 
                                                          device=next(iter(model.parameters())).device).unsqueeze(0)

  attributions, delta = lig.attribute(numericalized_instance, reference_indices,
                                      n_steps=500, return_convergence_delta=True)
  print('IG Attributions:', attributions)
  print('Convergence Delta:', delta)
  return attributions, scaled_pred, delta

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

    logits, return_dict = model(x, lengths, pred_only=False)

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
  val_iter = make_iterable(val_dataset, device, batch_size=args.batch_size)
  test_iter = make_iterable(test_dataset, device, batch_size=args.batch_size)

  if restore is not None:
    m, o, c = restore
    print(f"Loading model from {args.restore}", flush=True)
    # Only restore the main model
    model.load_state_dict(m)
    optimizer.load_state_dict(o)
    criterion.load_state_dict(c)

  loss = 0.
  # The actual training loop
  try:
    best_valid_loss = None
    best_valid_epoch = 0
    best_model = copy.deepcopy(model)

    for epoch in range(1, args.epochs + 1):

      total_time = time.time()

      train(model, train_iter, optimizer, criterion, args, meta)

      total_time = time.time()

      sample_sentence = "this is a very nice movie".split()
      sample_instance = torch.tensor(meta.vocab.numericalize(sample_sentence))
      sample_instance = sample_instance.to(device)

      # Try out various interpretability methods
      # lime_attributions = interpret_instance_lime(model, sample_instance)

      # Layer integrated gradients
      attributions, prediction, delta = interpret_instance_lig(model, sample_instance)
      print(attributions.shape) # B, T, E
      attributions = attributions.sum(dim=2).squeeze(0)
      attributions = attributions / torch.norm(attributions)
      attributions = attributions.cpu().detach().numpy()

      visualize_attributions([
          (attributions,
            prediction,
            str(round(prediction)),
            str(round(prediction)),
            "Pos",
            attributions.sum(),
            sample_sentence,
            delta)
        ])

      print(f"Epoch={epoch}, evaluating on validation set:")
      result_dict = evaluate(model, val_iter, args, meta)
      loss = result_dict['loss']

      if best_valid_loss is None or loss < best_valid_loss:
        best_valid_loss = loss
        best_valid_epoch = epoch
        best_model = copy.deepcopy(model) # clone params of model, this might be slow, maybe dump?

  except KeyboardInterrupt:
    print("[Ctrl+C] Training stopped!")

  results = {}
  results['loss'] = loss

  best_model.encoder.rnn.flatten_parameters()

  best_model_pack = (best_model, criterion, optimizer)
  return results, best_model_pack


def main():
  args = make_parser()
  (train, val, test), vocab = load_imdb()
  meta = Config()
  meta.num_labels = 2
  meta.num_tokens = len(vocab)
  meta.vocab = vocab

  experiment(args, meta, train, val, test)

if __name__ == '__main__':
  main()