import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os

RNNS = ['LSTM', 'GRU', 'RNN']


def generate_permutation(batch, lengths):
  # batch contains attention weights post scaling
  # [BxT]
  batch_size, max_len = batch.shape
  # repeat arange for batch_size times
  perm_idx = np.tile(np.arange(max_len), (batch_size, 1))

  for batch_index, length in enumerate(lengths):
    perm = np.random.permutation(length.item())
    #print(perm)
    perm_idx[batch_index, :length] = perm

  #permuted_batch = batch.gather(1, perm_idx)
  return torch.tensor(perm_idx)

def replace_with_uniform(tensor, lengths):
  # Assumed: [BxT] shape for tensor
  uniform = create_pad_mask_from_length(tensor, lengths).type(torch.float)
  for idx, l in enumerate(lengths):
    uniform[idx] /= l
  return uniform

def masked_softmax(attn_odds, masks) :
  attn_odds.masked_fill_(~masks, -float('inf'))
  attn = F.softmax(attn_odds, dim=-1)
  return attn

def create_pad_mask_from_length(tensor, lengths, idx=-1):
  # Creates a mask where `True` is on the non-padded locations
  # and `False` on the padded locations
  mask = torch.arange(tensor.size(idx))[None, :].to(lengths.device) < lengths[:, None]
  mask = mask.to(tensor.device)
  return mask

class _CaptumSubModel(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, word_embeddings, lengths=None):
    if lengths is None and word_embeddings.shape[0] == 1:
      # Assume it's a single instance
      lengths = torch.tensor(word_embeddings.shape[1]).unsqueeze(0).unsqueeze(0)
      print(lengths.shape, word_embeddings.shape, lengths)

    return self.model.forward_inner(
        embedded_tokens=word_embeddings,
        lengths=lengths,
    )


class JWAttentionClassifier(nn.Module):
  def __init__(self, config, meta):
    super(JWAttentionClassifier, self).__init__()

    self.hidden_dim = config.hidden_dim
    self.embedding_dim = config.embedding_dim
    self.embedding = nn.Embedding(meta.num_tokens, config.embedding_dim,
                                  padding_idx=meta.padding_idx)

    # TODO: add word vectors
    #if config.vectors in word_vector_files:
    #  self.embedding.weight.data.copy_(meta.vectors)

    if config.freeze:
      self.embedding.weight.requires_grad = False

    self.bidirectional = config.bi
    dimension_multiplier = 1 + sum([config.bi])
    attention_dim = dimension_multiplier * config.hidden_dim

    self.rnn_type = config.rnn_type
    assert config.rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
    rnn_cell = getattr(nn, config.rnn_type) # fetch constructor from torch.nn

    self.num_layers = config.num_layers
    self.rnn = rnn_cell(config.embedding_dim, config.hidden_dim, config.num_layers, 
                        dropout=config.dropout, bidirectional=config.bi)
    self.attention = QlessAdditiveAttention(query_dim=attention_dim,
                                            key_dim=attention_dim,
                                            value_dim=attention_dim)

    self.decoder = nn.Linear(attention_dim, meta.num_targets)

  def forward_inner(self, embedded_tokens, lengths):
    # For captum compatibility: obtain embeddings as inputs,
    # return only the prediction tensor

    h = torch.nn.utils.rnn.pack_padded_sequence(embedded_tokens, batch_first=True, lengths=lengths)
    o, h = self.rnn(h)
    o, _ = torch.nn.utils.rnn.pad_packed_sequence(o, batch_first=False)

    if isinstance(h, tuple): # LSTM
      h = h[1] # take the cell state

    if self.bidirectional: # need to concat the last 2 hidden layers
      h = torch.cat([h[-1], h[-2]], dim=1)
    else:
      h = h[-1]

    m = None
    # m = create_pad_mask_from_length(embedded_tokens, lengths)
#    if p_mask is not None:
#      #print(m.shape, p_mask.shape)
#      m = m & ~p_mask.transpose(0,1)


    # Perform self-attention
    # print(h.shape, o.shape) # m = 32, 300
    attn_weights, hidden = self.attention(h, o, o, attn_mask=m)

    # Perform decoding
    pred = self.decoder(hidden) # [Bx1]

    return pred

  def forward(self, inputs, lengths=None):
    # print(inputs.shape, lengths.shape)
    # inputs = [BxT]
    e = self.embedding(inputs)
    # e = [BxTxE]

    if lengths is None:
      # Assume fully packed batch
      lengths = torch.tensor(x.shape[1])
      lengths = lengths.repeat(x.shape[0])

    lengths = lengths.cpu()

    pred = self.forward_inner(e, lengths)

    return_dict = {
        'embeddings': e
    }

    return pred, return_dict

  def captum_sub_model(self):
    return _CaptumSubModel(self)

  #def instances_to_captum_inputs(self, labeled_instances):
    # Should map instances to word embedded inputs; TODO


class QlessAdditiveAttention(nn.Module):
  def __init__(self, query_dim, key_dim, value_dim):
    super(QlessAdditiveAttention, self).__init__()
    assert (key_dim) % 2 == 0, "Key dim should be divisible by 2"
    self.hidden_dim = (key_dim) // 2
    self.k2h = nn.Linear(key_dim, self.hidden_dim)
    self.h2e = nn.Linear(self.hidden_dim, 1, bias=False)
    # Not used right now, but consider using it as attention is flat
    #torch.nn.init.normal_(self.h2e.weight)


  def forward(self, query, keys, values, attn_mask=None, permutation=None):
    # Query = [BxQ]
    # Keys = [TxBxK]
    # Values = [TxBxV]
    # Outputs = a:[TxB], lin_comb:[BxV]

    # Here we assume q_dim == k_dim (dot product attention)
    t, b, _ = keys.shape

    keys = keys.transpose(0,1) # [TxBxK] -> [BxTxK]

    h = torch.tanh(self.k2h(keys)) # project into hidden, [BxTxH]

    energy = self.h2e(h).transpose(1,2) #  [BxTx1?] -> [Bx1xT] 

    # Masked softmax
    # print(attn_mask.shape, energy.shape)
    if attn_mask is not None:
      energy.masked_fill_(~attn_mask.unsqueeze(1), -float('inf'))
    energy = F.softmax(energy, dim=2) # scale, normalize


    if permutation is not None:
      key, permutation = permutation
      if key == 'uniform':
        energy = permutation.unsqueeze(1)
      elif key == 'permute':
        energy = energy.squeeze().gather(1, permutation).unsqueeze(1)

    values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
    linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
    return energy, linear_combination

class QlessAttention(nn.Module):
  def __init__(self, query_dim, key_dim, value_dim):
    super(QlessAttention, self).__init__()
    self.scale = 1. #/ math.sqrt(key_dim)
    self.k2h = nn.Linear(key_dim, key_dim, bias=False)
    self.learned_q = nn.Linear(key_dim, 1, bias=False)
    torch.nn.init.normal_(self.learned_q.weight)

  def forward(self, query, keys, values, attn_mask=None, permutation=None):
    # Query = [BxQ]
    # Keys = [TxBxK]
    # Values = [TxBxV]
    # Outputs = a:[TxB], lin_comb:[BxV]

    # Here we assume q_dim == k_dim (dot product attention)
    lengths = attn_mask.sum(-1).to(torch.float) # for scaling

    keys = keys.transpose(0,1) # [TxBxK] -> [BxTxK]

    keys = F.normalize(keys, p=2, dim=-1)


    energy = self.learned_q(keys).mul_(self.scale).transpose(1,2) # [BxTxK -> Bx1xT]

    if attn_mask is not None:
      energy.masked_fill_(~attn_mask.unsqueeze(1), -float('inf'))
    energy = F.softmax(energy, dim=2) # scale, normalize

    if permutation is not None:
      key, permutation = permutation
      if key == 'uniform':
        energy = permutation.unsqueeze(1)
      elif key == 'permute':
        energy = energy.squeeze().gather(1, permutation).unsqueeze(1)

    values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
    linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
    return energy, linear_combination


ATTNS = {'nqdot': QlessAttention,
        'nqadd' : QlessAdditiveAttention}
