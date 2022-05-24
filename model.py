from abc import abstractmethod
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from overrides import overrides

from util import create_pad_mask_from_length


class AcquisitionModel:
    @abstractmethod
    def get_encoder_dim(self, **kwargs):
        pass

    @abstractmethod
    def get_encoder(self, **kwargs):
        pass

    @abstractmethod
    def predict_probs(self, **kwargs):
        pass


# Taken from court-of-xai
# Captum expects the input to the model you are interpreting to be one or more
# Tensor objects, but AllenNLP Model classes often take Dict[...] objects as
# their input. To fix this we require the Models that are to be used with Captum
# to implement a set of methods that make it possible to use Captum.
class CaptumCompatible:
    def captum_sub_model(self):
        """
        Returns a PyTorch nn.Module instance with a forward that performs
        the same steps as the Model would normally, but starting from word embeddings.
        As such it accepts FloatTensors as input, which is required for Captum.
        """
        raise NotImplementedError()

    def instances_to_captum_inputs(self, inputs, lengths, labels):
        """
        Converts a set of Instances to a Tensor suitable to pass to the submodule
        obtained through captum_sub_model.
        Returns
          Tuple with (inputs, target, additional_forward_args)
          Both inputs and target tensors should have the Batch dimension first.
          The inputs Tensors should have the Embedding dimension last.
        """
        raise NotImplementedError()


class _CaptumSubModel(torch.nn.Module):
    """Wrapper around model instance

    Required for returning a single output value
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, word_embeddings, lengths=None):
        # print(lengths.shape)
        pred, _ = self.model.forward_inner(
            embedded_tokens=word_embeddings,
            lengths=lengths,
        )
        return pred


#######################################
# Jain & Wallace attention classifier #
#######################################


class JWAttentionClassifier(nn.Module, CaptumCompatible, AcquisitionModel):
    def __init__(self, config, meta):
        super(JWAttentionClassifier, self).__init__()
        # Store vocab for interpretability methods
        self.vocab = meta.vocab
        self.num_targets = meta.num_targets
        self.hidden_dim = config.hidden_dim
        # Initialize embeddings
        self.embedding_dim = config.embedding_dim
        self.embedding = nn.Embedding(
            meta.num_tokens, config.embedding_dim, padding_idx=meta.padding_idx
        )
        if meta.embeddings is not None:
            self.embedding.weight.data.copy_(meta.embeddings)

        if config.freeze:
            self.embedding.weight.requires_grad = False

        # Initialize network
        self.bidirectional = config.bi
        dimension_multiplier = 1 + sum([config.bi])
        self.attention_dim = dimension_multiplier * config.hidden_dim

        self.num_layers = config.num_layers
        self.rnn = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bi,
        )
        self.attention = AdditiveAttention(
            query_dim=self.attention_dim,
            key_dim=self.attention_dim,
            value_dim=self.attention_dim,
        )

        self.decoder = nn.Linear(self.attention_dim, meta.num_targets)

    def encode(self, embedded_tokens, lengths):
        # For captum compatibility: obtain embeddings as inputs,
        # return only the prediction tensor
        # print(embedded_tokens.shape)
        # print(lengths)
        if lengths is None:
            # Assume fully packed batch, [B,T,E]
            # print(embedded_tokens.shape)
            lengths = torch.tensor(embedded_tokens.shape[1])
            # print("S", lengths.shape, lengths)
            lengths = lengths.repeat(embedded_tokens.shape[0])
            # print("S", lengths.shape, lengths)

        lengths = lengths.cpu()

        h = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_tokens, batch_first=True, lengths=lengths,
            enforce_sorted=False
        )
        # print("FI", embedded_tokens.shape)
        o, h = self.rnn(h) # 
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o, batch_first=False)

        # o, h = self.rnn(embedded_tokens.transpose(0,1))


        if isinstance(h, tuple):  # LSTM
            h = h[1]  # take the cell state

        if self.bidirectional:  # need to concat the last 2 hidden layers
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

        # Also return RNN outputs for weight tying
        return attn_weights, hidden, o

    def forward_inner(self, embedded_tokens, lengths):
        _, hidden, rnn_outputs = self.encode(embedded_tokens, lengths)

        # Perform decoding
        pred = self.decoder(hidden)  # [Bx1]

        return pred, (hidden, rnn_outputs)

    def forward(self, inputs, lengths=None):
        # inputs = [BxT]
        e = self.embedding(inputs)  # [BxTxE]

        pred, (hidden, rnn_outputs) = self.forward_inner(e, lengths)

        # For additional return arguments
        return_dict = {
            "embeddings": e,
            "encoded": hidden,
            "hiddens": rnn_outputs
        }

        return pred, return_dict

    def predict_probs(self, inputs, lengths=None):
        with torch.inference_mode():
            logits, _ = self(inputs, lengths)
            if self.num_targets == 1:
                # Binary classification
                y_pred = torch.sigmoid(logits)
                y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
            else:
                # Multiclass classification
                y_pred = F.softmax(logits, dim=1)
            return y_pred

    def get_encoder_dim(self):
        return self.attention_dim

    def get_encoded(self, inputs, lengths=None):
        with torch.inference_mode():
            e = self.embedding(inputs)
            _, hidden, _ = self.encode(e, lengths)
            return hidden

    @overrides
    def captum_sub_model(self):
        return _CaptumSubModel(self)

    @overrides
    def instances_to_captum_inputs(self, inputs, lengths, labels=None):
        # Technically just does an index -> embedding map right now
        # inputs: [BxT]
        with torch.no_grad():
            e = self.embedding(inputs)
        # pad_mask = create_pad_mask_from_length(inputs, lengths)

        return e, labels, (lengths)


class AdditiveAttention(nn.Module):
    """Tanh attention; query is a learned parameter (same as JW paper)"""

    def __init__(self, query_dim, key_dim, value_dim):
        super(AdditiveAttention, self).__init__()
        assert (key_dim) % 2 == 0, "Key dim should be divisible by 2"
        self.hidden_dim = (key_dim) // 2
        self.k2h = nn.Linear(key_dim, self.hidden_dim)
        self.h2e = nn.Linear(self.hidden_dim, 1, bias=False)
        # Not used right now, but consider using it as attention is flat
        # torch.nn.init.normal_(self.h2e.weight)

    def forward(self, query, keys, values, attn_mask=None, permutation=None):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)
        t, b, _ = keys.shape

        keys = keys.transpose(0, 1)  # [TxBxK] -> [BxTxK]

        h = torch.tanh(self.k2h(keys))  # project into hidden, [BxTxH]

        energy = self.h2e(h).transpose(1, 2)  #  [BxTx1?] -> [Bx1xT]

        # Masked softmax
        if attn_mask is not None:
            energy.masked_fill_(~attn_mask.unsqueeze(1), -float("inf"))
        energy = F.softmax(energy, dim=2)  # scale, normalize

        values = values.transpose(0, 1)  # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(
            1
        )  # [Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination


class MLP(nn.Module, CaptumCompatible, AcquisitionModel):
    def __init__(self, config, meta):
        super().__init__()
        # Store vocab for interpretability methods
        self.vocab = meta.vocab
        self.num_targets = meta.num_targets
        self.hidden_dim = config.hidden_dim
        # Initialize embeddings
        self.embedding_dim = config.embedding_dim
        self.embedding = nn.Embedding(
            meta.num_tokens, config.embedding_dim, padding_idx=meta.padding_idx
        )
        if meta.embeddings is not None:
            self.embedding.weight.data.copy_(meta.embeddings)

        if config.freeze:
            self.embedding.weight.requires_grad = False

        self.hidden = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, meta.num_targets)

    def encode(self, embedded_tokens, lengths):
        # Reduce time dimension
        mean_emb = embedded_tokens.mean(dim=1)
        hidden = self.hidden(mean_emb)
        return hidden

    def forward_inner(self, embedded_tokens, lengths):
        hidden = self.encode(embedded_tokens, lengths)
        pred = self.decoder(hidden)  # [Bx1]

        return pred, hidden

    def forward(self, inputs, lengths=None):
        # inputs = [BxT]
        e = self.embedding(inputs)  # [BxTxE]

        pred, hidden = self.forward_inner(e, lengths)

        # For additional return arguments
        return_dict = {
            "embeddings": e,
            "encoded": hidden,
        }

        return pred, return_dict

    def get_encoder_dim(self):
        return self.hidden_dim

    def get_encoded(self, inputs, lengths=None):
        with torch.inference_mode():
            e = self.embedding(inputs)
            hidden = self.encode(e, lengths)
            return hidden

    def predict_probs(self, inputs, lengths=None):
        with torch.inference_mode():
            logits, _ = self(inputs, lengths)
            if self.num_targets == 1:
                # Binary classification
                y_pred = torch.sigmoid(logits)
                y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
            else:
                # Multiclass classification
                y_pred = F.softmax(logits, dim=1)
            return y_pred

    @overrides
    def captum_sub_model(self):
        return _CaptumSubModel(self)

    @overrides
    def instances_to_captum_inputs(self, inputs, lengths, labels=None):
        # Technically just does an index -> embedding map right now
        # inputs: [BxT]
        with torch.no_grad():
            e = self.embedding(inputs)
        # pad_mask = create_pad_mask_from_length(inputs, lengths)

        return e, None, (lengths)
