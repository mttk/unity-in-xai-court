import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Config(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError

    def __setattr__(self, key, value):
        self[key] = value


def create_pad_mask_from_length(tensor, lengths):
    # Creates a mask where `True` is on the non-padded locations
    # and `False` on the padded locations
    mask = torch.arange(tensor.size(-1))[None, :].to(lengths.device) < lengths[:, None]
    mask = mask.to(tensor.device)
    return mask


def generate_permutation(batch, lengths):
    # batch contains attention weights post scaling
    # [BxT]
    batch_size, max_len = batch.shape
    # repeat arange for batch_size times
    perm_idx = np.tile(np.arange(max_len), (batch_size, 1))

    for batch_index, length in enumerate(lengths):
        perm = np.random.permutation(length.item())
        perm_idx[batch_index, :length] = perm

    return torch.tensor(perm_idx)


def replace_with_uniform(tensor, lengths):
    # Assumed: [BxT] shape for tensor
    uniform = create_pad_mask_from_length(tensor, lengths).type(torch.float)
    for idx, l in enumerate(lengths):
        uniform[idx] /= l
    return uniform


def masked_softmax(attn_odds, masks):
    attn_odds.masked_fill_(~masks, -float("inf"))
    attn = F.softmax(attn_odds, dim=-1)
    return attn


def create_pad_mask_from_length(tensor, lengths, idx=-1):
    # Creates a mask where `True` is on the non-padded locations
    # and `False` on the padded locations
    mask = torch.arange(tensor.size(idx))[None, :].to(lengths.device) < lengths[:, None]
    mask = mask.to(tensor.device)
    return mask
