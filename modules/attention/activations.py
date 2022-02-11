"""Suite of activation functions to convert attention weights to probability distributions"""
from overrides import overrides

from allennlp.nn.util import replace_masked_values
from allennlp.nn import util

from entmax import entmax15, sparsemax, entmax_bisect

import torch
import torch.nn as nn


class AttentionActivationFunction(nn.Module):
    """Attention activation function base class"""

    def forward(self, scores, mask):
        """Map a score vector to a probability distribution"""
        raise NotImplementedError("Implement forward Model")

class UniformActivation(AttentionActivationFunction):
    @overrides
    def forward(self, scores, mask):
        """Map a score vector to the uniform probability distribution

        Args:
            scores (torch.Tensor): (Batch x Sequence Length)
                Attention scores (also referred to as weights)
            mask (torch.BoolTensor): (Batch x Sequence Length)
                Specifies which indices are just padding

        Returns:
            torch.Tensor: the Uniform distribution
        """
        lengths = mask.float().sum(dim=-1, keepdim=True)
        scores = 1.0 / lengths
        uniform = mask.float() * scores
        return uniform


class SoftmaxActivation(AttentionActivationFunction):
    @overrides
    def forward(self, scores, mask):
        """Map a score vector to a dense probability distribution

        Args:
            scores (torch.Tensor): (Batch x Sequence Length)
                Attention scores (also referred to as weights)
            mask (torch.BoolTensor): (Batch x Sequence Length)
                Specifies which indices are just padding

        Returns:
            torch.Tensor: Dense distribution
        """
        masked_scores = replace_masked_values(scores, mask, -float("inf"))
        return torch.nn.Softmax(dim=-1)(masked_scores)


class SparsemaxActivation(AttentionActivationFunction):
    @overrides
    def forward(self, scores, mask):
        """Map a score vector to a sparse probability distribution

        Args:
            scores (torch.Tensor): (Batch x Sequence Length)
                Attention scores (also referred to as weights)
            mask (torch.BoolTensor): (Batch x Sequence Length)
                Specifies which indices are just padding

        Returns:
            torch.Tensor: Sparse distribution
        """
        masked_scores = replace_masked_values(scores, mask, -float("inf"))
        return sparsemax(masked_scores, dim=-1)


class Entmax15Activation(AttentionActivationFunction):
    @overrides
    def forward(self, scores, mask):
        """Map a score vector to a probability distribution halfway between softmax and sparsemax

        Args:
            scores (torch.Tensor): (Batch x Sequence Length)
                Attention scores (also referred to as weights)
            mask (torch.BoolTensor): (Batch x Sequence Length)
                Specifies which indices are just padding

        Returns:
            torch.Tensor: Distribution halfway between softmax and sparsemax
        """
        masked_scores = replace_masked_values(scores, mask, -float("inf"))
        return entmax15(masked_scores, dim=-1)


class EntmaxAlphaActivation(AttentionActivationFunction):

    def __init__(self, alpha: float):
        super().__init__()

        # pylint: disable=E1102 
        # (https://github.com/pytorch/pytorch/issues/24807)
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, requires_grad=True))

    @overrides
    def forward(self, scores, mask):
        """Map a score vector to a probability distribution akin to softmax (alpha=1) and sparsemax (alpha=2)

        Args:
            scores (torch.Tensor): (Batch x Sequence Length)
                Attention scores (also referred to as weights)
            mask (torch.BoolTensor): (Batch x Sequence Length)
                Specifies which indices are just padding

        Returns:
            torch.Tensor: Distribution resulting from entmax with specified alpha
        """
        # Entmax is only defined for alpha > 1
        self.alpha.data = torch.clamp(self.alpha.data, min=1.001)

        masked_scores = replace_masked_values(scores, mask, -float("inf"))
        return entmax_bisect(masked_scores, self.alpha, dim=-1)
