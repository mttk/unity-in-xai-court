"""
Taken from court-of-xai
Measures to calculate the correlation between the top-k elements of two identical length lists of scores
"""

from dataclasses import dataclass, field
from collections import Counter
import itertools
import math
import numbers
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator

import numpy as np
from overrides import overrides
from pyircor.tauap import tauap_b
import scipy.stats as stats
from scipy.stats import kendalltau, spearmanr, pearsonr, weightedtau


def unordered_cartesian_product(x: List[Any], y: List[Any]) -> Iterator[List[Tuple[Any, Any]]]:
    """Yield all unique unordered pairs from the Cartesian product of x and y"""
    seen = set()
    for (i, j) in itertools.product(x, y):
        if i == j or (i, j) in seen:
            continue
        seen.add((i, j))
        seen.add((j, i))

        yield (i, j)

def bucket_order(x: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Associate a top-k full (no ties) or partial ranking (with ties) with a bucket order per Fagin et al. (2004)

    Args:
        x (np.ndarray): A full or partial ranked list
        k (int): Only the indices elements of x in the top k buckets are returned. Defaults to the size of x.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Full bucket ranking, Indices of the elements of x in the top-k buckets
    """
    if k is None:
        k = x.size

    x_ranked = stats.rankdata(x, method='dense')
    rank_count = Counter(x_ranked)
    unique_ranks = sorted(list(set(x_ranked)), reverse=True)

    # The position of bucket B_i is the average location within bucket B_i
    bucket_sizes = [rank_count[rank] for rank in unique_ranks]
    bucket_positions = []
    def get_bucket_position(bucket_sizes, bucket_index):
        return sum(bucket_sizes[:bucket_index]) + (bucket_sizes[bucket_index] + 1) / 2
    bucket_positions = {rank: get_bucket_position(bucket_sizes, i) for i, rank in zip(range(len(bucket_sizes)), unique_ranks)}

    bucket_order = np.array([bucket_positions[i] for i in x_ranked])

    top_k_bucket_positions = sorted(bucket_positions.values())[:k]
    top_k = [i for i, bp in enumerate(x_ranked) if bucket_positions[bp] in top_k_bucket_positions]

    return bucket_order, top_k


def kendall_top_k(a: Any, b: Any, k: int = None, kIsNonZero: bool = False, p: float = 0.5) -> Tuple[float, int]:
    """
    Compute the top-k kendall-tau correlation metric for the given full (no ties) or partial (with ties) ranked lists

    Args:
        a (ArrayLike):
            The first ranked list. Can be any array-like type (e.g. list, numpy array or cpu-bound tensor)
        b (ArrayLike):
            The second ranked list. Can be any array-like type (e.g. list, numpy array or cpu-bound tensor)
        k (int, optional):
            Only the top "k" elements are compared. Defaults to the size of the first list
        kIsNonZero (bool, optional):
            If specified, overrides k to be the minimum number of non-zero elements in either list. Defaults to False
        p (float, optional):
            The penalty parameter in the range (0, 1]. This is a metric if p is in the range [1/2, 1] and a near metric
            if p is in the range (0, 1/2). Defaults to the neutral case, p = 1/2

    Raises:
        ValueError: If p is not defined as described or if the lists are not equal in length

    Returns:
        Tuple[float, int]: A tuple of the computed correlation and the value used for k
    """
    if not (isinstance(p, numbers.Real) and p > 0 and p <= 1):
        raise ValueError("The penalty parameter p must be numeric and in the range (0,1]")

    x = np.array(a).ravel()
    y = np.array(b).ravel()

    if x.size != y.size:
        raise ValueError("The ranked lists must have same lengths")

    if kIsNonZero:
        k = min(np.count_nonzero(x), np.count_nonzero(y))
    elif k is None:
        k = x.size

    k = min(k, x.size)

    x_bucket_order, x_top_k = bucket_order(x, k)
    y_bucket_order, y_top_k = bucket_order(y, k)

    kendall_distance = 0
    normalization_constant = 0

    for i, j in unordered_cartesian_product(x_top_k, y_top_k):

        normalization_constant += 1

        i_bucket_x = x_bucket_order[i]
        j_bucket_x = x_bucket_order[j]
        i_bucket_y = y_bucket_order[i]
        j_bucket_y = y_bucket_order[j]

        # Case 1: i and j are in different buckets in both x and y: penalty = 1
        if i_bucket_x != j_bucket_x and i_bucket_y != j_bucket_y:
            opposite_order_x = i_bucket_x > j_bucket_x and i_bucket_y < j_bucket_y
            opposite_order_y = i_bucket_x < j_bucket_x and i_bucket_y > j_bucket_y
            if opposite_order_x or opposite_order_y:
                kendall_distance += 1

        # Case 2: i and j are in the same bucket in both x and y: penalty = 0 (so we can ignore)

        # Case 3: i and j are in the same bucket in one of the partial rankings, but in different buckets in the other
        # penalty = p
        elif (i_bucket_x == j_bucket_x and i_bucket_y != j_bucket_y) or (i_bucket_y == j_bucket_y and i_bucket_x != j_bucket_x):
            kendall_distance += p


    # Normalize to range [-1, 1]
    correlation = kendall_distance / max(1, normalization_constant)
    correlation *= -2
    correlation += 1

    return (correlation, k)


def enforce_same_shape(func):
    def wrapper_enforce_same_shape(*args, **kwargs):
        rel_args = args[1:] # skip self
        all_same_shape = all(_arg.shape == rel_args[0].shape for _arg in rel_args)
        if not all_same_shape:
            raise ValueError(f"All arguments must have the same shape")
        return func(*args, **kwargs)
    return wrapper_enforce_same_shape


@dataclass
class CorrelationResult:
    correlation: float
    k: int = field(default=None)


CorrelationMap = Dict[str, CorrelationResult]


class CorrelationMeasure:
    """
    A uniquely identifiable measure to calculate correlation(s) between the top-k elements of two identical
    length lists of scores

    Args:
        identifier (str):
            Unique name of the measure
        unfair_in_isolation (Optional[bool]):
            This metric uses a dynamic value for k which may produce correlations that cannot be compared with each
            other directly. If so, the caller is responsible for passing in a 'fair' override for k when required.

            For example: sparse attention distributions may produce scores of zero but feature importance measures
            do not. Thus, if we are calculating the KendallTauTopKNonZero metric and want an "apples to apples"
            comparison we must ensure the correlation calculation between two feature importance measures uses the
            average k value from correlation calculations with at least one attention interpreter.

            Defaults to False.
    """
    def __init__(self, identifier: str, unfair_in_isolation: Optional[bool] = False): 
        self._id = identifier
        self.unfair_in_isolation = unfair_in_isolation

    @property
    def id(self):
        return self._id

    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        """
        Calculates a number of uniquely identifiable correlations between the top-k elements of two
        identical length lists of scores. For example, if the measure calculates the correlation between the
        top 10% of scores in a and b and the top 20% of scores the return value would look something like:

        {
            '{id}_top_10_percent': ...
            '{id}_top_20_percent': ...
        }

        Args:
            a (np.ndarray): List of scores. Same length as b.
            b (np.ndarray): List of scores. Same length as a.

        Returns:
            CorrelationMap: Mapping of some identifier to a CorrelationResult
        """
        raise NotImplementedError("Implement correlation calculation")


class KendallTau(CorrelationMeasure):

    def __init__(self):
        super().__init__(identifier="kendall_tau")

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        kt, _ = kendalltau(a, b)
        return {
            self.id: CorrelationResult(correlation=kt, k=len(a))
        }


class SpearmanRho(CorrelationMeasure):

    def __init__(self):
        super().__init__(identifier="spearman_rho")

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        sr, _ = spearmanr(a, b)
        return {
            self.id: CorrelationResult(correlation=sr, k=len(a))
        }


class PearsonR(CorrelationMeasure):

    def __init__(self):
        super().__init__(identifier="pearson_r")

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        pr, _ = pearsonr(a, b)
        return {
            self.id: CorrelationResult(correlation=pr, k=len(a))
        }


class KendallTauTopKVariable(CorrelationMeasure):

    def __init__(self, percent_top_k: List[float]):
        super().__init__(identifier="kendall_top_k_variable")
        self.variable_lengths = percent_top_k

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        results = {}
        for variable_length in self.variable_lengths:
            k = max(1, math.floor(len(a) * variable_length))
            kt_top_k, k = kendall_top_k(a=a, b=b, k=k)
            results[f"{self.id}_{variable_length}"] = CorrelationResult(correlation=kt_top_k, k=k)
        return results


class KendallTauTopKFixed(CorrelationMeasure):

    def __init__(self, fixed_top_k: List[int]):
        super().__init__(identifier="kendall_top_k_fixed")
        self.fixed_lengths = fixed_top_k

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        results = {}
        for fixed_length in self.fixed_lengths:
            kt_top_k, k = kendall_top_k(a=a, b=b, k=fixed_length)
            results[f"{self.id}_{fixed_length}"] = CorrelationResult(correlation=kt_top_k, k=k)
        return results


class KendallTauTopKNonZero(CorrelationMeasure):

    def __init__(self):
        super().__init__(identifier="kendall_top_k_non_zero", unfair_in_isolation=True)

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:

        # k may be explicitly specified in some cases to ensure fair comparisons
        k = kwargs.get("k")
        kIsNonZero = (k==None)

        kt_top_k, k = kendall_top_k(a=a, b=b, kIsNonZero=kIsNonZero, k=k)
        return {
            self.id: CorrelationResult(correlation=kt_top_k, k=k)
        }


class WeightedKendallTau(CorrelationMeasure):

    def __init__(self, alphas: List[float]):
        super().__init__(identifier="weighted_kendall_tau")
        self.alphas = alphas

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        results = {}
        for alpha in self.alphas:
            weigher = lambda x: (1 / (x + 1) ** alpha)
            wkt, _ = weightedtau(a, b, weigher=weigher)
            results[f"{self.id}_{alpha}"] = CorrelationResult(correlation=wkt, k=len(a))
        return results


class KendallTauAPB(CorrelationMeasure):

    def __init__(self):
        super().__init__(identifier="kendall_tau_ap_b")

    @enforce_same_shape
    @overrides
    def correlation(self, a: np.ndarray, b: np.ndarray, **kwargs) -> CorrelationMap:
        tau_ap_b = tauap_b(a, b)
        return {
            self.id: CorrelationResult(correlation=tau_ap_b, k=len(a))
        }
