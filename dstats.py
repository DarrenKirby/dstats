"""
Implementation of some common statistics formulas
"""

from typing import Sequence, Union
import numpy as np


# 1. Measures of central tendancy

def mean(x: Sequence) -> float:
    "Return the arithmetic mean of sequence x"
    x_arr = np.array(x)
    x_sum = np.sum(x)
    return x_sum / x_arr.size


def median(x: Union[Sequence, np.ndarray]) -> float:
    "Return the median value of sequence x"
    x_arr = np.array(x)
    x_arr.sort(kind='mergesort')
    length = x_arr.size
    if length % 2 != 0:
        return x_arr[length//2].item()
    return ((x_arr[length//2] + x_arr[(length//2)-1]) / 2)


def mode(x: Sequence) -> list:
    """
    Return the mode(s) of sequence x

    The mode may have more than one value, and thus, for consistency
    zero or more modes are returned in a list no matter the length.
    """
    arr = np.array(x)
    unique, counts = np.unique(arr, return_counts=True)
    max_count = np.max(counts)

    # If all counts are 1, then all elements are unique, hence no mode
    if max_count == 1:
        return []

    mode_indices = np.where(counts == max_count)[0]
    return unique[mode_indices].tolist()


# 2. Measures of dispersion

def variance(x: Sequence, ddof=1) -> float:
    """
    Return the variance of sequence x

    The optional ddof argument is delta degrees of freedom. It defaults
    at 1. Leave at the default for sample variance,
    and pass a 0 for population variance
    """
    arr = np.array(x)
    arr = arr - arr.mean()
    arr = arr ** 2
    return np.sum(arr) / (arr.size - ddof)


def std(x: Sequence, ddof: int=1) -> float:
    """
    Return the standard deviation of sequence x

    The optional ddof argument is delta degrees of freedom. It defaults
    at 1. Leave at the default for sample variance,
    and pass a 0 for population variance
    """
    return np.sqrt(variance(x, ddof=ddof))


def fano_factor(x: Sequence) -> float:
    "Return the Fano factor of sequence x."
    return variance(x, ddof=1) / mean(x)


def coefficient_of_variance(x: Sequence) -> float:
    "Return the coefficient of variance of sequence x."
    m = mean(x)
    if m <= 0:
        raise ValueError('Cannot calculate COV of sequence with negative mean')
    return std(x) / m


def iqr(x: Sequence) -> list:
    """
    Return the inter-quartile range and quartiles of sequence x.

    The return value is a list with the IQR at indice [0], and each of
    the first, second, and third quartiles respectively. Values may be
    integers, floats, or a mix of both depending on the sequence.
    """
    arr = np.array(x)
    arr.sort(kind='mergesort')
    q2 = median(arr)
    arr_l = arr[0:int(arr.size/2)]

    if arr.size % 2 == 0:
        arr_r = arr[int(arr.size/2):]
    # If n is odd we leave the median
    else:
        arr_r = arr[int(arr.size/2+1):]

    q1 = median(arr_l)
    q3 = median(arr_r)

    return [q3 - q1, q1, q2, q3]


def corr(x: Sequence, y: Sequence) -> float:
    "Return the Pearson correlation coefficient for sequences x and y."
    x_arr = np.array(x)
    y_arr = np.array(y)

    x_centered = x_arr - np.mean(x_arr)
    y_centered = y_arr - np.mean(y_arr)

    ut = np.dot(x_centered, y_centered)

    # Compute the L2 norms (magnitudes) of x and y after centering
    norm_x = np.sqrt(np.dot(x_centered, x_centered))
    norm_y = np.sqrt(np.dot(y_centered, y_centered))

    lt = norm_x * norm_y

    # Avoid division by zero
    if lt == 0:
        return 0.0

    return (ut / lt).item()


# 3. Measures of shape/symmetry

def skew(x: Sequence) -> float:
    "Return the skew aka third moment of sequence x."
    arr = np.array(x)
    arr = arr - arr.mean()
    arr = arr ** 3
    return np.sum(arr) / (arr.size * std(x) ** 3)


def kurtosis(x: Sequence, fisher: bool=True) -> float:
    "Return the kurtosis aka fourth moment of sequence x."
    arr = np.array(x)
    arr = arr - arr.mean()
    arr = arr ** 4
    k = np.sum(arr) / (arr.size * std(x) ** 4)
    return k - 3.0 if fisher else k


# 4. Transforms

def zscore(x: Sequence) -> Sequence[float]:
    "Return the z-score values of a sequence"
    arr = np.array(x)
    return (arr - arr.mean()) / arr.std(ddof=1)


def r_zscore(x: Sequence, mu: float, sigma: float) -> Sequence[float]:
    """
    Transform z-score data to its original values

    This transform requires the mean and standard deviation
    of the original distribution to reconstruct
    """
    arr = np.array(x)
    return (arr + mu) * sigma


def mod_zscore(x: Sequence) -> Sequence[float]:
    "Return the modified z-score values of a sequence"
    arr = np.array(x)
    mad = median_abs_dev(arr)
    return .6745*(x - median(x)) / mad


def mean_abs_diff(x: Sequence, ddof: int=1) -> float:
    "Return the mean absolute difference"
    arr = np.array(x)
    arr = arr - arr.mean()
    arr = np.abs(arr)
    return np.sum(arr) / (arr.size - ddof)


def median_abs_dev(x: Sequence) -> float:
    "Return the median absolute deviation"
    arr = np.array(x)
    return median(np.abs(x - median(x)))


def fisherz(x: Sequence) -> Sequence:
    "Fisher-z transform dataset"
    arr = np.array(x)
    # We have to scale to -1/1 for the transform but
    # using -1/1 exactly will produce infs from log()
    arr = minmax_scale(arr, -0.99999999, 0.99999999)
    return  0.5 * np.log((1 + arr) / (1 - arr))


def fisherz_scalar(x: float) -> float:
    "Fisher-z transform a scalar"
    if x <= -1 or x >= 1:
        raise ValueError("Scalar must be in open interval (-1,1)")
    return 0.5 * np.log((1 + x) / (1 - x))


def unity_scale(x: Sequence) -> Sequence:
    "Scale dataset to between 0 and 1"
    arr = np.array(x)
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)


def minmax_scale(x: Sequence, n_min: float, n_max: float):
    "scale dataset to arbitrary min/max"
    arr = np.array(x)
    unity_arr = unity_scale(arr)
    return n_min + (n_max - n_min) * unity_arr


def moments(x: Sequence) -> Sequence:
    "Return the first four statistical moments"
    return [mean(x), variance(x), skew(x), kurtosis(x)]



# Aliases
moment1 = mean
moment2 = variance
moment3 = skew
moment4 = kurtosis
