" reimplementation of common stats functions"
from typing import Sequence, Union
import numpy as np


def mean(x: Sequence) -> float:
    " Return the arithmetic mean of sequence x"
    x_arr = np.array(x)
    x_sum = np.sum(x)
    return x_sum / x_arr.size


def median(x: Union[Sequence, np.ndarray]) -> float:
    " Return the median value of sequence x"
    x_arr = np.array(x)
    x_arr.sort(kind='mergesort')
    length = x_arr.size
    if length % 2 != 0:
        return x_arr[length//2].item()
    return ((x_arr[length//2] + x_arr[(length//2)-1]) / 2).item()


def mode(x: Sequence) -> list:
    """ Return the mode(s) of sequence x

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


def variance(x: Sequence, ddof=1) -> float:
    """ Return the variance of sequence x

    The optional ddof argument is delta degrees of freedom. It defaults
    at 1. Leave at the default for sample variance,
    and pass a 0 for population variance
    """
    arr = np.array(x)
    arr = arr - arr.mean()
    arr = arr ** 2
    return np.sum(arr) / (arr.size - ddof)


def std(x: Sequence, ddof: int=1) -> float:
    """ Return the standard deviation of sequence x

    The optional ddof argument is delta degrees of freedom. It defaults
    at 1. Leave at the default for sample variance,
    and pass a 0 for population variance
    """
    return np.sqrt(variance(x, ddof=ddof))


def fano_factor(x: Sequence) -> float:
    " Return the Fano factor of sequence x."
    return variance(x, ddof=1) / mean(x)


def coefficient_of_variance(x: Sequence) -> float:
    " Return the coefficient of variance of sequence x."
    m = mean(x)
    if m <= 0:
        raise ValueError('Cannot calculate COV of sequence with negative mean')
    return std(x) / m


def iqr(x: Sequence) -> list:
    """ Return the inter-quartile range and quartiles of sequence x.

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
    " Return the Pearson correlation coefficient for sequences x and y."
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


def skew(x: Sequence) -> float:
    " Returns the skew aka third moment of sequence x."
    arr = np.array(x)
    arr = arr - arr.mean()
    arr = arr ** 3
    return np.sum(arr) / (arr.size * std(x) ** 3)


def kurtosis(x: Sequence, fisher: bool=True) -> float:
    " Returns the kurtosis aka fourth moment of sequence x."
    arr = np.array(x)
    arr = arr - arr.mean()
    arr = arr ** 4
    k = np.sum(arr) / (arr.size * std(x) ** 4)
    return k - 3.0 if fisher else k


def zscore(x: Sequence) -> Sequence[float]:
    " Returns an array with the values z-normalized"
    arr = np.array(x)
    return (arr - arr.mean()) / arr.std(ddof=1)


# Aliases
moment1 = mean
moment2 = variance
moment3 = skew
moment4 = kurtosis
