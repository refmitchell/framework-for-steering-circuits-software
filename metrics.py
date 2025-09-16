"""
metrics.py

This file contains any error metrics used during optimisation.
"""

import numpy as np

from util import signed_angle

def mse(a, b, angular=False):
    """
    Mean squared error between two samples a and b.
    If angular is set to True, the error will be computed using the perp dot 
    product method to get the signed difference between two angles. 

    :param a: Sample a (Array-like)
    :param b: Sample b (Array-like)
    :param angular: Set True if values are angular, otherwise False
    """
    if angular:
        return np.mean(np.array(list(map(lambda x: x**2, signed_angle(a, b)))))

    return np.mean(np.array(list(map(lambda x: x**2, a - b))))

def rmse(a, b, angular=False):
    """
    Root mean squared error

    :param a: Sample a (Array-like)
    :param b: Sample b (Array-like)
    :param angular: Set True if values are angular, otherwise False
    """
    return np.sqrt(mse(a, b, angular=angular))


