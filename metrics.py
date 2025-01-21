"""
metrics.py

This file contains any error metrics used during optimisation.
"""

import numpy as np

def mse(a, b):
    """
    Mean squared error between two samples a and b.
    """
    return np.mean(np.array(list(map(lambda x: x**2, a - b))))

def rmse(a, b):
    """
    Root mean squared error
    """
    return np.sqrt(mse(a,b))
