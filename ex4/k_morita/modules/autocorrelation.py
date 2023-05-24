"""Auto-Correlation module."""

import numpy as np


def autocorr(y):
    """Calculate autocorrelation."""
    length = len(y)
    autocorr = [np.sum(y[:length-m] * y[m:]) for m in range(length)]
    return np.array(autocorr)


def detect_peak_index(arr):
    """Detect peak index of array."""
    diff = np.diff(arr)
    peak_indices = np.where(np.diff(np.sign(diff)) < 0)[0] + 1
    peak_values = arr[peak_indices]
    sorted_indices = np.argsort(peak_values)

    if sorted_indices.size > 0:
        return peak_indices[sorted_indices[-1]]
    return None
