"""Cepstrum modeule."""

import numpy as np


def _cepstrum(data):
    spectrum = np.fft.fft(data)
    cepstrum = np.fft.ifft(np.log10(np.abs(spectrum))).real
    return cepstrum


def _lifter(len, sr, low_limit=0.002, high_limit=0.02):
    """Generate lifter."""
    lifter = np.ones(len)
    low_time_limit = int(sr * low_limit)
    high_time_limit = int(sr * high_limit)
    lifter[:low_time_limit] = 0
    lifter[high_time_limit:] = 0
    return lifter
