"""F0-estimation module."""
import numpy as np

from modules import autocorrelation as ac
from modules import cepstrum as cp


def f0_autocorrelation(signal, sample_rate, win_len=2048, ol=0.75):
    """f0-estimation using autocorrelation method."""
    hop_len = int(win_len * (1 - ol))
    n_steps = (len(signal) - win_len) // hop_len

    f0s = np.zeros(n_steps)
    for i in range(n_steps):
        seg = signal[i * hop_len: i * hop_len + win_len]
        windowed_seg = seg * np.hamming(win_len)
        corr = ac.autocorr(windowed_seg)
        peak_index = ac.detect_peak_index(corr)
        f0 = sample_rate / peak_index if peak_index else 0
        f0s[i] = f0
    return f0s


def f0_cepstrum(signal, sample_rate, win_len=2048, ol=0.75):
    """f0-estimation using cepstrum."""
    hop_len = int(win_len * (1 - ol))
    n_steps = (len(signal) - win_len) // hop_len
    window = np.hamming(win_len)

    f0s = np.zeros(n_steps)
    for i in range(n_steps):
        seg = signal[i * hop_len: i * hop_len + win_len]
        cepstrum = cp._cepstrum(seg * window)
        lifter = cp._lifter(win_len, sample_rate)
        liftered_cepstrum = cepstrum * lifter
        f0s[i] = sample_rate / liftered_cepstrum.argmax()
    return f0s
