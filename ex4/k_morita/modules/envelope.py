"""Spectrum envelope module."""

import numpy as np
import scipy

from modules import autocorrelation as ac
from modules import cepstrum as cp


def spectrum_envelope_cepstrum(signal, sample_rate):
    """Calculate spectrum-envelope using cepstrum."""
    signal_len = len(signal)
    windowed_signal = signal * np.hamming(signal_len)

    spectrum = np.fft.fft(windowed_signal)
    log_spectrum = 20 * np.log10(np.abs(spectrum))
    log_spectrum = log_spectrum[:log_spectrum.shape[0] // 2]

    cepstrum = cp._cepstrum(windowed_signal)
    lifter = cp._lifter(len(cepstrum), sample_rate, low_limit=0.)
    liftered_cept = cepstrum * lifter

    envelope = 20 * np.fft.fft(liftered_cept).real
    envelope = envelope[:envelope.shape[0] // 2]
    freq = np.linspace(0, sample_rate // 2, len(envelope))
    return envelope, log_spectrum, freq


def _levinson_durbin(r, lpc_order):
    """Levinson-Durbin recursion."""
    lpc_coefs = np.zeros(lpc_order + 1)
    e = np.zeros(lpc_order + 1)

    lpc_coefs[0] = 1
    e[0] = r[0]

    for k in range(1, lpc_order + 1):
        lambda_ = - (r[k] + np.sum(lpc_coefs[1:k+1] * r[k-1::-1])) / e[k-1]
        temp_coefs = np.r_[lpc_coefs[:k+1], np.zeros(lpc_order-k)]
        lpc_coefs = temp_coefs + lambda_ * \
            np.r_[0, temp_coefs[:k][::-1], np.zeros(lpc_order-k)]
        e[k] = (1 - lambda_ ** 2) * e[k-1]

    return lpc_coefs, e[-1]


def spectrum_envelope_lpc(signal, sample_rate, P):
    """Calculate spectrum-envelope using LPC."""
    signal_len = len(signal)
    windowed_signal = signal * np.hamming(signal_len)
    autocorr = ac.autocorr(windowed_signal)
    lpc_coefs, e = _levinson_durbin(autocorr, P)
    w, h = scipy.signal.freqz(np.sqrt(e), lpc_coefs, fs=sample_rate)
    envelope = 20 * np.log10(np.abs(h))
    freq = np.linspace(0, sample_rate//2, len(envelope))
    return envelope, freq
