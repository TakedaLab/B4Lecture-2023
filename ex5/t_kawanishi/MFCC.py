"""Store the functions needed for MFCC"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack.realtransforms import dct

import ex1Function as F


def hz2mel(f: np.ndarray, f0=700.0) -> np.ndarray:
    """Hz to mel scale

    Args:
        f (np.ndarray): Hz
        f0 (int, optional): f0. Defaults to 700.

    Returns:
        np.ndarray: mal scale
    """
    m0 = cal_m0(f0)
    return m0 * np.log(f / f0 + 1.0)


def mel2hz(m: np.ndarray, f0=700.0) -> np.ndarray:
    """mel scale to Hz

    Args:
        m (np.ndarray): mel scale
        f0 (int, optional): f0. Defaults to 700.

    Returns:
        np.ndarray: Hz
    """
    m0 = cal_m0(f0)
    return f0 * (np.exp(m / m0) - 1.0)


def cal_m0(f0: int) -> int:
    """calculate m0

    Args:
        f0 (int): f0

    Returns:
        int: m0
    """
    return 1000.0 / np.log(1000.0 / f0 + 1.0)


def melFilterBank(fs: int, N: int, channels: int) -> np.ndarray:
    """_summary_

    Args:
        fs (int): sample rate
        N (int): samples number of fft
        channels (int): number of channels

    Returns:
        np.ndarray: mel filter bank
    """
    # nyquist frequency(mel)
    f_ny = fs / 2
    # nyquist frequency(mel)
    melmax = hz2mel(f_ny)
    # frequency index
    nmax = N // 2
    # size
    df = fs / N
    # calculate center fs at both scale
    dmel = melmax / (channels + 1)
    mel_centers = np.arange(1, channels + 1) * dmel
    f_centers = mel2hz(mel_centers)
    # calculate center fs index
    i_center = np.round(f_centers / df)
    # calculate start fs index
    i_start = np.hstack(([0], i_center[0: channels - 1]))
    # calculate end fs index
    i_end = np.hstack((i_center[1:channels], [nmax]))

    # start creating mel filter bank
    filterbank = np.zeros([channels, nmax])
    for i in range(0, channels):
        # slope of the left part
        slope = 1.0 / (i_center[i] - i_start[i])
        for j in range(int(i_start[i]), int(i_center[i])):
            filterbank[i, j] = (j - i_start[i]) * slope
        # slope of the right part
        slope = 1.0 / (i_end[i] - i_center[i])
        for j in range(int(i_center[i]), int(i_end[i])):
            filterbank[i, j] = (i_end[i] - j) * slope

    return filterbank, f_centers


def calc_mfcc(
    data: np.ndarray, filterbank: np.ndarray, hop: float, shift_s: int
) -> np.ndarray:
    """To calculate mfcc

    Args:
        data (np.ndarray): data
        filterbank (np.ndarray): filter bank
        shift_s (int): shift size
        dim (int): dimension

    Returns:
        np.ndarray: mfcc
    """
    spec = F.stft(data, hop, shift_s)

    mel_spec = np.dot(filterbank, np.abs(spec[:-1]))

    mfcc = np.zeros_like(mel_spec)
    for i in range(mel_spec.shape[1]):
        mfcc[:, i] = dct(mel_spec[:, i], type=2, norm="ortho", axis=-1)

    return mel_spec, mfcc


def delta_mfcc(mfcc: np.ndarray, k=2) -> np.ndarray:
    """calculate delta from mfcc

    Args:
        mfcc (np.ndarray): Mel frequency cepstrum coefficient
        k (int, optional): window of regression. Defaults to 2.

    Returns:
        np.ndarray: Delta of MFCC
    """
    mfcc_pad = np.pad(mfcc, [(k, k + 1), (0, 0)], "edge")
    m = np.arange(-k, k + 1)
    k_sq = np.sum(m**2)
    d_mfcc = np.zeros_like(mfcc)
    for i in range(mfcc.shape[0]):
        d_mfcc[i] = np.dot(m, mfcc_pad[i: i + k * 2 + 1])
    return d_mfcc / k_sq


def plot_filter(filterbank: np.ndarray, fs: int):
    """To plot MFCC

    Args:
        filterbank (np.ndarray): mel filter bank
        fs (int): sample rate
    """
    N = len(filterbank[0]) * 2
    for c in np.arange(0, len(filterbank)):
        plt.plot(np.arange(0, N / 2) * fs / N, filterbank[c])

    plt.title("Mel filter bank")
    plt.xlabel("Frequency[Hz]")


if __name__ == "__main__":
    filterbank, f_centers = melFilterBank(48000, 2048, 20)
    plot_filter(filterbank, 48000)
    plt.show()
    print(cal_m0(700))
