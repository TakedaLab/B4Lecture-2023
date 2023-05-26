"""Mel-Frequency Cepstrum Coefficients."""
import argparse
import re

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import ex1Function as F
import MFCC

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is a program to adapt MFCC."
        )
    parser.add_argument("path", help="sound file path")
    parser.add_argument("-f", "--f_size", default=512, help="frame size")

    hop = 0.5
    N = 512
    channels = 20
    hop_length = int(N * hop)
    # get args
    args = parser.parse_args()

    # create file name
    f_name = re.sub(r".+\\", "", args.path)
    f_name = re.sub(r"\..+", "", f_name)

    # read out sound
    s_data, sr = sf.read(args.path)

    # make mel filter bank
    filterbank, fcenters = MFCC.melFilterBank(sr, N, channels)
    MFCC.plot_filter(filterbank, sr)
    plt.savefig("filterbank")

    # spectrogram
    fig, ax = plt.subplots(nrows=1, ncols=1)
    amp = F.stft(s_data, hop, N)
    db = librosa.amplitude_to_db(np.abs(amp))
    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        ax=ax,
        cmap="rainbow",
    )
    ax.set(title="Spectrogram", xlabel=None, ylabel="Frequency [Hz]")
    fig.colorbar(img, aspect=10, pad=0.01, ax=ax, format="%+2.f dB")
    plt.savefig(f_name + "_spectrogram")

    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10, 6))
    plt.subplots_adjust(hspace=0.6)

    # calculate mel spectrogram and mfcc
    mel_spec, mfcc = MFCC.calc_mfcc(s_data, filterbank, hop, N)

    # mel spectrogram
    s_time = s_data.shape[0] // sr
    f_nyq = sr // 2
    extent = [0, s_time, 0, f_nyq]

    img = ax[0].imshow(
        librosa.amplitude_to_db(mel_spec),
        aspect="auto",
        extent=extent,
        cmap="rainbow",
    )
    ax[0].set(
        title="Mel spectrogram",
        xlabel=None,
        ylabel="Mel frequency [mel]",
        ylim=[0, 8000],
        yticks=range(0, 10000, 2000),
    )
    fig.colorbar(img, aspect=10, pad=0.01, ax=ax[0], format="%+2.f dB")

    # mfcc
    n_mfcc = 12
    extent = [0, s_time, 0, n_mfcc]
    img = ax[1].imshow(
        np.flipud(mfcc[:n_mfcc]), aspect="auto", extent=extent, cmap="rainbow"
    )
    ax[1].set(
        title="MFCC sequence",
        xlabel=None,
        ylabel="MFCC",
        yticks=range(0, 13, 4),
    )
    fig.colorbar(img, aspect=10, pad=0.01, ax=ax[1], format="%+2.f dB")

    # d-mfcc
    d_mfcc = MFCC.delta_mfcc(mfcc, k=2)

    img = ax[2].imshow(
        np.flipud(d_mfcc[:n_mfcc]), aspect="auto",
        extent=extent, cmap="rainbow"
    )
    ax[2].set(
        title="ΔMFCC sequence",
        xlabel=None,
        ylabel="ΔMFCC",
        yticks=range(0, 13, 4),
    )
    fig.colorbar(img, aspect=10, pad=0.01, ax=ax[2], format="%+2.f dB")

    # dd-mfcc
    dd_mfcc = MFCC.delta_mfcc(d_mfcc, k=2)
    img = ax[3].imshow(
        np.flipud(dd_mfcc[:n_mfcc]), aspect="auto",
        extent=extent, cmap="rainbow"
    )
    ax[3].set(
        title="ΔΔMFCC sequence",
        xlabel="Time [s]",
        ylabel="ΔΔMFCC",
        yticks=range(0, 13, 4),
    )
    fig.colorbar(img, aspect=10, pad=0.01, ax=ax[3], format="%+2.f dB")
    plt.savefig(f_name + "_result")
    plt.show()
