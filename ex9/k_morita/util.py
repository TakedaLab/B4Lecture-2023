
import os
import pandas as pd
import numpy as np

import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


def wavfile_to_spectrogram(
    audio_path,
    save_path,
    spectrogram_dimensions=(64, 64),
    noverlap=16,
    cmap="gray_r"
):
    if os.path.exists(save_path):
        return

    smaple_rate, samples = wav.read(audio_path)
    print(smaple_rate)
    fig = plt.figure()
    fig.set_size_inches((
        spectrogram_dimensions[0]/fig.get_dpi(), 
        spectrogram_dimensions[1]/fig.get_dpi()
        ))
    ax = plt.Axes(fig, [0., 0., 1., 1.,])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.specgram(samples, cmap=cmap, Fs=2, noverlap=noverlap)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    os.sleep(0.1)
    plt.clf()
    plt.close()


def files_to_spectrogram(
    files,
    save_dir,
    spectrogram_dimensions=(64, 64),
    noverlap=12,
    cmap="gray_r"
):
    for file_name in files:
        audio_path = os.path.join(os.path.pardir, file_name)
        spectrogram_path = os.path.join(save_dir, os.path.basename(file_name).replace(".wav", ".png"))
        wavfile_to_spectrogram(
            audio_path,
            spectrogram_path,
            spectrogram_dimensions=spectrogram_dimensions,
            noverlap=noverlap,
            cmap=cmap
        )


if __name__ == "__main__":
    train_csv = pd.read_csv("../training.csv")
    test_csv = pd.read_csv("../test_truth.csv") 
    files_to_spectrogram(train_csv["path"].values, save_dir="spectrograms/train")
    files_to_spectrogram(test_csv["path"].values, save_dir="spectrograms/test")