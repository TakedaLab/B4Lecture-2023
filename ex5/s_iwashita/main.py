"""Perform clustering and MFCC analysis."""
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import librosa


import k_means
import mel_spec

def run_k_means(filename):
    data = pd.read_csv(filename)
    data = np.array(data)
    dimension = len(data[0])

    N = 4
    
    if dimension == 2:
        fig, ax = plt.subplots((N + 1) // 2, 2, figsize=(12, 10))
        fig.subplots_adjust(hspace=0.3, wspace=0.2)
        for k in range(0, N):
            x, y, cluster_n, center, centroids = k_means.k_means_2d(data, k + 2)
            ax[k // 2][k % 2].set_title("k = {}".format(k + 2))
            ax[k // 2][k % 2].scatter(x, y, c=cluster_n, s=30)
            ax[k // 2][k % 2].plot(centroids.T[0], centroids.T[1], "x", c="r", label = "Initial centroid")
            ax[k // 2][k % 2].scatter(center.T[0], center.T[1], c="r", label = "Calculated centroids")
            ax[k // 2][k % 2].set_xlabel("$x_1$")
            ax[k // 2][k % 2].set_ylabel("$x_2$")
            ax[k // 2][k % 2].legend()
        plt.savefig("{}.png".format(filename[3:8]))

    elif dimension == 3:
        fig = plt.figure(figsize=(12, 10))
        fig.subplots_adjust(hspace=0.3, wspace=0.2)
        def rotate(angle):
            [axes[i].view_init(azim=angle) for i in range(N)]
        axes = []
        for k in range(0, N):
            x, y, z, cluster_n, center, centroids = k_means.k_means_3d(data, k + 2)
            ax = fig.add_subplot((N + 1)// 2, 2, k + 1, projection="3d")
            ax.scatter(x, y, z, c=cluster_n, s=10)
            ax.set_title("k = {}".format(k + 2))
            ax.plot(centroids.T[0], centroids.T[1],centroids.T[2], "x", c="r", label = "Initial centroid")
            ax.scatter(center.T[0], center.T[1], center.T[2], c="r", label = "Calculated centroids")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel("$x_3$")
            ax.legend()
            axes.append(ax)
        rot_animation = animation.FuncAnimation(fig, rotate, frames=100, interval=50)
        rot_animation.save('{}.gif'.format(filename[3:8]), writer="pillow", dpi=80)

def run_mfcc(filename):
    data, fs = librosa.load("sample.wav")

    win_length = 512
    hop_length = 256

    spectrogram = librosa.stft(data, win_length=win_length, hop_length=hop_length)
    spectrogram_db = 20 * np.log10(np.abs(spectrogram))

    fig = plt.figure(figsize=(12,10))
    
    ax0 = fig.add_subplot(411)
    img = librosa.display.specshow(
        spectrogram_db,
        y_axis="log",
        sr=fs,
        cmap="rainbow",
        ax=ax0
        )
    ax0.set_title("Spectrogram")
    ax0.set_ylabel("frequency [Hz]")
    fig.colorbar(
        img,
        aspect=10,
        pad=0.01,
        extend="both",
        ax=ax0,
        format="%+2.f dB"
        )

    # mfcc 表示
    mfcc_dim = 12
    ax1 = fig.add_subplot(412)
    mfcc = mel_spec.calc_mfcc(data, fs, win_length, hop_length, mfcc_dim)
    wav_time = data.shape[0] // fs
    extent = [0, wav_time, 0, mfcc_dim]
    img1 = ax1.imshow(
        np.flipud(mfcc),
        aspect="auto",
        extent=extent,
        cmap="rainbow"
        )
    ax1.set_title("MFCC sequence")
    ax1.set_ylabel("MFCC")
    ax1.set_yticks(range(0, 13, 2))
    fig.colorbar(
        img1,
        aspect=10,
        pad=0.01,
        extend="both",
        ax=ax1,
        format="%+2.f dB"
        )

    # Δmfcc 表示
    ax2 = fig.add_subplot(413)
    dmfcc = mel_spec.delta(mfcc)
    img2 = ax2.imshow(
        np.flipud(dmfcc),
        aspect="auto",
        extent=extent,
        cmap="rainbow"
        )
    ax2.set(
        title="ΔMFCC sequence",
        ylabel="ΔMFCC",
        yticks=range(0, 13, 2)
        )
    fig.colorbar(
        img2,
        aspect=10,
        pad=0.01,
        extend="both",
        ax=ax2,
        format="%+2.f dB"
        )

    # ΔΔmfcc 表示
    ax3 = fig.add_subplot(414)
    ddmfcc = mel_spec.delta(dmfcc)
    img3 = ax3.imshow(
        np.flipud(ddmfcc),
        aspect="auto",
        extent=extent,
        cmap="rainbow"
        )
    ax3.set(
        title="ΔΔMFCC sequence",
        xlabel="time[s]",
        ylabel="ΔΔMFCC",
        yticks=range(0, 13, 2)
        )
    fig.colorbar(img3,
                aspect=10,
                pad=0.01,
                extend="both",
                ax=ax3,
                format="%+2.f dB"
                )

    fig.tight_layout()
    fig.savefig("mfcc.png")

def main():
    parser = argparse.ArgumentParser(
        description="Perform clustering and MFCC analysis."
    )

    parser.add_argument("-m", "--mode", help="k: k-means, m: mfcc")
    parser.add_argument("-f", "--name", help="File name")

    args = parser.parse_args()

    mode = args.mode
    filename = args.name

    if mode == "k":
        run_k_means(filename)

    elif mode == "m":
        run_mfcc(filename)
    




if __name__ == "__main__":
    main()