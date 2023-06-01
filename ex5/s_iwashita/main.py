"""Perform clustering and MFCC analysis."""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import k_means
import mel_spec


def main():
    parser = argparse.ArgumentParser(
        description="Perform clustering and MFCC analysis."
    )

    parser.add_argument("name", help="File name")

    args = parser.parse_args()

    filename = args.name

    data = k_means.csv_open(filename)
    dimension = len(data[0])

    N = 4
    
    if dimension == 2:
        fig, ax = plt.subplots((N + 1) // 2, 2)
        fig.subplots_adjust(hspace=0.5, wspace=0.4)
        for k in range(0, N):
            x, y, cluster_n = k_means.k_means_2d(data, k + 2)
            ax[k // 2][k % 2].set_title("k = {}".format(k + 2))
            ax[k // 2][k % 2].scatter(x, y, c=cluster_n, s=10)
            ax[k // 2][k % 2].set_xlabel("$x_1$")
            ax[k // 2][k % 2].set_ylabel("$x_2$")
        plt.savefig("{}.png".format(filename[3:8]))
    elif dimension == 3:
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.5, wspace=0.4)
        def rotate(angle):
            [axes[i].view_init(azim=angle) for i in range(N)]
        axes = []
        for k in range(0, N):
            x, y, z, cluster_n = k_means.k_means_3d(data, k + 2)
            ax = fig.add_subplot((N + 1)// 2, 2, k + 1, projection="3d")
            ax.scatter(x, y, z, c=cluster_n, s=10)
            ax.set_title("k = {}".format(k + 2))
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel("$x_3$")
            axes.append(ax)
        rot_animation = animation.FuncAnimation(fig, rotate, frames=100, interval=50)
        rot_animation.save('{}.gif'.format(filename[3:8]), writer="pillow", dpi=80)
    




if __name__ == "__main__":
    main()