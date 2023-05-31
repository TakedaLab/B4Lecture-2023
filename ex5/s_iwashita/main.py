"""Perform clustering and MFCC analysis."""
import argparse

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Perform clustering and MFCC analysis."
    )

    parser.add_argument("name", help="File name")

    args = parser.parse_args()

    filename = args.name

    data = np.genfromtxt(filename, delimiter=",", skip_header=1)
    dimension = len(data[0])

    x = data[:, 0]
    y = data[:, 1]
    if dimension == 3:
        z = data[:, 2]
        plt.subplot(projection="3d")
        plt.scatter(x, y, z)
    else:
        plt.scatter(x, y)
    plt.show()



if __name__ == "__main__":
    main()