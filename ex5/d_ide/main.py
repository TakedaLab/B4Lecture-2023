'Main.'
import argparse
import matplotlib.pyplot as plt
import numpy as np
import k_means as km

def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(description="Perform regression analysis.")
    parser.add_argument(
        "--csv-file",
        type=str,
        required=True,
        help="data csv file",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=2,
        help="number of cluster",
    )

    return parser.parse_args()


def open_csv(file_path):
    """Read csv file.

    Args:
        file_path (str): Csv file to read

    Returns:
        ndarray: Data read
    """
    data_set = np.loadtxt(fname=file_path, delimiter=",", skiprows=1)

    return data_set

def scat_plot2d(data, k):
    """Plot two-dimensional data.

    Args:
        x (ndarray): data of x axis
        y (ndarray): data of y axis
        beta_r (ndarray): Regularized regression coefficient
        beta (ndarray): regression coefficient
    """
    labels, _ = km.k_means2d(data, k, 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], c=labels)
    ax.set_title("K-Means 2d")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    plt.savefig('result/Kmeans.png')
    plt.show()


def scat_plot3d(data, k):
    """Plot three-dimensional data.

    Args:
        x (ndarray): data of x axis
        y (ndarray): data of y axis
        z (ndarray): data of z axis
        beta_r (ndarray): Regularized regression coefficient
        beta (ndarray): regression coefficient
        N1 (int) : dimension of x
        N2 (int) : dimension of y
    """
    labels, _ = km.k_means3d(data, k, 100)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title("K-Means 3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)
    plt.savefig('result/Kmeans.png')
    plt.show()


def main():
    """Regression analysis using the least squares method."""
    args = parse_args()

    file_path = args.csv_file
    k = args.cluster

    data = open_csv(file_path)

    if data.shape[1] == 2:
        scat_plot2d(data, k)

    elif data.shape[1] == 3:
        scat_plot3d(data, k)


if __name__ == "__main__":
    main()