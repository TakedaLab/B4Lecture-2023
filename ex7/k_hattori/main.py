"""Soft clustering of data."""
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    """
    Load csv files.

    Args:
        path (str): A file path.

    Returns:
        ndarray: Contents of csv file.
    """
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        buf = [row for row in reader]
    # Convert to ndarray in float64
    array = np.array(buf)
    array = array.astype(np.float64)

    return array


def Initialize(data, k):
    """
    Set initial values for the EM algorithm.

    Args:
        data (ndarray): Data to be applied.
        k (int): Number of centroids.

    Returns:
        ndarray: Mean value array (random).
        ndarray: Covariance matrix (identity matrix).
        ndarray: Mixing rate array (uniformly distributed).
    """
    Dim = data.shape[1]
    Mu = np.random.randn(k, Dim)
    Sigma = np.array([np.identity(Dim) for i in range(k)])
    Pi = np.ones(k) / k

    return Mu, Sigma, Pi


def normal_dist(data, Mu, Sigma):
    """
    Calculate probability of a normal distribution for data.

    Args:
        data (ndarray): Data to be calculated.
        Mu (ndarray): Mean value array.
        Sigma (ndarray): Covariance matrix.

    Returns:
        ndarray: Probability in the normal distribution for data.
    """
    Dim = data.shape[1]
    data_0 = data - Mu[:, None]
    tmp = data_0 @ np.linalg.inv(Sigma) @ data_0.transpose(0, 2, 1)
    tmp = tmp.diagonal(axis1=1, axis2=2)
    # calculate probability
    denom = np.sqrt((2 * np.pi) ** Dim) * np.sqrt(np.linalg.det(Sigma))
    numer = np.exp(-1 * tmp / 2)
    gauss = numer / denom[:, None]

    return gauss


def gmm(data, Mu, Sigma, Pi):
    """
    Superimpose several normal distributions.

    Args:
        data (ndarray): Data to be calculated.
        Mu (ndarray): Mean value array.
        Sigma (ndarray): Covariance matrix.
        Pi (ndarray): Mixing rate array.

    Returns:
        ndarray: Mixed normal distribution.
        ndarray: Weighted normal distribution.
    """
    weighted_gauss = normal_dist(data, Mu, Sigma) * Pi[:, None]
    mixed_gauss = np.sum(weighted_gauss, axis=0)

    return mixed_gauss, weighted_gauss


def EM_algorithm(data, Mu, Sigma, Pi, epsilon=1e-4):
    """
    Maximum likelihood estimation using EM Algorithm.

    Args:
        data (ndarray): data to be calculated.
        Mu (ndarray): Mean value array.
        Sigma (ndarray): Covariance matrix.
        Pi (ndarray): Mixing rate array.
        epsilon (float, optional): Error considered as convergence.

    Returns:
        ndarray: Mean value array.
        ndarray: Covariance matrix.
        ndarray: Mixing rate array.
        list: Transition of the log-likelihood function.
    """
    # initialize
    N = data.shape[0]
    mixed_gauss, weighted_gauss = gmm(data, Mu, Sigma, Pi)
    log_likelihood = [np.sum(np.log(mixed_gauss))]
    Diff = 100
    # repeat until convergence
    while Diff > epsilon:
        # update parameters
        gamma = weighted_gauss / mixed_gauss
        nk = np.sum(gamma, axis=1)
        Mu = np.sum(gamma[:, :, None] * data / nk[:, None, None], axis=1)
        data_0 = data - Mu[:, None]
        Sigma = (gamma[:, None] * data_0.transpose(0, 2, 1) @ data_0) / nk[
            :, None, None
        ]
        Pi = nk / N
        # recalculation of probabilities
        mixed_gauss, weighted_gauss = gmm(data, Mu, Sigma, Pi)
        # calculation of likelihood function
        log_newlikelihood = np.sum(np.log(mixed_gauss))
        log_likelihood.append(log_newlikelihood)
        Diff = log_newlikelihood - log_likelihood[-2]

    return Mu, Sigma, Pi, log_likelihood


def main():
    """
    Soft clustering of data with gmm.

    Returns:
       None
    """
    # make parser
    parser = argparse.ArgumentParser(
        prog="main.py",
        usage="Demonstration of argparser",
        description="description",
        epilog="end",
        add_help=True,
    )
    # add arguments
    parser.add_argument("-f", dest="filename", help="Filename", required=True)
    parser.add_argument("-k", dest="k", help="Number of cluster",
                        type=int, default=2)
    # parse arguments
    args = parser.parse_args()

    path = args.filename
    k = args.k

    fname_index = path.find("/")
    fname = path[fname_index + 1:]
    data = load_csv(path)
    Dim = data.shape[1]

    Mu, Sigma, Pi = Initialize(data, k)
    Mu, Sigma, Pi, log_likelihood = EM_algorithm(data, Mu, Sigma, Pi)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    # plot log-likelihood function
    plt.plot(log_likelihood)
    plt.title(f"Log-Likelihood(k={k}, {fname})")
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.tight_layout()
    plt.show()

    if Dim == 1:
        # calculate result of gmm
        x = np.linspace(np.min(data), np.max(data), 100)[:, None]
        mixed_gauss = gmm(x, Mu, Sigma, Pi)[0]
        # plot gmm
        plt.scatter(
            data,
            np.zeros(data.shape[0]),
            marker="o",
            facecolor="None",
            edgecolors="blue",
            alpha=0.3,
            label="Data",
        )
        plt.scatter(Mu, np.zeros(Mu.shape[0]), marker="x",
                    color="red", label="Centroid")
        plt.plot(x, mixed_gauss, label="GMM")
        plt.title(f"k={k}, {fname}")
        plt.xlabel("x")
        plt.ylabel("probability")
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif Dim == 2:
        # calculate result of gmm
        X = np.linspace(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1, 100)
        Y = np.linspace(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1, 100)
        XX, YY = np.meshgrid(X, Y)
        coor = np.dstack((XX, YY))
        gauss_3d = np.array([gmm(cod, Mu, Sigma, Pi)[0] for cod in coor])
        # plot gmm in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            *data.T,
            np.zeros(data.shape[0]),
            marker="o",
            facecolor="None",
            edgecolors="blue",
            alpha=0.3,
            label="Data",
        )
        ax.scatter(*Mu.T, np.zeros(Mu.shape[0]), marker="x",
                   color="red", label="Centroid")
        ax.plot_wireframe(XX, YY, gauss_3d, color="magenta", alpha=0.3)
        ax.set_title(f"k={k}, {fname}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("probability")
        ax.legend()
        plt.tight_layout()
        plt.show()
        # plot contour map
        plt.scatter(*data.T, marker="o", facecolor="None", alpha=0.3,
                    edgecolors="blue", label="Data")
        plt.scatter(*Mu.T, marker="x", color="red", label="Centroid")
        plt.contour(XX, YY, gauss_3d, cmap="jet")
        plt.title(f"Contour map (k={k}, {fname})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1)
        plt.ylim(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
