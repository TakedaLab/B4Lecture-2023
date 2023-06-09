"Main."
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as std


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(description="Perform pca")
    parser.add_argument(
        "--csv_file",
        type=str,
        default="../data1.csv",
        help="data csv file",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        help="compress dimension",
    )

    return parser.parse_args()


def open_csv(file_path):
    """Read csv file.

    Args:
        file_path (str): Csv file to read

    Returns:
        ndarray: Data read
    """
    data_set = np.loadtxt(fname=file_path, delimiter=",")

    return data_set


def std_cal(data):
    """Calculate std.

    Args:
        data (ndarray): target data

    Returns:
        ndarray: Standardized data
    """
    return std.zscore(data)


def covariance_cal(data):
    """Calculate covariance.

    Args:
        data (ndarray): Standardized data

    Returns:
        ndarray: covariance matrix
    """
    return np.cov(data.T)


def eig_cal(data):
    """Calculate eigen values and eigen vectors.

    Args:
        data (ndarray): covariance matrix

    Returns:
        ndarray: eigen values and eigen vectors
    """
    eigen_values, eigen_vecs = np.linalg.eig(data)
    return eigen_values, eigen_vecs


def contribution_rate(eigen_vals, d_dim):
    """Plot contribution rate.

    Args:
        eigen_vals (ndarray): eigen values
        d_dim (int): data dimension

    Returns:
        ndarray: contribution rate
    """
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.bar(
        range(1, d_dim + 1),
        var_exp,
        alpha=0.5,
        align="center",
        label="individual explained variance",
    )
    ax.step(
        range(1, d_dim + 1),
        cum_var_exp,
        where="mid",
        label="cumulative explained variance",
    )
    ax.axhline(y=0.9, c="red", label="line of 90%")
    ax.set_title("Observed data")
    ax.set_xlabel("Explained variance ratio")
    ax.set_ylabel("Principal component index")
    ax.legend()
    plt.savefig("result/contribution_rate.png")
    plt.show()
    plt.close()

    return var_exp


def eig_sort(eigen_vals, eigen_vecs):
    """Sort eigen values.

    Args:
        eigen_vals (ndarray): eigen values
        eigen_vecs (ndarray): eigen vectors

    Returns:
        ndarray: pair of eigen values and eigen vectors
    """
    eigen_pairs = [
        (np.abs(eigen_vals[i]), eigen_vecs[:, i])
        for i in range(len(eigen_vals))
    ]
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)
    return eigen_pairs


def proj_cal(dim, eig_pairs):
    """Calculate projection matrix.

    Args:
        dim (int): dimension
        eigen_pairs (ndarray): eigen pairs

    Returns:
        ndarray: projection matrix
    """
    for k in range(1, dim):
        w = np.hstack(
            (eig_pairs[0][1][:, np.newaxis], eig_pairs[k][1][:, np.newaxis])
        )
    return w


def pca(data_std, w):
    """Calculate pca.

    Args:
        data_std (ndarray): Standardized data
        w (ndarray): projection matrix
    """
    data_pca = data_std.dot(w)
    print(data_pca.shape)

    plt.scatter(data_pca[:, 0], data_pca[:, 1], c="black", label="transformed data")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title('transformed data')
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig("result/pca_compress.png")
    plt.show()


def plot2d(data, eigen_vecs, var_exp):
    """Plot 2 dimension data.

    Args:
        data (ndarray): target data
        eigen_vecs (ndarray): eigen vectors
    """

    slope = eigen_vecs[1] / eigen_vecs[0]
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 1]), 100)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], label="original data")
    ax.plot(
        x,
        slope[0] * x,
        label=f"primary component(rate={var_exp[0]:.3f})"
    )
    ax.plot(
        x,
        slope[1] * x,
        label=f"second principal component(rate={var_exp[0]:.3f})"
    )
    ax.set_title("Observed data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.savefig("result/pca_data1.png")
    plt.show()
    plt.close()


def plot3d(data, eigen_vecs, var_exp):
    """Plot 3 dimension data

    Args:
        data (ndarray): target data
        eigen_vecs (ndarray): eigen vectors
    """

    slope1 = eigen_vecs[1] / eigen_vecs[0]
    slope2 = eigen_vecs[2] / eigen_vecs[0]
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], label="original data")
    ax.plot(
        x,
        slope1[0] * x,
        slope2[0] * x,
        label=f"primary component(rate={var_exp[0]:.3f})",
    )
    ax.plot(
        x,
        slope1[1] * x,
        slope2[1] * x,
        label=f"second principal component(rate={var_exp[1]:.3f})",
    )
    ax.plot(
        x,
        slope1[2] * x,
        slope2[2] * x,
        label=f"third principal component(rate={var_exp[2]:.3f})",
    )
    ax.set_title("Observed data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.savefig("result/pca_data2.png")
    plt.show()
    plt.close()


def main():
    """Calculate PCA."""
    args = parse_args()

    file_path = args.csv_file
    dim = args.dimension
    data = open_csv(file_path)
    d_dim = data.shape[1]

    data_std = std_cal(data)

    cov_mat = covariance_cal(data_std)
    eigen_vals, eigen_vecs = eig_cal(cov_mat)

    if d_dim == 2:
        var_exp = contribution_rate(eigen_vals, d_dim)
        plot2d(data, eigen_vecs, var_exp)
    elif d_dim == 3:
        var_exp = contribution_rate(eigen_vals, d_dim)
        plot3d(data, eigen_vecs, var_exp)
        eigen_pairs = eig_sort(eigen_vals, eigen_vecs)
        w = proj_cal(dim, eigen_pairs)
        pca(data_std, w)
    else:
        contribution_rate(eigen_vals, d_dim)
        eigen_pairs = eig_sort(eigen_vals, eigen_vecs)
        w = proj_cal(dim, eigen_pairs)
        pca(data_std, w)


if __name__ == "__main__":
    main()
