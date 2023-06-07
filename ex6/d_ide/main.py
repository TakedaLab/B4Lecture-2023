'Main.'
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as std


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(description="Perform k-means and mfcc")
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
    data_set = np.loadtxt(fname=file_path, delimiter=",")

    return data_set


def main():
    """_summary_
    """
    args = parse_args()

    file_path = args.csv_file
    dim = args.dimension
    data = open_csv(file_path)
    d_dim = data.shape[1]
    x = data

    x_std = std.zscore(x)
    print(x_std)

    cov_mat = np.cov(x_std.T)
    print(cov_mat)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    #print('\nEigenvalues \n%s' % eigen_vals)
    print(eigen_vecs)

    """plt.scatter(data[:,0], data[:,1])
    plt.show()"""

    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    plt.bar(range(1,d_dim+1), var_exp, alpha=0.5, align='center',label='individual explained variance')
    plt.step(range(1,d_dim+1), cum_var_exp, where='mid', label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    for k in range(1, dim):
        w=np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[k][1][:, np.newaxis]))

    print(w)

    x_pca = x_std.dot(w)
    print(x_pca.shape)

    plt.scatter(x_pca[ : , 0], x_pca[ : , 1], c="black")
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
