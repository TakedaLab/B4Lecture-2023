"""K-means module."""
import matplotlib.pyplot as plt
import numpy as np
import sys


def loadcsv(filename):
    """Load data file."""
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    return data


def generate_rondom_2d(k, data):
    """Generate random 2d vectors."""
    rng = np.random.default_rng()
    x, y = data.T
    rand_x = rng.uniform(min(x), max(x), k)
    rand_y = rng.uniform(min(y), max(y), k)
    random_coords = np.array([rand_x, rand_y]).T
    return random_coords


def generate_random_3d(k, data):
    """Generate random 3d vectors."""
    rng = np.random.default_rng()
    x, y, z = data.T
    rand_x = rng.uniform(min(x), max(x), k)
    rand_y = rng.uniform(min(y), max(y), k)
    rand_z = rng.uniform(min(z), max(z), k)
    random_coords = np.array([rand_x, rand_y, rand_z]).T
    return random_coords


def update_cluster(cluster, centroids, data):
    """Update cluster function."""
    new_cluster = np.zeros_like(cluster)
    for i in range(len(data)):
        d = data[i]
        distances = np.array([np.linalg.norm(d - cent, ord=2) for cent in centroids])
        new_cluster[i] = distances.argmin()
    return cluster, new_cluster


def update_centroids(centroids, cluster, data):
    """Update centroids function."""
    new_centroids = np.zeros_like(centroids)
    for i in range(len(centroids)):
        target_data = data[cluster == i]
        new_centroids[i] = np.array(
            [np.mean(target_data[:, i]) for i in range(target_data.shape[1])]
        )
    return new_centroids


def k_means_2d(k, data):
    """K-means 2-dimension."""
    centroids = generate_rondom_2d(k, data)
    initial_centroids = centroids
    prev_cluster = np.zeros(len(data))
    cluster = np.zeros(len(data))
    while True:
        prev_cluster, cluster = update_cluster(cluster, centroids, data)
        if np.array_equal(prev_cluster, cluster):
            break
        else:
            centroids = update_centroids(centroids, cluster, data)

    clustered_data = np.concatenate([data, cluster.reshape(-1, 1)], axis=1)
    return clustered_data, centroids, initial_centroids


def k_means_3d(k, data):
    """K-means 3-dimension."""
    centroids = generate_random_3d(k, data)
    initial_centroids = centroids
    prev_cluster = np.zeros(len(data))
    cluster = np.zeros(len(data))
    while True:
        prev_cluster, cluster = update_cluster(cluster, centroids, data)

        if np.array_equal(prev_cluster, cluster):
            break
        else:
            centroids = update_centroids(centroids, cluster, data)

    clustered_data = np.concatenate([data, cluster.reshape(-1, 1)], axis=1)
    return clustered_data, centroids, initial_centroids


def main2d():
    """K-means(2d) main function."""
    file = sys.argv[1]
    k = int(sys.argv[2])
    data = loadcsv(file)
    clustered_data, centroids, initial_centroids = k_means_2d(k, data)

    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(k):
        d = clustered_data[clustered_data[:, 2] == i]
        x, y = d[:, 0], d[:, 1]
        ax.scatter(x, y)
    ax.scatter(
        initial_centroids[:, 0],
        initial_centroids[:, 1],
        marker="x",
        c="purple",
        label="Initial Centroids",
    )
    ax.scatter(
        centroids[:, 0], centroids[:, 1], marker="x", c="black", label="Final Centroids"
    )
    ax.set_title("k-means clustering")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()


def main3d():
    """K-means(3d) main function."""
    file = "../data3.csv"
    k = int(sys.argv[1])
    data = loadcsv(file)
    clustered_data, centroids, initial_centroids = k_means_3d(k, data)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in range(k):
        d = clustered_data[clustered_data[:, 3] == i]
        x, y, z = d[:, 0], d[:, 1], d[:, 2]
        ax.scatter(x, y, z, marker="o")
    ax.scatter(
        initial_centroids[:, 0],
        initial_centroids[:, 1],
        initial_centroids[:, 2],
        marker="x",
        c="purple",
        label="Initial Centroids",
    )
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        marker="x",
        c="black",
        label="Final Centroids",
    )
    ax.set_title("k-means clustering")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # main2d()
    main3d()
