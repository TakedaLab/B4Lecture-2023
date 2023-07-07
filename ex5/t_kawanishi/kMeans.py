"""k-means function."""
import random

import numpy as np


def init_r(data: np.ndarray, g_num: int) -> np.ndarray:
    """Random generate first centroid.

    Args:
        data (np.ndarray): dataset
        g_num (int): the number of the group

    Returns:
        np.ndarray: initial centroid shape is (g_num, dim)
    """
    # dimension of the data
    dim = len(data)

    # generate initial centroid
    if dim == 2:
        centroid = np.concatenate(
            [
                np.random.uniform(np.min(data[0]), np.max(data[0]), (g_num, 1)),
                np.random.uniform(np.min(data[1]), np.max(data[1]), (g_num, 1)),
            ],
            axis=1,
        )
    elif dim == 3:
        centroid = np.concatenate(
            [
                np.random.uniform(np.min(data[0]), np.max(data[0]), (g_num, 1)),
                np.random.uniform(np.min(data[1]), np.max(data[1]), (g_num, 1)),
                np.random.uniform(np.min(data[2]), np.max(data[2]), (g_num, 1)),
            ],
            axis=1,
        )
    return centroid


def init_rp(data: np.ndarray, g_num: int) -> np.ndarray:
    """Random generate first centroid with k-means++.

    Args:
        data (np.ndarray): dataset
        g_num (int): the number of the group

    Returns:
        np.ndarray: initial centroid shape is (g_num, dim)
    """
    # dimension of the data
    dim = len(data)

    # first centroid
    First_c = random.randint(0, len(data[0]))

    # generate initial centroid
    if dim == 2:
        centroid = np.array([data[0][First_c], data[1][First_c]]).reshape(1, 2)
        for i in range(g_num - 1):
            l, dis = group_c(data, centroid)
            p = dis / np.sum(dis)
            np.random.choice(np.arange(data.shape[1]), p=p)
            index = np.random.choice(np.arange(data.shape[1]), p=p)
            centroid = np.concatenate(
                (centroid, np.array([data[0][index], data[1][index]]).reshape(1, 2))
            )

    elif dim == 3:
        centroid = np.array(
            [data[0][First_c], data[1][First_c], data[2][First_c]]
        ).reshape(1, 3)
        for i in range(g_num - 1):
            l, dis = group_c(data, centroid)
            p = dis / np.sum(dis)
            index = np.random.choice(np.arange(data.shape[1]), p=p)
            centroid = np.concatenate(
                (
                    centroid,
                    np.array([data[0][index], data[1][index], data[2][index]]).reshape(
                        1, 3
                    ),
                )
            )

    return centroid


def group_c(data: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Compute the nearest centroid.

    Args:
        data (np.ndarray): dataset
        centroid (np.ndarray): centroid

    Returns:
        np.ndarray: clustered outcome and distance to the closet centroid
    """
    # centroid shape is (g_num, dim)

    # create label to clustering
    label = np.zeros(len(data[0]))
    disToCen = np.zeros(len(data[0]))
    first_time = True

    # compute distance
    for i in range(len(centroid)):
        dis = data.T - centroid[i]
        dis = np.sum(dis * dis, axis=1)

        # select the closet centroid
        if first_time:
            disToCen = dis
            first_time = False
        else:
            disToCen = np.minimum(disToCen, dis)
            label[disToCen == dis] = i

    return label, disToCen


def gen_cen(data: np.ndarray, label: np.ndarray, g_num: int) -> np.ndarray:
    """Update centroid.

    Args:
        data (np.ndarray): dataset
        label (np.ndarray): cluster label
        g_num (int): the number of group

    Returns:
        np.ndarray: updated centroid
    """
    # new centroid matrix
    dim = len(data)
    centroid = np.zeros((g_num, dim))
    for i in range(g_num):
        # number of members in specific group
        num = np.sum(label == i)

        # zeros all members not in specific group
        Gdata = np.array(data)
        Gdata[:, label != i] = 0

        # sum all axis
        centroid[i] = np.sum(Gdata, axis=1) / num

    return centroid


def k_means(data: np.ndarray, g_num: int, iter=10000, plus_algo=False) -> np.ndarray:
    """To clustering data with k_means.

    Args:
        data (np.ndarray): dataset
        g_num (int): the number of group
        iter (int, optional): iteration. Defaults to 10000.

    Returns:
        np.ndarray: label and centroid
    """
    if plus_algo:
        centroid = init_rp(data, g_num)
    else:
        centroid = init_r(data, g_num)
    label_b = np.zeros(len(data[0]))

    # loop to search optimum centroid
    for i in range(iter):
        # clustering
        label, dis = group_c(data, centroid)

        # compute new centroid
        centroid = gen_cen(data, label, g_num)

        # decide whether break
        if np.all(label_b == label):
            break
        label_b = label
    return label, centroid


if __name__ == "__main__":
    pass
