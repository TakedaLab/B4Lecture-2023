"""To adapt GMM."""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

import csvOpe
import GMM

# iteration times
ITER = 10000
E = 0.0001
LOOP = 100


def main():
    # create parser and argument
    parser = argparse.ArgumentParser(description="This is a program to adapt GMM")
    parser.add_argument("path", help="the path to the dataset.")
    parser.add_argument(
        "-g", "--group", default=2, type=int, help="The number of clusters"
    )

    # read out parser
    args = parser.parse_args()

    # get file name
    f_name = csvOpe.get_fname(args.path)

    # read out csv
    data = csvOpe.read_csv(args.path)

    # initialize variable stage
    weight, mean, var = GMM.init(data, args.group)
    # print(weight)
    # print(mean.shape)
    # print(var.shape)

    # get likelihood information
    L = GMM.log_likelihood(data, weight, mean, var)
    L_g = np.array(L)
    # loop stage
    for i in range(ITER):
        gamma = GMM.update_gamma(data, weight, mean, var)
        weight, mean, var = GMM.new_para(data, gamma)
        L_n = GMM.log_likelihood(data, weight, mean, var)
        # print(L_n)
        if L_n - L <= E:
            print(i)
            break
        L_g = np.hstack((L_g, L_n))
        L = L_n

    """
    -------------test runtime-----------------------------
    """
    """
    time_s = time.time()
    for i in range(LOOP):
        kmeans = KMeans(n_clusters=args.group, max_iter=30, init="random", n_init='auto')
        kmeans.fit(data)
        means1 = kmeans.cluster_centers_
    kmeans_t = (time.time() - time_s) / LOOP

    time_s = time.time()
    i_sum = 0
    for j in range(LOOP):
        weight, mean, var = GMM.init(data, args.group)
        L = GMM.log_likelihood(data, weight, mean, var)
        L_g = np.array(L)
        # loop stage
        for i in range(ITER):
            gamma = GMM.update_gamma(data, weight, mean, var)
            weight, mean, var = GMM.new_para(data, gamma)
            L_n = GMM.log_likelihood(data, weight, mean, var)
            # print(L_n)
            if L_n - L <= E:
                i_sum += i
                break
            L_g = np.hstack((L_g, L_n))
            L = L_n
    random_t = (time.time() - time_s )/ LOOP


    time_s = time.time()
    i_sum1 = 0
    for j in range(LOOP):
        weight, mean, var = GMM.init(data, args.group)
        mean = means1
        L = GMM.log_likelihood(data, weight, mean, var)
        L_g = np.array(L)
        # loop stage
        for i in range(ITER):
            gamma = GMM.update_gamma(data, weight, mean, var)
            weight, mean, var = GMM.new_para(data, gamma)
            L_n = GMM.log_likelihood(data, weight, mean, var)
            # print(L_n)
            if L_n - L <= E:
                i_sum1 += i
                break
            L_g = np.hstack((L_g, L_n))
            L = L_n
    centroid_t = (time.time() - time_s )/ LOOP

    i_sum = i_sum /LOOP
    i_sum1 = i_sum1 /LOOP
    print((str(LOOP) + "  times average").center(40,'-'))
    print("GMM time with random centroid: "  +  "{:.3f}".format(random_t))
    print("kmeans to generate centroid: " + "{:.3f}".format(kmeans_t))
    print("GMM time with kmeans centroid: " + "{:.3f}".format(centroid_t))
    print(("comparison").center(40,'-'))
    print("random times: " + "{:.3f}".format(random_t) +",loop: "+ str(i_sum))
    print("kmeans times: " + "{:.3f}".format(kmeans_t+centroid_t)+",loop: "+ str(i_sum1))
    """
    """
    --------------plot stage -------------------------------
    """
    fig = plt.figure(figsize=(18, 9))
    # 1-dim
    if data.shape[1] == 1:
        # label
        A = fig.add_subplot(111)
        A.set_title("gauss function", fontsize=16)
        A.set_xlabel("x", fontsize=14)
        A.set_ylabel("probability", fontsize=14)
        A.set_title(f_name + " probability")

        # range
        x_range = np.array([np.min(data), np.max(data)])
        x_range = np.linspace(x_range[0], x_range[1], 10000).reshape(-1, 1)
        Y = np.sum(weight[:, np.newaxis] * GMM.gauss(x_range, mean, var), axis=0)

        # plot
        A.plot(data, np.zeros_like(data), ".", color="b", label="data")
        A.plot(
            mean,
            np.zeros_like(mean),
            "x",
            color="r",
            label="centroid",
            markersize=10,
            markeredgewidth=3,
        )
        A.plot(x_range, Y, label="GMM")
        A.legend()
        plt.savefig(f_name + "_probability_" + str(args.group))

    # 2-dim
    elif data.shape[1] == 2:
        # plot range
        x_0_line = np.linspace(np.min(data.T[0]), np.max(data.T[0]), num=100)
        x_1_line = np.linspace(np.min(data.T[1]), np.max(data.T[1]), num=100)
        x_0_grid, x_1_grid = np.meshgrid(x_0_line, x_1_line)
        dim = x_0_grid.shape
        x_point_arr = np.stack([x_0_grid.flatten(), x_1_grid.flatten()], axis=1)

        Y = np.sum(weight[:, np.newaxis] * GMM.gauss(x_point_arr, mean, var), axis=0)

        A = fig.add_subplot(121)
        A.plot(data.T[0], data.T[1], ".", color="b", label="data")
        A.plot(
            mean.T[0],
            mean.T[1],
            "x",
            color="r",
            markersize=8,
            markeredgewidth=3,
            label="centroid",
        )
        contour = A.contour(
            x_point_arr.T[0].reshape(dim), x_point_arr.T[1].reshape(dim), Y.reshape(dim)
        )  # 尤度
        A.clabel(contour, fmt="%.2f")
        A.set_xlabel("$X_{1}$")
        A.set_ylabel("$X_{2}$")
        A.set_title(f_name + "2D-probability")
        plt.legend()

        B = fig.add_subplot(122, projection="3d")
        B.plot(data.T[0], data.T[1], ".", color="b", label="data")
        B.plot(
            mean.T[0],
            mean.T[1],
            "x",
            color="r",
            markersize=8,
            markeredgewidth=3,
            label="centroid",
        )
        B.plot_surface(
            x_point_arr.T[0].reshape(dim),
            x_point_arr.T[1].reshape(dim),
            Y.reshape(dim),
            alpha=0.7,
            cmap="plasma",
        )
        B.set_xlabel("$X_{1}$")
        B.set_ylabel("$X_{2}$")
        B.set_zlabel("probability")
        B.set_title(f_name + "3D-probability")
        plt.legend()
        plt.savefig(f_name + "_probability_" + str(args.group))

    # plot likelihood

    plt.figure()
    L_g = L_g[np.where(L_g != 0)]
    plt.plot(range(len(L_g)), L_g)
    plt.xlabel("iteration")
    plt.ylabel("log likelihood")
    plt.title(f_name + " likelihood")
    plt.savefig(f_name + "_likelihood_" + str(args.group))
    plt.show()


if __name__ == "__main__":
    main()
