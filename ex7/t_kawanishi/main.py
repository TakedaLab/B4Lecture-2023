"""To adapt GMM."""
import argparse

import matplotlib.pyplot as plt
import numpy as np

import csvOpe
import GMM

# iteration times
ITER=10000
E = 0.0001

def main():
    # create parser and argument
    parser = argparse.ArgumentParser(
        description="This is a program to adapt GMM"
        )
    parser.add_argument("path",help="the path to the dataset.")
    parser.add_argument("-g","--group",default=2,type=int,help="The number of clusters")

    # read out parser
    args = parser.parse_args()

    # get file name
    f_name = csvOpe.get_fname(args.path)

    # read out csv
    data = csvOpe.read_csv(args.path)

    # initialize variable stage
    weight, mean, var = GMM.init(data,args.group)
    L = GMM.log_likelihood(data,weight,mean,var)
    #print(weight)
    #print(mean.shape)
    #print(var.shape)
    #print(L)

    # get likelihood information
    L_g = np.array(L)
    # loop stage
    for i in range(ITER):
        gamma = GMM.update_gamma(data,weight,mean,var)
        weight, mean, var = GMM.new_para(data,gamma)
        L_n = GMM.log_likelihood(data,weight,mean,var)
        #print(L_n)
        if L_n - L <= E:
            print(i)
            break
        L_g = np.hstack((L_g,L_n))
        L = L_n

    """
    --------------plot stage -------------------------------
    """
    fig = plt.figure(figsize=(18,9))
    # 1-dim
    if data.shape[1] == 1:

        # label
        A = fig.add_subplot(111)
        A.set_title("gauss function", fontsize = 16)
        A.set_xlabel("x", fontsize = 14)
        A.set_ylabel("probability", fontsize = 14)
        A.set_title(f_name + " probability")

        # range
        x_range = np.array([np.min(data),np.max(data)])
        x_range = np.linspace(x_range[0],x_range[1],10000).reshape(-1,1)
        Y = np.sum(
            weight[:,np.newaxis] * GMM.gauss(x_range,mean,var),
            axis = 0
        )

        # plot
        A.plot(data,np.zeros_like(data),".", color="b", label="data")
        A.plot(mean,np.zeros_like(mean),"x",color="r", label="centroid",markersize=10,markeredgewidth=3)
        A.plot(x_range,Y,label="GMM")
        A.legend()
        plt.savefig(f_name + "_probability_" + str(args.group))

    # 2-dim
    elif data.shape[1] == 2:

        # plot range
        x_0_line = np.linspace(
            np.min(data.T[0]),
            np.max(data.T[0]),
            num=100
        )
        x_1_line = np.linspace(
            np.min(data.T[1]),
            np.max(data.T[1]),
            num=100
        )
        x_0_grid, x_1_grid = np.meshgrid(x_0_line, x_1_line)
        dim = x_0_grid.shape
        x_point_arr = np.stack([x_0_grid.flatten(), x_1_grid.flatten()], axis=1)


        Y = np.sum(
            weight[:,np.newaxis] * GMM.gauss(x_point_arr,mean,var),
            axis = 0
        )

        A = fig.add_subplot(121)
        A.plot(data.T[0],data.T[1],".",color="b",label = "data")
        A.plot(mean.T[0],mean.T[1],"x",color="r",markersize= 8,markeredgewidth=3, label= "centroid")
        contour = A.contour(x_point_arr.T[0].reshape(dim), x_point_arr.T[1].reshape(dim), Y.reshape(dim)) # 尤度
        A.clabel(contour,fmt='%.2f')
        A.set_xlabel("$X_{1}$")
        A.set_ylabel("$X_{2}$")
        A.set_title(f_name + "2D-probability")
        plt.legend()

        B = fig.add_subplot(122,projection="3d")
        B.plot(data.T[0],data.T[1],".",color="b",label = "data")
        B.plot(mean.T[0],mean.T[1],"x",color="r",markersize= 8,markeredgewidth=3, label= "centroid")
        B.plot_surface(x_point_arr.T[0].reshape(dim), x_point_arr.T[1].reshape(dim), Y.reshape(dim),alpha=0.7,cmap="plasma")
        B.set_xlabel("$X_{1}$")
        B.set_ylabel("$X_{2}$")
        B.set_zlabel("probability")
        B.set_title(f_name + "3D-probability")
        plt.legend()
        plt.savefig(f_name + "_probability_" + str(args.group))

    #plot likelihood

    plt.figure()
    L_g = L_g[np.where(L_g != 0)]
    plt.plot(range(len(L_g)),L_g)
    plt.xlabel("iteration")
    plt.ylabel("log likelihood")
    plt.title(f_name + " likelihood")
    plt.savefig(f_name + "_likelihood_" + str(args.group))
    plt.show()

if __name__ == "__main__":
    main()

    """

    #plot contour

    # 真の平均パラメータを指定
    mu_truth_d = np.array([[25.0, 50.0]])


    # (既知の)分散共分散行列を指定
    sigma2_truth_dd = np.array([[[600.0, 100.0], [100.0, 400.0]]])


    # 格子点を作成
    n = 125
    x_0_line = np.linspace(
        mu_truth_d[0, 0] - 3 * np.sqrt(sigma2_truth_dd[0, 0, 0]),
        mu_truth_d[0, 0] + 3 * np.sqrt(sigma2_truth_dd[0 ,0, 0]),
        num=n
    )

    # 作図用のxのx軸の値を作成
    x_1_line = np.linspace(
        mu_truth_d[0, 1] - 3 * np.sqrt(sigma2_truth_dd[0, 1, 1]),
        mu_truth_d[0, 1] + 3 * np.sqrt(sigma2_truth_dd[0, 1, 1]),
        num=n
    )
    x_0_grid, x_1_grid = np.meshgrid(x_0_line, x_1_line)
    dim = x_0_grid.shape
    x_point_arr = np.stack([x_0_grid.flatten(), x_1_grid.flatten()], axis=1)

    Z = GMM.gauss(x_point_arr,mu_truth_d,sigma2_truth_dd)


    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot()
    ax.contour(x_point_arr.T[0].reshape(dim), x_point_arr.T[1].reshape(dim), Z[0].reshape(dim)) # 尤度

    plt.show()
    """