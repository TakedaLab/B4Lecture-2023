"""Ex7."""

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np

import func


class Gausian:
    def __init__(self, dim: int, pi: float) -> None:
        """Class Gausian init section.

        Args:
            dim (int): The dimention of the data.
            pi (float): The rate of this Gausian when GMM.
        """
        random_num = random.random()
        if dim == 1:
            self.mu = random_num  # 平均
            self.sigma = random_num  # 分散
        else:
            self.mu = np.full(dim, random_num)  # 平均
            self.sigma = np.diag(np.array([random_num for _ in range(dim)]))  # 分散共分散行列
        self.pi = pi
        self.dim = dim

    def calculate_probability(self, x: np.ndarray):
        """Calculate probability about a one data.

        Args:
            x (np.ndarray): Data.

        Returns:
            _type_: Probability.
        """
        if x.ndim == 0:
            return (
                np.exp(-((x - self.mu) ** 2) / 2.0 / self.sigma)
                / (np.sqrt(2 * np.pi * self.sigma))
                * self.pi
            )
        else:
            det = np.linalg.det(self.sigma)  # 分散共分散行列の行列式
            inv = np.linalg.inv(self.sigma)  # 分散共分散行列の逆行列
            return (
                np.exp(-(x - self.mu) @ inv @ (x - self.mu).T / 2.0)
                / (np.sqrt((2 * np.pi) ** self.dim * det))
                * self.pi
            )

    def calculate_probability_all(self, x: np.ndarray):
        """Calculate probability about all data.

        Args:
            x (np.ndarray): Data.

        Returns:
            _type_: Probability.
        """
        prabability_results = []
        for i in range(x.shape[0]):
            prabability_results.append(self.calculate_probability(x[i]))
        return np.array(prabability_results)


class GMM:
    def __init__(self, x: np.ndarray, mix_num: int, error: float) -> None:
        """Class GMM init section.

        Args:
            x (np.ndarray): Data.
            mix_num (int): The num of Gausian.
            error (float): The threshold for likelihood.
        """
        self.x = x  # データ
        dim = x.ndim  # データの次元数
        if dim != 1:
            dim = x.shape[1]
        self.mix_num = mix_num  # ガウスモデルの数
        self.gausian_list = []  # ガウスモデルを格納
        pi = 1 / mix_num
        for i in range(mix_num):
            self.gausian_list.append(Gausian(dim, pi))
        self.burden_rate = None  # 負担率
        self.error = error  # 誤差閾値
        self.likelihood = []  # 尤度を記録

    def calculate_probability_for_all_gausian(self, x=None):
        """Calculate probability about all gausian.

        Args:
            x (_type_, optional): Data for ploting. Defaults to None.

        Returns:
            _type_: Probability.
        """
        result = []
        for gausian in self.gausian_list:
            if type(x) == np.ndarray:
                result.append(gausian.calculate_probability_all(x))
            else:
                result.append(gausian.calculate_probability_all(self.x))
        return np.array(result).T  # 行サンプル数、列モデル数で確率を出力

    def step_e(self):
        """Calculate step e in GMM."""
        probability = self.calculate_probability_for_all_gausian()
        self.burden_rate = np.array(
            [
                probability[i] / np.sum(probability, axis=1)[i]
                for i in range(probability.shape[0])
            ]
        )

    def step_m(self):
        """Calculate stem m in GMM."""
        N = self.x.shape[0]  # サンプル数
        n_k = np.sum(self.burden_rate, axis=0)
        for i in range(len(self.gausian_list)):
            # piの更新
            self.gausian_list[i].pi = n_k[i] / np.sum(n_k)
            # muの更新
            burden_x = np.array([self.burden_rate[j, i] * self.x[j] for j in range(N)])
            mu = np.sum(burden_x, axis=0) / n_k[i]
            self.gausian_list[i].mu = mu
            # sigmaの更新
            if self.x.ndim == 1:
                sigma = 0
                for j in range(len(self.x)):
                    sigma += (self.x[j] - mu) ** 2 * self.burden_rate[j, i]
                self.gausian_list[i].sigma = sigma / n_k[i]
            else:
                sigma = np.zeros((self.x.shape[1], self.x.shape[1]))
                for j in range(len(self.x)):
                    for k in range(self.x.shape[1]):
                        sigma[k] += (
                            (self.x[j] - mu)
                            * (self.x[j] - mu)[k]
                            * self.burden_rate[j, i]
                        )
                self.gausian_list[i].sigma = sigma / n_k[i]

    def fit(self):
        counter = 0
        log_likelihood_new = 0
        finish = False
        while not finish:
            log_likelihood_old = log_likelihood_new
            self.step_e()
            self.step_m()
            # 対数尤度の計算
            probability = self.calculate_probability_for_all_gausian()
            log_likelihood_new = np.sum(np.log(np.sum(probability, axis=1)))
            self.likelihood.append(log_likelihood_new)  # 尤度を記録
            # 収束判定
            if np.abs(log_likelihood_new - log_likelihood_old) < self.error:
                finish = True
            # print(log_likelihood_new - log_likelihood_old)
            counter += 1


error = 0.0001  # 誤差閾値

if "__main__" == __name__:
    parser = argparse.ArgumentParser(
        prog="main.py",  # プログラム名
        usage="B4 Lecture Ex6.",  # プログラムの利用方法
        description="Principal Component Analysis.",  # 引数のヘルプの前に表示
        epilog="end",  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument("data_path", help="Select File.")
    parser.add_argument("mix_num", help="Input the number of cluster.", type=int)
    args = parser.parse_args()

    data = func.read_csv("../" + args.data_path)
    if args.data_path == "data1.csv":
        fig0, ax0 = plt.subplots(1, 1)
        ax0.set_title(f"Gausian {args.data_path}. k={args.mix_num}")
        ax0.set_xlabel("x")
        ax0.set_ylabel("Probability")
        fig3, ax3 = plt.subplots(1, 1)
        ax3.set_title(f"Log Likelihood data1. k={args.mix_num}")
        ax3.set_xlabel("The number of loops")
        ax3.set_ylabel("Log likelihood")

        gmm1 = GMM(data, args.mix_num, error)
        gmm1.fit()
        plot_x = np.linspace(np.min(data), np.max(data), 100)
        probability1 = gmm1.calculate_probability_for_all_gausian()
        probability1_plot = gmm1.calculate_probability_for_all_gausian(x=plot_x)
        # ax0.scatter(data, [0 for i in range(data.shape[0])]) #データを図示
        # データをクラスタリングして図示
        max_probability_index = []
        for i in range(data.shape[0]):
            max_probability_index.append(np.argmax(probability1[i]))
        clustered_data = np.append(data, max_probability_index)
        clustered_data = clustered_data.reshape([2, -1]).T
        for i in range(args.mix_num):
            clustered_data_per = clustered_data[clustered_data[:, -1] == i]
            ax0.scatter(
                clustered_data_per[:, 0], [0 for i in range(len(clustered_data_per))]
            )
        # GMMを図示
        ax0.plot(plot_x, np.sum(probability1_plot, axis=1))
        # 重心を図示
        ax0.scatter(
            [gmm1.gausian_list[i].mu for i in range(gmm1.mix_num)],
            [0 for i in range(gmm1.mix_num)],
            color="red",
            s=600,
            marker="*",
        )
        # 尤度の変遷を出力
        ax3.plot(np.arange(1, len(gmm1.likelihood) + 1), gmm1.likelihood)

        fig0.savefig("data1_gausian_kari.png")
        fig3.savefig("data1_likelihood_kari.png")

    elif args.data_path == "data2.csv" or args.data_path == "data3.csv":
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection="3d")
        ax1.set_title(f"Gausian 3D {args.data_path}. k={args.mix_num}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        fig2, ax2 = plt.subplots(1, 1)
        ax2.set_title(f"Gausian 2D {args.data_path}. k={args.mix_num}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        fig3, ax3 = plt.subplots(1, 1)
        ax3.set_title(f"Log Likelihood {args.data_path}. k={args.mix_num}")
        ax3.set_xlabel("The number of loops")
        ax3.set_ylabel("Log likelihood")

        gmm2 = GMM(data, args.mix_num, error)
        gmm2.fit()
        # ax1.scatter(data[:, 0], data[:, 1], [0 for i in range(data.shape[0])]) #データを図示
        # ax2.scatter(data[:, 0], data[:, 1])
        # データをクラスタリングして図示
        probability2 = gmm2.calculate_probability_for_all_gausian()
        max_probability_index = []
        for i in range(data.shape[0]):
            max_probability_index.append(np.argmax(probability2[i]))
        clustered_data = np.append(data, np.array([max_probability_index]).T, axis=1)
        for i in range(args.mix_num):
            clustered_data_per = clustered_data[clustered_data[:, -1] == i]
            ax2.scatter(clustered_data_per[:, 0], clustered_data_per[:, 1])
            ax1.scatter(
                clustered_data_per[:, 0],
                clustered_data_per[:, 1],
                [0 for i in range(clustered_data_per.shape[0])],
            )  # データを図示
        # GMMを図示
        plot_x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
        plot_y = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
        mesh_x, mesh_y = np.meshgrid(plot_x, plot_y)
        mesh_x_reshape = mesh_x.reshape(-1)
        mesh_y_reshape = mesh_y.reshape(-1)
        probability2_plot = gmm2.calculate_probability_for_all_gausian(
            x=np.array([mesh_x_reshape, mesh_y_reshape]).T
        )
        mesh_z = np.sum(probability2_plot, axis=1).reshape([100, -1])
        ax1.plot_surface(mesh_x, mesh_y, mesh_z, alpha=0.5)  # GMMを図示(3次元)
        ax2.contour(mesh_x, mesh_y, mesh_z, alpha=0.5)  # GMMを図示(等高線)
        # 重心を図示
        ax1.scatter(
            [gmm2.gausian_list[i].mu[0] for i in range(gmm2.mix_num)],
            [gmm2.gausian_list[i].mu[1] for i in range(gmm2.mix_num)],
            [0 for i in range(gmm2.mix_num)],
            color="red",
            s=600,
            marker="*",
        )
        ax2.scatter(
            [gmm2.gausian_list[i].mu[0] for i in range(gmm2.mix_num)],
            [gmm2.gausian_list[i].mu[1] for i in range(gmm2.mix_num)],
            color="red",
            s=600,
            marker="*",
        )
        # 尤度の変遷を出力
        ax3.plot(np.arange(1, len(gmm2.likelihood) + 1), gmm2.likelihood)

        data_name = args.data_path.replace(".csv", "")
        fig1.savefig(f"{data_name}_gausian3d_kari.png")
        fig2.savefig(f"{data_name}_gausian2d_kari.png")
        fig3.savefig(f"{data_name}_likelihood_kari.png")

    plt.show()
    plt.clf()
    plt.close()
