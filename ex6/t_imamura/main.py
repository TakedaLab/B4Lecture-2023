"""Ex6."""

import argparse
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

import func


class PCA:
    def __init__(self, data: np.ndarray, normalize: bool):
        """Init.

        Args:
            data (np.ndarray): The data for pca.
            normalize (bool): Normalize or not.
        """
        if normalize:
            self.data = (data - data.mean(axis=0)) / np.std(data, axis=0)  # 標準化
        else:
            self.data = data  # 入力データ

        self.dim = self.data.shape[1]  # データの次元数
        self.gravity = (
            np.dot(np.ones(self.data.shape[0]), self.data) / self.data.shape[0]
        )  # 重心
        self.cov = np.cov(self.data, rowvar=False)  # 分散共分散行列
        self.eig_val, self.eig_vec = np.linalg.eig(self.cov)  # 分散共分散行列の固有値・固有ベクトル
        desc = np.argsort(self.eig_val)[::-1]  # 降順
        self.eig_vec = self.eig_vec[:, desc]  # 固有ベクトルを固有値の大きい順に
        self.eig_val = np.sort(self.eig_val)[::-1]
        self.contribution_rate = self.eig_val / np.sum(self.eig_val)  # 寄与率
        self.plot_data = np.linspace(
            np.min(self.data[:, 0]), np.max(self.data[:, 0]), 10
        )  # 出力用のxのデータ

    def plot(self, m: int, ax):
        """Plot pc.

        Args:
            m (int): The num of component.
            ax : For plot.
        """
        a = self.eig_vec[m - 1][1:] / self.eig_vec[m - 1][0]  # 割合
        cr = self.contribution_rate[m - 1]  # 寄与率
        if self.dim == 2:
            ax.scatter(self.data[:, 0], self.data[:, 1], color="b")  # データ点
            ax.plot(
                self.plot_data + self.gravity[0],
                self.plot_data * a + self.gravity[1],
                label=f"pc{m} cr = {cr:.3f}",
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        elif self.dim == 3:
            ax.scatter(
                self.data[:, 0], self.data[:, 1], self.data[:, 2], color="b"
            )  # データ点
            ax.plot(
                self.plot_data + self.gravity[0],
                self.plot_data * a[0] + self.gravity[1],
                self.plot_data * a[1] + self.gravity[2],
                label=f"pc{m} cr = {cr:.3f}",
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

    def transform(self, dim: int):
        """Transform the data based of pca.

        Args:
            dim (int): Dimention.

        Returns:
            np.ndarray: Transformed data.
        """
        transformed_data = np.dot(self.eig_vec[:, :dim].T, self.data.T)
        return transformed_data[:dim]

    def accum(self, dim: int):
        """Calculate Cumlative contribution rate.

        Args:
            dim (int): Dimention.

        Returns:
            float: Cumlative contributoin rate.
        """
        return np.sum(self.contribution_rate[:dim])


if "__main__" == __name__:
    parser = argparse.ArgumentParser(
        prog="main.py",  # プログラム名
        usage="B4 Lecture Ex6.",  # プログラムの利用方法
        description="Principal Component Analysis.",  # 引数のヘルプの前に表示
        epilog="end",  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument("data_path", help="Select File.")
    args = parser.parse_args()

    data = func.read_csv(f"../{args.data_path}")  # データの読み込み
    pca = PCA(data, True)  # インスタンス
    if args.data_path == "data1.csv":
        fig0, ax0 = plt.subplots(1, 1)
        ax0.set_title("principal component analysis data1")
        pca.plot(1, ax0)
        pca.plot(2, ax0)
        ax0.scatter(
            pca.gravity[0], pca.gravity[1], color="red", label="Center of Gravity"
        )  # 重心を表示
        ax0.legend()
        fig0.savefig("Figure/data1_pca.png")
    elif args.data_path == "data2.csv":
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection="3d")
        ax1.set_title("principal component analysis data2")
        fig2, ax2 = plt.subplots(1, 1)
        ax2.set_title("The result of compression data2")
        pca.plot(1, ax1)
        pca.plot(2, ax1)
        pca.plot(3, ax1)
        ax1.scatter(
            pca.gravity[0],
            pca.gravity[1],
            pca.gravity[2],
            color="red",
            label="Center of Gravity",
        )  # 重心を表示
        transformed_data = pca.transform(2)
        ax2.scatter(transformed_data[0], transformed_data[1])
        ax2.set_xlabel("pc1")
        ax2.set_ylabel("pc2")
        ax1.legend()
        fig1.savefig("Figure/data2_pca.png")
        fig2.savefig("Figure/data2_compression.png")
    elif args.data_path == "data3.csv":
        fig3, ax3 = plt.subplots(1, 1)
        ax3.set_title("Cumlative contribution rate data3")
        for i in range(1, pca.dim + 1):
            # print(pca.accum(i))
            if pca.accum(i) >= 0.9:
                print(i)
                break
        num_pc = [i for i in range(100)]
        data3_acr = [pca.accum(i) for i in range(100)]
        ax3.plot(num_pc, data3_acr, label="ccumlative contribution rate")
        ax3.plot(num_pc, [0.9 for i in range(100)], label="90%")
        ax3.plot([66, 66], [0, 1], label="66")
        ax3.set_xlabel("The num of principal components ")
        ax3.set_ylabel("Cumlative contriburion rate")
        ax3.legend()
        fig3.savefig("Figure/data3_acr.png")
    else:
        print("Wrong data path.")

    plt.show()
    plt.clf()
    plt.close()
