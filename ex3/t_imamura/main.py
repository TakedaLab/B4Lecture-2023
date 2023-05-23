"""ex3 least squares method"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import scipy


def read_csv(filename: str):
    """
    Read csv file.

    Args:
        filename (str): File name.

    Returns:
       np.ndarray: The data in the csv file.
    """
    return np.loadtxt(filename, delimiter=",", skiprows=1)  # csvをnumpy型の行列で読み込み


def least_squares_method(csv_file: np.ndarray, dim: int, lam: float):
    """
    Calculate Least Squares Method.

    Args:
        csv_file (np.ndarray):  The data in the csv file.
        dim (int): The dimension of the each variable.
        lam (float): Regularization factor.

    Returns:
       np.ndarray: The x data for plots.
       np.ndarray: The y data for plots.
       np.ndarray: The factors for print y from x.
    """
    X = csv_file[:, :-1]  # 学習用に入力データの次元-1分を確保
    X_plot = []  # プロット用のx座標データ

    # プロット用のx座標データを作成
    if X.shape[1] != 1:  # 3次元データ以上の時
        for_meshgrid = []
        for i in range(X.shape[1]):
            for_meshgrid.append(
                np.linspace(np.min(X[:, i]), np.max(X[:, i]), 128)
            )  # 入力データの次元-1分のプロットデータを作成
        meshgrid_x1, meshgrid_x2 = np.meshgrid(
            for_meshgrid[0], for_meshgrid[1]
        )  # 3次元グラフ出力のためにメッシュを作成
        X_plot.append(np.ravel(meshgrid_x1))  # メッシュを1次元化してappend
        X_plot.append(np.ravel(meshgrid_x2))
    else:  # 2次元データの時
        X_plot.append(np.linspace(np.min(X), np.max(X), 128))  # 入力データの次元-1分のプロットデータを作成
    X_plot = np.array(X_plot).T

    # 拡張に備えて元状態を記憶
    X_train = copy.deepcopy(X)
    X_train_plot = copy.deepcopy(X_plot)

    # データをdim分拡張
    for i in range(dim - 1):
        for j in range(X_train.shape[1]):  # 入力データの次元全てに対して拡張を適用
            X_dim = X_train[:, j] ** (i + 2)
            X = np.insert(X, -1, X_dim, axis=1)
            X_dim_plot = X_train_plot[:, j] ** (i + 2)
            X_plot = np.insert(X_plot, -1, X_dim_plot, axis=1)

    # 左に1の列を追加(定数項)
    X = np.insert(X, 0, 1, axis=1)
    X_plot = np.insert(X_plot, 0, 1, axis=1)

    # yを入力データの最後の次元と定義
    y = csv_file[:, -1]

    # 最小二乗法を適用
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    if lam > 0:  # 正則化するとき
        w = scipy.linalg.solve(XTX + lam * np.eye(X.shape[1]), XTy)
    else:  # しない時
        w = scipy.linalg.solve(XTX, XTy)

    # プロット用の元次元-1データと学習後のyデータを出力
    return X_train_plot, np.dot(X_plot, w), w


def set_label(w: np.ndarray, filedim: int, funcdim: int, lam=-1):
    """
    Generate label to plot data.

    Args:
        w (np.ndarray): The factors for print y from x.
        filedim (int): The number of variable.
        funcdim (int): The dimension of each variable
        lam (int, optional): Regularization factor. Defaults to -1.

    Returns:
        str: Label name.
    """
    label = f"y = {round(w[0], 2)}"
    filedim -= 1  # 計算の便宜上1ひく
    if filedim == 1:  # 元データが2次元の時
        for j in range(funcdim):
            if w[1 + filedim * j] < 0:
                label += f"{round(w[1 + filedim * j], 2)}$x^{j+1}$"
            elif w[1 + filedim * j] > 0:
                label += f"+{round(w[1 + filedim * j], 2)}$x^{j+1}$"
    else:  # 元データが3次元以上の時
        for i in range(1, filedim + 1):
            for j in range(funcdim):
                if w[i + filedim * j] < 0:
                    label += f"{round(w[i + filedim * j], 2)}$x_{i}^{j+1}$"
                elif w[i + filedim * j] > 0:
                    label += f"+{round(w[i + filedim * j], 2)}$x_{i}^{j+1}$"
    if lam > 0:
        label += " (Regularized)"
    else:
        label += " (Non-Regularized)"
    return label


def two_dim_show(filename: str, dim: int, lam=-1):
    """
    Print 2d-data.

    Args:
        filename (str): The file name.
        dim (int): The dimension of each variable
        lam (int, optional): Regularization factor. Defaults to -1.
    """
    # ファイル読み込み
    csv_file = read_csv(filename)
    fig, ax = plt.subplots(1, 1)
    # データの散布図を出力
    ax.scatter(csv_file[:, 0], csv_file[:, 1], label=filename)
    # 最小二乗法計算
    x_pre, y_pre, w = least_squares_method(csv_file, dim, lam)
    # 近似グラフを出力
    ax.plot(x_pre, y_pre, color="red", label=set_label(w, 2, dim, lam))
    ax.set_title(filename)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()
    fig.savefig(filename.replace(".csv", ".png"))


def three_dim_show(filename: str, dim: int, lam=-1):
    """
    Print 3d-data.

    Args:
        filename (str): The file name.
        dim (int): The dimension of each variable
        lam (int, optional):  Regularization factor. Defaults to -1.
    """
    # ファイル読み込み
    csv_file = read_csv(filename)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # データの散布図を出力
    ax.scatter(csv_file[:, 0], csv_file[:, 1], csv_file[:, 2], label=filename)
    # 最小二乗法計算
    x_pre, y_pre, w = least_squares_method(csv_file, dim, lam)
    # 近似グラフを出力
    ax.plot_wireframe(
        x_pre[:, 0].reshape(128, 128),
        x_pre[:, 1].reshape(128, 128),
        y_pre.reshape(128, 128),
        color="red",
        alpha=0.3,
        label=set_label(w, 3, dim, lam),
    )
    ax.set_title(filename)
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.set_zlabel("y")
    ax.legend()
    plt.show()
    fig.savefig(filename.replace(".csv", ".png"))


def main():
    """Do main action."""
    lam = -1  # 正則化係数

    two_dim_show("data1.csv", 1, lam)
    two_dim_show("data2.csv", 3, lam)
    three_dim_show("data3.csv", 2, lam)

    plt.clf()
    plt.close()


if "__main__" == __name__:
    main()
