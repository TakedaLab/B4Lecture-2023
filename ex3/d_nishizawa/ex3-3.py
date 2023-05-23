"""データを散布図にプロットし、回帰曲線を計算するモジュール"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path

def sort_rows_and_corresponding(matrix: list, row_index: int) -> list:
    """_summary_

    Args:
        matrix (list): ソートする行列
        row_index (int): ソートの基準となる行列の行番号

    Returns:
        list: ソート後の行列
    """
    sorted_indices = sorted(range(len(matrix[row_index])), key=lambda x: matrix[row_index][x])
    sorted_matrix = [[matrix[i][j] for j in sorted_indices] for i in range(len(matrix))]
    return sorted_matrix

def plot_regression_curve(x1: list, x2: list, y: list, degree: int) -> list:
    """x1, x2, yから最高次数degreeの回帰曲線を計算する関数

    Args:
        x1 (list): 説明変数
        x2 (list): 説明変数
        y (list): 目的変数
        degree (int): 回帰曲線の最高次数

    Returns:
        list: 予測値のリスト
    """
    n = len(y)
    X = np.ones((n, 1))  # 切片の列を追加

    # 多項式特徴量の生成と特徴行列の構築
    for d in range(1, degree + 1):
        for i in range(d + 1):
            # x1^i * x2^(d - i)を計算
            X = np.column_stack((X, (np.power(x1, i) * np.power(x2, d - i))))

    # 重みを計算 linalg.invが逆行列, (X.T @ X)が行列の積
    weights = np.linalg.inv(X.T @ X) @ X.T @ y
    # 予測値を計算
    y_pred = X @ weights
    return y_pred

parent = Path(__file__).resolve().parent

rows = []
with open(parent.joinpath("data3.csv")) as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

header = rows.pop(0) # ラベルをpopで除去し、ラベルをheaderに入れる
data = np.float_(np.array(rows).T)
data = sort_rows_and_corresponding(data, 0)
y_pred = plot_regression_curve(data[0], data[1], data[2], 1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.scatter(data[0], data[1], data[2], label = "data3", marker='o')
new_data0, new_data1 = np.meshgrid(np.unique(data[0]), np.unique(data[1]))
y_new = griddata((data[0], data[1]), y_pred, (new_data0, new_data1))
ax.plot_surface(new_data0, new_data1, y_new, color = "red")
ax.set_xlabel(header[0])
ax.set_ylabel(header[1])
ax.set_zlabel(header[2])
plt.legend()
plt.savefig(parent.joinpath("figs/data3.png"))
plt.show()