"""データを散布図にプロットし、回帰曲線を計算するモジュール"""
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def sort_rows_and_corresponding(matrix: list, row_index: int) -> list:
    """ある行をソートしたら対応する列も一緒にソートする関数

    Args:
        matrix (list): ソートする行列
        row_index (int): ソートの基準となる行列の行番号

    Returns:
        list: ソート後の行列
    """
    # ある行をソートした時の列番号の順番を記録
    sorted_indices = sorted(range(len(matrix[row_index])), key=lambda x: matrix[row_index][x])
    # 列番号の順番をもとに、他の行も並び替える
    sorted_matrix = [[matrix[i][j] for j in sorted_indices] for i in range(len(matrix))]
    return sorted_matrix

def plot_regression_curve(x: list, y: list, degree: int) -> [list, list]:
    """x, yから最高次数degreeの回帰曲線を計算する関数

    Args:
        x (list): 説明変数
        y (list): 目的変数
        degree (int): 回帰曲線の最高次数

    Returns:
        y_pred (list): 予測値のリスト
        weights (list): 回帰曲線の係数
    """
    n = len(y)
    X = np.ones((n, 1))  # 切片の列を追加

    # 多項式特徴量の生成と特徴行列の構築 xのd乗をXに追加
    for d in range(1, degree + 1):
            X = np.column_stack((X, np.power(x, d)))
    # 重みを計算 linalg.invが逆行列, (X.T @ X)が行列の積
    weights = np.linalg.inv(X.T @ X) @ X.T @ y
    # 予測値を計算
    y_pred = X @ weights
    return y_pred, weights

def build_equation(weights: list) -> str:
    """weightから回帰曲線の文字列を返す関数

    Args:
        weights (list): 回帰曲線の係数

    Returns:
        str: 回帰曲線の式
    """
    equation = "y = "
    # 回帰曲線の最高次数
    num_features = len(weights) - 1

    for i, w in reversed(list(enumerate(weights))):
        if w != 0:# 値が0でなければ
            # 最高次数でないかつ係数が0より大きいなら係数の前に"+"を追加
            if i != num_features and w > 0:
                equation += " + "
                # 定数項ならxをつけずに出力
            if i == 0:
                equation += f"{w:.2f}"
                #xの１乗ならxに次数をつけずに出力
            elif i == 1:
                equation += f"{w:.2f} * x"
            else:
                equation += f"{w:.2f} * x^{i}"
    return equation


parent = Path(__file__).resolve().parent

rows = []
with open(parent.joinpath("data1.csv")) as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

header = rows.pop(0) # ラベルをpopで除去し、ラベルをheaderに入れる
data = np.float_(np.array(rows).T)
data = sort_rows_and_corresponding(data, 0)
y_pred ,weights= plot_regression_curve(data[0], data[1], 1)

fig, ax = plt.subplots()

ax.scatter(data[0], data[1], label = "data1", marker='o')
regression_label = build_equation(weights)
ax.plot(data[0], y_pred, color = "red", label = regression_label)
ax.set_xlabel(header[0])
ax.set_ylabel(header[1])
plt.legend()
plt.savefig(parent.joinpath("figs/data1.png"))
plt.show()