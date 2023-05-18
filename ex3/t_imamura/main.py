"""ex3"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import soundfile
import csv
import copy


def read_csv(filename: str):
    return np.loadtxt(filename, delimiter=",", skiprows=1) #csvをnumpy型の行列で読み込み

def least_squares_method(csv_file: np.ndarray, dim:int):
    X = csv_file[:, :-1] #学習用に入力データの次元-1分を確保
    X_plot = [] #プロット用のx座標データ
    
    #プロット用のx座標データを作成
    print(X)
    print(np.ndim(X))
    if X.shape[1] != 1:
        for_meshgrid = []
        for i in range(X.shape[1]):
            for_meshgrid.append(np.linspace(np.min(X[:,i]), np.max(X[:,i]), 128)) #入力データの次元-1分のプロットデータを作成
        meshgrid_x1, meshgrid_x2 = np.meshgrid(for_meshgrid[0], for_meshgrid[1]) #3次元グラフ出力のためにメッシュを作成
        X_plot.append(np.ravel(meshgrid_x1)) #メッシュを1次元化してappend
        X_plot.append(np.ravel(meshgrid_x2))
    else:
        X_plot.append(np.linspace(np.min(X), np.max(X), 128))
    X_plot = np.array(X_plot).T
    
    #拡張に備えて元状態を記憶
    X_train = copy.deepcopy(X)
    X_train_plot = copy.deepcopy(X_plot)
    
    #データをdim分拡張
    for i in range(dim-1):
        if np.ndim(X_train) != 1:
            for j in range(X_train.shape[1]): #入力データの次元全てに対して拡張を適用
                X_dim = X_train[ : , j] ** (i+2)
                X = np.insert(X, -1, X_dim, axis=1)
                X_dim_plot = X_train_plot[ : , j] ** (i+2)
                X_plot = np.insert(X_plot, -1, X_dim_plot, axis=1)
        else:
            X_dim = X_train ** i
            X = (np.concatenate([X, X_dim], 0)).reshape(-1, 2)
            X_dim_plot = X_train_plot ** i
            X_plot = (np.concatenate([X_plot, X_dim_plot], 0)).reshape(-1, 2)
    
    #左に1の列を追加(定数項)
    X = np.insert(X, 0, 1, axis=1)
    X_plot = np.insert(X_plot, 0, 1, axis=1)
    
    #yを入力データの最後の次元と定義
    y = csv_file[:, -1]

    #最小二乗法を適用
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    w = scipy.linalg.solve(XTX, XTy)
    print(w)
    #プロット用の元次元-1データと学習後のyデータを出力
    return X_train_plot, np.dot(X_plot, w)


def two_dim_show(filename:str, dim:int):
    csv_file = read_csv(filename)
    fig, ax = plt.subplots(1,1)
    ax.scatter(csv_file[:, 0], csv_file[:, 1])
    x_pre, y_pre = least_squares_method(csv_file, dim)
    ax.plot(x_pre, y_pre, color="red")


def three_dim_show(filename:str, dim:int):
    csv_file = read_csv(filename)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")
    ax.scatter(csv_file[:, 0], csv_file[:, 1], csv_file[:, 2])
    x_pre, y_pre = least_squares_method(csv_file, dim)
    ax.plot_wireframe(x_pre[:, 0].reshape(128, 128), x_pre[:, 1].reshape(128, 128), y_pre.reshape(128, 128), color="red", alpha=0.3)

def main():
    """Do main action."""
    """
    filename = "data2.csv"
    csv_file = read_csv(filename)
    plt.scatter(csv_file[:, 0], csv_file[:, 1])
    
    x_pre, y_pre = least_squares_method(csv_file, 3)
    plt.plot(x_pre, y_pre, color="red")
    
    X = csv_file[:, :-1]
    X = np.insert(X, 0, 1, axis=1)
    print(X)
    y = csv_file[:, -1]

    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    w = scipy.linalg.solve(XTX, XTy)
    print(w)
    
    x_start = np.min(csv_file[0])
    x_goal = np.max(csv_file[0])
    
    start = np.array([1, x_start]).T
    goal = np.array([1, x_goal]).T
    """
    #plt.plot([x_start, x_goal], [np.dot(start, w), np.dot(goal, w)], color="red")

    two_dim_show("data1.csv", 1)
    two_dim_show("data2.csv", 3)
    three_dim_show("data3.csv", 3)
    
    plt.show()
    plt.clf()
    plt.close()


if "__main__" == __name__:
    main()