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
    X = csv_file[:, :-1]
    
    X_plot = []
    if np.ndim(X_plot) != 1:
        for i in range(X.shape[1]):
            X_plot.append(np.linspace(np.min(X[:,i]), np.max(X[:,i]), 100))
    else:
        X_plot.append(np.linspace(np.min(X), np.max(X), 100))
    X_plot = np.array(X_plot).T
    print(X_plot)
    
    X_train = copy.deepcopy(X)
    X_train_plot = copy.deepcopy(X_plot)
    for i in range(dim-1):
        if np.ndim(X_train) != 1:
            for j in range(X_train.shape[1]):
                X_dim = X_train[ : , j] ** (i+2)
                X = np.insert(X, -1, X_dim, axis=1)
                X_dim_plot = X_train_plot[ : , j] ** (i+2)
                X_plot = np.insert(X_plot, -1, X_dim_plot, axis=1)
        else:
            X_dim = X_train ** i
            X = (np.concatenate([X, X_dim], 0)).reshape(-1, 2)
            X_dim_plot = X_train_plot ** i
            X_plot = (np.concatenate([X_plot, X_dim_plot], 0)).reshape(-1, 2)
    
    X = np.insert(X, 0, 1, axis=1) #左に1の列を追加(定数項)
    X_plot = np.insert(X_plot, 0, 1, axis=1)
    y = csv_file[:, -1]
    #print(X_plot)

    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    #print(X)
    #print(XTX)
    w = scipy.linalg.solve(XTX, XTy)
    print(w)
    
    return np.dot(X_plot, w)


def main():
    """Do main action."""
    filename = "data2.csv"
    csv_file = read_csv(filename)
    csv_file = csv_file.T #転置して操作しやすく
    plt.scatter(csv_file[0], csv_file[1])

    X = csv_file[:-1].T
    X = np.insert(X, 0, 1, axis=1)
    print(X)
    y = csv_file[-1].T

    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    w = scipy.linalg.solve(XTX, XTy)
    print(w)
    
    x_start = np.min(csv_file[0])
    x_goal = np.max(csv_file[0])
    
    start = np.array([1, x_start]).T
    goal = np.array([1, x_goal]).T
    
    #plt.plot([x_start, x_goal], [np.dot(start, w), np.dot(goal, w)], color="red")
    plt.plot(least_squares_method(csv_file.T, 3))

    plt.show()
    plt.clf()
    plt.close()


if "__main__" == __name__:
    main()