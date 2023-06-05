"""The Function File."""

import copy

import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import soundfile


def read_csv(filename: str):
    """
    Read csv file.

    Args:
        filename (str): File name.

    Returns:
    np.ndarray: The data in the csv file.
    """
    return np.loadtxt(filename, delimiter=",", skiprows=1)  # csvをnumpy型の行列で読み込み


def load_sound_file(sample_path: str) -> tuple[np.ndarray, int, int]:
    """Load sound file.

    Args:
        sample_path (str): The path to the sample file.

    Returns:
        tuple[np.ndarray, int, int]:
        The sample data, sample rate, the length of sample data.
    """
    # 音源の読み込み
    sample_data, sample_rate = soundfile.read(sample_path)
    N = len(sample_data)  # サンプル数を出しておく
    return sample_data, sample_rate, N


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


def spectrogram(data: np.ndarray, sample_rate: int, ax, fig, indx_ax=-1, F_size=1024):
    """Calculate STFT and show the spectrogram.

    Args:
        ax (list): The place to plot.
        fig: The figure of plt.
        indx_ax (int): The number of ax.
        data (np.ndarray): The data of the sound.
        sample_rate (int): The sample rate of sample_data.
        F_size (int, optional): The size of frame. Defaults to 1024.
    """
    f, t, spec = scipy.signal.spectrogram(
        data, sample_rate, nperseg=F_size
    )  # 短時間フーリエ変換して振幅成分を取り出す
    if indx_ax != -1:
        ax_spec = ax[indx_ax].pcolormesh(
            t, f, 20 * np.log10(spec), vmax=1e-6, cmap="CMRmap"
        )  # dbに変換してスペクトログラムを表示
        ax[indx_ax].set_title("Spectrogram")
        ax[indx_ax].set_xlabel("Time [sec]")
        ax[indx_ax].set_ylabel("Fequency [Hz]")
        fig.colorbar(ax_spec, ax=ax[indx_ax], aspect=5, location="right")
    else:
        ax_spec = ax.pcolormesh(
            t, f, 20 * np.log10(spec), vmax=1e-6, cmap="CMRmap"
        )  # dbに変換してスペクトログラムを表示
        ax.set_title("Spectrogram")
        ax.set_xlabel("Time [sec]")
        ax.set_ylabel("Fequency [Hz]")
        fig.colorbar(ax_spec, ax=ax, aspect=5, location="right")


def auto_correlation(data: np.ndarray) -> np.ndarray:
    """Calculate Auto Correlation.

    Args:
        data (np.ndarray): Data to apply auto correlation.

    Returns:
        np.ndarray: The result of auto correlation.
    """
    ac = np.zeros(len(data))
    for m in range(len(data)):
        if m == 0:
            ac[m] += np.sum(data * data)
        else:
            ac[m] += np.sum(data[0:-m] * data[m:])
    return ac


def stft(
    sample_data: np.ndarray, F_size=1024, OverRap=0.5, window=scipy.signal.hamming
) -> tuple[np.ndarray, int]:
    """Calculate stft.

    Args:
        sample_data (np.ndarray): Data to apply stft.
        F_size (int, optional): Window size. Defaults to 1024.
        OverRap (float, optional): The rate of Overrap. Defaults to 0.5.
        window (_type_, optional): Window function. Defaults to scipy.signal.hamming.

    Returns:
        tuple[np.ndarray, int]: The result of stft, the number of Frame.
    """
    fft = []  # fftの結果を格納する
    dist_of_Frame = int(F_size - F_size * OverRap)  # フレーム間の距離
    fft_start = 0  # フレームの開始位置
    counter = 0  # フレーム数を数える
    while fft_start + F_size <= len(sample_data):  # 次のフレーム幅の末端がサンプル数を超えるまでfft
        Frame = sample_data[fft_start : fft_start + F_size]  # 音源データからフレームを切り出す
        fft_per = np.fft.fft(window(F_size) * Frame)  # fftを実行
        fft.append(fft_per)  # 結果を格納
        fft_start += dist_of_Frame
        counter += 1
    return np.array(fft), counter


def Quefrency(
    sample_data: np.ndarray,
    sample_rate: int,
    F_size: int,
    OverRap: float,
    lift_ms=0.018,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Calculate quefrency method.

    Args:
        sample_data (np.ndarray): Data to apply quefrancy method.
        sample_rate (int): The sample rate.
        F_size (int): Window size.
        OverRap (float): The rate of Overrap.
        lift_ms (float, optional): The lift time. Defaults to 0.018.

    Returns:
        tuple[np.ndarray, np.ndarray, int]:
        Low quefrency ingredient,
        High quefrency ingredient,
        The number of frame.
    """
    low_quefrency = []
    high_quefrancy = []
    lift = int(lift_ms * F_size)  # ケフレンシーでのリフト境界
    dist_of_Frame = F_size - F_size * OverRap  # フレーム間の距離
    fft_start = 0  # フレームの開始位置
    fft, counter = stft(sample_data, F_size, OverRap)  # fftを実行
    for fft_per in fft:
        fft_log = 20 * np.log10(fft_per)  # 対数をとる(ついでにdb変換)
        quefrency = np.real(np.fft.ifft(fft_log))  # 逆変換
        low_quef_per = copy.deepcopy(quefrency)  # 低周波成分
        low_quef_per[lift : F_size - lift] = 0
        high_quef_per = copy.deepcopy(quefrency)  # 高周波成分
        high_quef_per -= low_quef_per
        fft_start += dist_of_Frame
        low_quefrency.append(low_quef_per)
        high_quefrancy.append(high_quef_per)

    return np.array(low_quefrency), np.array(high_quefrancy), counter


def LevinsonDurbin(r, lpc_parameter):
    """Calculate Levinson Durbin matrix.

    Args:
        r (_type_): The ingredient of Levinson Durbin matrix.
        lpc_parameter (_type_): LPC parameter.

    Returns:
        _type_: The result of Levinson Durbin matrix.
    """
    a = np.zeros(lpc_parameter + 1)
    e = np.zeros(lpc_parameter + 1)
    for i in range(lpc_parameter):
        if i == 0:
            a[0] = 1.0
            a[1] = -r[1] / r[0]
            e[1] = r[0] + r[1] * a[1]
            k = -r[1] / r[0]
        else:
            k = 0.0
            for j in range(i + 1):
                k -= a[j] * r[i + 1 - j]  # kを更新
            k /= e[i]
            U = [1]
            U.extend([a[x] for x in range(1, i + 1)])
            U.append(0)
            V = [0]
            V.extend([a[x] for x in range(i, 0, -1)])
            V.append(1)
            a = np.array(U) + k * np.array(V)  # aを更新
            e[i + 1] = e[i] * (1.0 - k * k)  # eを更新

    return a, e[-1]


def select_random(data: np.ndarray, k):
    """Randomly selecting algorism.

    Args:
        data (np.ndarray): Data which randomly select from.
        k (_type_): The number of selected data.

    Returns:
        Randomly selected data.
    """
    index = random.sample(range(len(data)), k)
    value = data[index]
    return np.array(value)


def k_means(
    data: np.ndarray, k: int, max_loop: int, projection: str, ax=None, centroids=-1
):
    """K-means clustering.

    Args:
        data (np.ndarray): Clustering data.
        k (int): The num of cluster.
        max_loop (int): The max times of loop.
        projection (str): Dimention of data.
        ax (_type_, optional): For plot. Defaults to None.
        centroids (int, optional): The centroids. Defaults to -1.

    Returns:
        The centroids, cluster, clustering error.
    """
    counter = 0
    if type(centroids) != np.ndarray:
        centroids = select_random(data, k)  # 初期セントロイド
    cluster = np.zeros(data.shape[0])
    cluster_pre = np.ones(data.shape[0])
    error = 0  # 誤差
    while (
        not ((cluster == cluster_pre).all()) and counter < max_loop
    ):  # クラスタに変動がなくなるか、試行回数がmaxをこえるまで
        cluster_pre = copy.deepcopy(cluster)
        # クラスタ再分類
        for data_index in range(len(data)):
            dist = []
            for centroids_per in centroids:
                dist.append(
                    np.linalg.norm(data[data_index] - centroids_per, ord=2)
                )  # ユークリッド距離計算
            cluster[data_index] = np.argmin(np.array(dist))  # クラスタ再分類
            error += np.min(np.array(dist))  # 誤差評価

        # セントロイド再計算
        centroids = np.zeros_like((centroids))  # セントロイドを初期化
        for i in range(k):
            cluster_num = 0
            for ind in range(len(cluster)):
                if i == cluster[ind]:
                    centroids[i] += data[ind]
                    cluster_num += 1
            centroids[i] /= cluster_num
        counter += 1
    error /= data.shape[0]

    # クラスタリングの結果を表示
    if ax:
        cmap = plt.get_cmap("tab10")
        clustered_data = np.append(data, cluster.reshape(cluster.shape[0], 1), axis=1)
        for i in range(k):
            clustered_data_per = clustered_data[clustered_data[:, -1] == i]
            if projection == "2d":
                ax.scatter(
                    clustered_data_per[:, 0],
                    clustered_data_per[:, 1],
                    color=cmap(i),
                    label=f"cluster{i}",
                )
                ax.scatter(
                    centroids[:, 0], centroids[:, 1], s=600, c="black", marker="*"
                )
            elif projection == "3d":
                ax.scatter(
                    clustered_data_per[:, 0],
                    clustered_data_per[:, 1],
                    clustered_data_per[:, 2],
                    color=cmap(i),
                    label=f"cluster{i}",
                )
                ax.scatter(
                    centroids[:, 0],
                    centroids[:, 1],
                    centroids[:, 2],
                    s=600,
                    c="black",
                    marker="*",
                )
        ax.legend()
    # print(f"error:{error}")

    return centroids, cluster, error


def LBG(
    data: np.ndarray, k: int, max_loop: int, projection: str, ax=None, delta=0.0001
):
    """LBG clustering.

    Args:
        data (np.ndarray): clustering data.
        k (int): The num of cluster.
        max_loop (int): _description_
        projection (str):The max times of loop.
        projection (str): Dimention of data.
        ax (_type_, optional): For plot. Defaults to None.
        delta (float, optional): Micro vector. Defaults to 0.0001.

    Returns:
        The centroids, clustering error.
    """
    centroids = select_random(data, 1)  # 初期セントロイド
    while centroids.shape[0] <= k:
        centroids, cluster, error = k_means(
            data, centroids.shape[0], max_loop, projection, centroids=centroids
        )  # k-means法を実行
        if centroids.shape[0] == k:
            break
        centroids_a = centroids + delta  # deltaでセントロイドを分割
        centroids_b = centroids - delta
        centroids = np.concatenate([centroids_a, centroids_b], axis=0)

    # 結果の表示
    if ax:
        cmap = plt.get_cmap("tab10")
        clustered_data = np.append(data, cluster.reshape(cluster.shape[0], 1), axis=1)
        for i in range(k):
            clustered_data_per = clustered_data[clustered_data[:, -1] == i]
            if projection == "2d":
                ax.scatter(
                    clustered_data_per[:, 0],
                    clustered_data_per[:, 1],
                    color=cmap(i),
                    label=f"cluster{i}",
                )
                ax.scatter(
                    centroids[:, 0], centroids[:, 1], s=600, c="black", marker="*"
                )
            elif projection == "3d":
                ax.scatter(
                    clustered_data_per[:, 0],
                    clustered_data_per[:, 1],
                    clustered_data_per[:, 2],
                    color=cmap(i),
                    label=f"cluster{i}",
                )
                ax.scatter(
                    centroids[:, 0],
                    centroids[:, 1],
                    centroids[:, 2],
                    s=600,
                    c="black",
                    marker="*",
                )
        ax.legend()
    return centroids, error


def Hz_to_mel(f):
    """Convert mel from Hz.

    Args:
        f: The num or list of Hz.

    Returns:
        Mel.
    """
    return 1127 * np.log(f / 700.0 + 1.0)


def mel_to_Hz(mel):
    """Convert Hz from mel.

    Args:
        mel:  The num or list of mel.

    Returns:
        Hz.
    """
    return 700 * (np.exp(mel / 1127) - 1.0)


def mel_filter_bank(data: np.ndarray, sample_rate: int, channel_num: int):
    """Making mel filter bank.

    Args:
        data (np.ndarray): The data using mel filter bank.
        sample_rate (int): The sample rate.
        channel_num (int): The num of channel.

    Returns:
        Mel frequency, Mel spectrogram.
    """
    mel_max = Hz_to_mel(sample_rate // 2)
    low_mel = mel_max / (channel_num + 1)
    high_mel = mel_max / (channel_num + 1) * channel_num

    # low_mel = Hz_to_mel(low_f) #周波数の下限をmelに変換
    # high_mel = Hz_to_mel(high_f) #周波数の上限をmelに変換
    mel_list = np.linspace(low_mel, high_mel, channel_num)  # メルを等間隔に分割
    mel_freq = mel_to_Hz(mel_list)  # 逆変換してメル周波数に
    h = np.round(mel_freq / sample_rate * data.shape[0]).astype("int64")  # サンプル数に反映

    mel_spec = []
    for h_index in range(len(h)):
        mfl_per = np.zeros(data.shape[0] // 2)
        if h_index == 0:  # 下限だけ例外処理
            for k in range(h[1] - h[0] * 2, h[0]):
                # 下限のときの一つ前の周波数は0
                mfl_per[k] = (k - 0) / (h[0] - 0) * 2 / (2 * h[1] - 2 * h[0])
            for k in range(h[0], h[1]):
                mfl_per[k] = (h[1] - k) / (h[1] - h[0]) * 2 / (2 * h[1] - 2 * h[0])
        elif h_index == len(h) - 1:  # 上限だけ例外処理
            for k in range(h[h_index - 1], h[h_index]):
                mfl_per[k] = (
                    (k - h[h_index - 1])
                    / (h[h_index] - h[h_index - 1])
                    * 2
                    / (2 * h[h_index] - 2 * h[h_index - 1])
                )
            for k in range(h[h_index], data.shape[0] // 2):
                # 上限のときの一つ後の周波数は最大周波数のインデックスver.
                mfl_per[k] = (
                    (data.shape[0] // 2 - 1 - k)
                    / (data.shape[0] // 2 - 1 - h[h_index])
                    * 2
                    / (2 * h[h_index] - 2 * h[h_index - 1])
                )
        else:
            for k in range(h[h_index - 1], h[h_index]):
                mfl_per[k] = (
                    (k - h[h_index - 1])
                    / (h[h_index] - h[h_index - 1])
                    * 2
                    / (h[h_index + 1] - h[h_index - 1])
                )
            for k in range(h[h_index], h[h_index + 1]):
                mfl_per[k] = (
                    (h[h_index + 1] - k)
                    / (h[h_index + 1] - h[h_index])
                    * 2
                    / (h[h_index + 1] - h[h_index - 1])
                )

        mfl_per_2 = np.concatenate([mfl_per, mfl_per[::-1]], 0)  # メルフィルタバンクを折り返す
        mel_spec.append(np.dot(data, mfl_per_2))
    return mel_freq, mel_spec


def MFCC(
    sample_data: np.ndarray, sample_rate: int, F_size: int, OverRap: float, dim=12
):
    """Calculate MFCC.

    Args:
        sample_data (np.ndarray): The data for MFCC.
        sample_rate (int): The sample rate.
        F_size (int): Frame size.
        OverRap (float): The rate of Overrap.
        dim (int, optional): The dimention of MFCC. Defaults to 12.

    Returns:
        MFCC, Mel spectrogram.
    """
    mfcc_result = []
    mel_spec_result = []
    fft, counter = stft(sample_data, F_size, OverRap)  # fftを実行
    counter = 0
    for fft_per in np.abs(fft):
        fft_log = 20 * np.log10(fft_per)  # 対数をとる(ついでにdb変換)
        mel_freq, mel_spec = mel_filter_bank(fft_log, sample_rate, 20)
        mel_spec_result.append(mel_spec)
        dct_result = scipy.fftpack.dct(mel_spec, type=2, norm="ortho", axis=-1)
        mfcc_result.append(dct_result[1 : dim + 1])
        counter += 1

    return np.array(mfcc_result), np.array(mel_spec_result)


def delta(data: np.ndarray):
    """Calculate delta parameter.

    Args:
        data (np.ndarray): The data calculating delta parameter.

    Returns:
        Delta parameter.
    """
    delta_result = []
    data = data.T
    for data_per in data:
        delta_result.append([])
        delta_result[-1].append(0)
        for i in range(1, len(data_per) - 1):
            delta_result[-1].append(data_per[i] - data_per[i - 1])  # 前の結果との差を使用
        delta_result[-1].append(0)
    return np.array(delta_result).T


def imshow(
    data: np.ndarray,
    ax,
    fig,
    index: int,
    extent: list,
    title: str,
    xlabel: str,
    ylabel: str,
):
    """For plt.imshow.

    Args:
        data (np.ndarray): The data to show.
        ax (_type_): For imshow.
        fig (_type_): The figure of imshow.
        index (int): The index of ax.
        extent (list): The scale of data.
        title (str): The title of figure.
        xlabel (str): The x label of figure.
        ylabel (str): The y label of figure.
    """
    ax_index = ax[index].imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=(extent[0], extent[1], extent[2], extent[3]),
        cmap="plasma",
    )
    fig.colorbar(ax_index, ax=ax[index], aspect=5, location="right")
    ax[index].set_title(title)
    ax[index].set_xlabel(xlabel)
    ax[index].set_ylabel(ylabel)
