"""Ex5 k-means clustering and MFCC."""

import argparse
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy

import func


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
                mfl_per[k] = (k - 0) / (h[0] - 0) / (2 * h[1] - 2 * h[0])
            for k in range(h[0], h[1]):
                mfl_per[k] = (h[1] - k) / (h[1] - h[0]) / (2 * h[1] - 2 * h[0])
        elif h_index == len(h) - 1:  # 上限だけ例外処理
            for k in range(h[h_index - 1], h[h_index]):
                mfl_per[k] = (
                    (k - h[h_index - 1])
                    / (h[h_index] - h[h_index - 1])
                    / (2 * h[h_index] - 2 * h[h_index - 1])
                )
            for k in range(h[h_index], data.shape[0] // 2):
                # 上限のときの一つ後の周波数は最大周波数のインデックスver.
                mfl_per[k] = (
                    (data.shape[0] // 2 - 1 - k)
                    / (data.shape[0] // 2 - 1 - h[h_index])
                    / (2 * h[h_index] - 2 * h[h_index - 1])
                )
        else:
            for k in range(h[h_index - 1], h[h_index]):
                mfl_per[k] = (
                    (k - h[h_index - 1])
                    / (h[h_index] - h[h_index - 1])
                    / (h[h_index + 1] - h[h_index - 1])
                )
            for k in range(h[h_index], h[h_index + 1]):
                mfl_per[k] = (
                    (h[h_index + 1] - k)
                    / (h[h_index + 1] - h[h_index])
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
    fft, counter = func.stft(sample_data, F_size, OverRap)  # fftを実行
    counter = 0
    for fft_per in np.abs(fft):
        fft_log = 20 * np.log10(fft_per)  # 対数をとる(ついでにdb変換)
        mel_freq, mel_spec = mel_filter_bank(fft_log, sample_rate, 20)
        if counter == 30:
            fig0, ax0 = plt.subplots(1, 1)
            ax0.plot(
                np.linspace(0, sample_rate // 2, F_size // 2),
                fft_log[: F_size // 2],
                label="Spectrogram",
            )
            ax0.plot(mel_freq, mel_spec, label="Mel spectrogram")
            ax0.set_title("Mel spectrogram")
            ax0.set_xlabel("Frequency [Hz]")
            ax0.set_ylabel("Amplitude [db]")
            ax0.legend()
            fig0.savefig("mel_spec.png")
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


def main():
    """Do main action."""
    parser = argparse.ArgumentParser(
        prog="main.py",  # プログラム名
        usage="B4 Lecture Ex5.",  # プログラムの利用方法
        description="K-means or LBG clustering and MFCC.",  # 引数のヘルプの前に表示
        epilog="end",  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )
    parser.add_argument(
        "-k", "--kmeans", action="store_true", help="For activating k-means clustering."
    )
    parser.add_argument(
        "-l", "--lbg", action="store_true", help="For activating lbg clustering."
    )
    parser.add_argument(
        "-m", "--mfcc", action="store_true", help="For activating mfcc."
    )
    args = parser.parse_args()

    if args.kmeans or args.lbg:
        fig0, ax0 = plt.subplots(1, 1)
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        fig1, ax1 = plt.subplots(1, 1)
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        fig2, ax2 = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        # データの読み込み
        data1 = func.read_csv("data1.csv")
        data2 = func.read_csv("data2.csv")
        data3 = func.read_csv("data3.csv")
        if args.kmeans:
            # k-means法実行
            _, _, error_data1 = k_means(data1, 4, 10000, "2d", ax0)
            ax0.set_title(f"data1 k-means error = {error_data1:.1f}")
            _, _, error_data2 = k_means(data2, 2, 10000, "2d", ax1)
            ax1.set_title(f"data2 k-means error = {error_data2:.1f}")
            _, _, error_data3 = k_means(data3, 4, 10000, "3d", ax2)
            ax2.set_title(f"data3 k-means error = {error_data3:.1f}")

            fig0.savefig("data1_kmeans_pre.png")
            fig1.savefig("data2_kmeans_pre.png")
            fig2.savefig("data3_kmeans_pre.png")
        elif args.lbg:
            # LBG法実行
            _, error_data1 = LBG(data1, 4, 10000, "2d", ax0, delta=0.005)
            ax0.set_title(f"data1 LBG error = {error_data1:.1f}")
            _, error_data2 = LBG(data2, 4, 10000, "2d", ax1)
            ax1.set_title(f"data2 LBG error = {error_data2:.1f}")
            _, error_data3 = LBG(data3, 4, 10000, "3d", ax2)
            ax2.set_title(f"data3 LBG error = {error_data3:.1f}")

            fig0.savefig("data1_lbg_pre.png")
            fig1.savefig("data2_lbg_pre.png")
            fig2.savefig("data3_lbg_pre.png")

    """
    data1 = func.read_csv("data1.csv")
    error_all_kmeans = []
    error_all_lbg = []
    for i in range(10):
        error_all_kmeans.append(k_means(data1, 4, 10000, "2d")[2])
        error_all_lbg.append(LBG(data1, 4, 10000, "2d", delta=0.005)[1])
    error_all_kmeans = np.array(error_all_kmeans)
    error_all_lbg = np.array(error_all_lbg)
    print(f"kmeans 10 times error: max = {np.max(error_all_kmeans):.1f}, min = {np.min(error_all_kmeans):.1f}, ave = {np.mean(error_all_kmeans):.1f}")
    print(f"LBG 10 times error:    max = {np.max(error_all_lbg):.1f}, min = {np.min(error_all_lbg):.1f}, ave = {np.mean(error_all_lbg):.1f}")
    """

    if args.mfcc:
        fig3, ax3 = plt.subplots(5, 1, layout="constrained", figsize=(15, 10))
        sample_data, sample_rate, N = func.load_sound_file(
            "sound_file/akasatanahamayarawa.wav"
        )  # 音源の読み込み
        func.spectrogram(sample_data, sample_rate, ax3, fig3, 0)  # スペクトログラムの表示
        mfcc_result, mel_spec_result = MFCC(
            sample_data, sample_rate, 2048, 0.0
        )  # MFCC実行
        imshow(
            mel_spec_result.T,
            ax3,
            fig3,
            1,
            [0, N / sample_rate, 0, sample_rate // 2],
            "Mel spectrogram",
            "Time [s]",
            "Frequency [Hz]",
        )
        imshow(
            mfcc_result.T,
            ax3,
            fig3,
            2,
            [0, N / sample_rate, 0, 12],
            "MFCC",
            "Time [s]",
            "MFCC",
        )
        delta_mfcc = delta(mfcc_result)
        imshow(
            delta_mfcc.T,
            ax3,
            fig3,
            3,
            [0, N / sample_rate, 0, 12],
            "ΔMFCC",
            "Time [s]",
            "ΔMFCC",
        )
        deltadelta_mfcc = delta(delta_mfcc)
        imshow(
            deltadelta_mfcc.T,
            ax3,
            fig3,
            4,
            [0, N / sample_rate, 0, 12],
            "ΔΔMFCC",
            "Time [s]",
            "ΔΔMFCC",
        )
        fig3.savefig("mfcc.png")

    plt.show()
    plt.clf()
    plt.close()


if "__main__" == __name__:
    main()
