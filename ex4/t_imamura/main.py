"""Ex4 fundamental frequency and spectral envelope."""

import copy

import matplotlib.pyplot as plt
import numpy as np
import scipy
import soundfile


def spectrogram(data: np.ndarray, sample_rate: int, ax, fig, indx_ax=None, F_size=1024):
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
    if indx_ax:
        ax_spec = ax[indx_ax].pcolormesh(
            t, f, 20 * np.log10(spec), vmax=1e-6, cmap="CMRmap"
        )  # dbに変換してスペクトログラムを表示
        ax[indx_ax].set_xlabel("Time [sec]")
        ax[indx_ax].set_ylabel("Fequency [Hz]")
        fig.colorbar(ax_spec, ax=ax[indx_ax], aspect=5, location="right")
    else:
        ax_spec = ax.pcolormesh(
            t, f, 20 * np.log10(spec), vmax=1e-6, cmap="CMRmap"
        )  # dbに変換してスペクトログラムを表示
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
        # quefrency_ff.append((sample_rate//2) / np.argmax(high_quef_per[:F_size//2]))
        # print(np.argmax(high_quef[:F_size//2]))
        fft_start += dist_of_Frame
        # counter += 1
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


def main():
    """Do main action."""
    fig, ax = plt.subplots(1, 1, layout="constrained", sharex=True)  # スペクトログラムと基本周波数表示
    ax.set_title("Spectrogram and fundamental frequency.")
    fig1, ax1 = plt.subplots(1, 1, layout="constrained")  # 基本周波数表示
    ax1.set_title("Fundamental frequency.")
    ax1.set_xlabel("Time [sec]")
    ax1.set_ylabel("Fequency [Hz]")
    fig2, ax2 = plt.subplots(1, 1, layout="constrained")  # スペクトル包絡を表示
    ax2.set_title("Spectral envelope.")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Amplitude [db]")

    # 音源の読み込み
    sample_path = "ex4_sample2.wav"  # 音源へのパスを指定
    sample_data, sample_rate = soundfile.read(sample_path)
    N = len(sample_data)  # サンプル数を出しておく

    # 自己相関法で基本周波数を推定
    ac = []  # 結果の格納
    F_size = 2048  # 切り出すフレーム幅
    ac_start = 0  # フレームの開始位置
    counter = 0
    while ac_start + F_size <= N:  # 次のフレーム幅の末端がサンプル数を超えるまでfft
        Frame = sample_data[ac_start : ac_start + F_size]  # 音源データからフレームを切り出す
        ac_per = auto_correlation(Frame)  # 自己相関関数を計算
        peak = []
        for i in range(ac_per.shape[0] - 2):  # 最初と最後はピークから除く
            if (
                ac_per[i] < ac_per[i + 1] and ac_per[i + 1] > ac_per[i + 2]
            ):  # ピークの候補を持ってくる
                peak.append([i + 1, ac_per[i + 1]])
        peak = np.array(peak)
        peak_value = peak[:, 1]
        peak_value_sorted = np.sort(peak_value)[::-1]  # 大きい順にソート
        second_peak_value = peak_value_sorted[0]  # 先頭を取ってくる
        second_peak_index = np.where(ac_per == second_peak_value)[0]  # 最大値のインデックスを取る
        ac.append(sample_rate / second_peak_index[0])
        ac_start += F_size
        counter += 1
    ax.set_ylim(0, 5000)
    spectrogram(sample_data, sample_rate, ax, fig, F_size=F_size)
    ax.plot(
        np.arange(0, counter) * F_size / sample_rate,
        ac,
        color="black",
        label="By autocorrelation method",
    )
    ax1.plot(
        np.arange(0, counter) * F_size / sample_rate,
        ac,
        color="black",
        label="By autocorrelation method",
    )

    # ケフレンシー分析を実行
    F_size = 2048
    OverRap = 0.0  # オーバーラップ率0%
    low_quefrency, high_quefrency, count_fft = Quefrency(
        sample_data, sample_rate, F_size, OverRap
    )

    # ケフレンシー法で基本周波数を推定
    f0_quef = []
    for high_quef_per in high_quefrency:
        f0_quef.append((sample_rate) / np.argmax(high_quef_per[: F_size // 2]))
    ax.plot(
        np.arange(0, count_fft) * (F_size) / sample_rate,
        f0_quef,
        color="blue",
        label="By cepstrum",
    )
    ax1.plot(
        np.arange(0, count_fft) * (F_size) / sample_rate,
        f0_quef,
        color="blue",
        label="By cepstrum",
    )

    # ケフレンシー法でスペクトル包絡を推定
    extract_time = 1.0  # スペクトル包絡を出す部分の時間
    extract_frame = int(extract_time * sample_rate / F_size)  # 時間をフレーム数に変換
    stft_result, _ = stft(sample_data, F_size=2048, OverRap=0.0)  # スペクトログラム出力用にstft
    stft_db = 20 * np.log10(np.abs(stft_result[extract_frame]))  # stftのdbを出す
    env_quef = np.real(np.fft.fft(low_quefrency[extract_frame]))  # 低ケフレンシー成分をfftして実部をとる
    ax2.plot(
        np.linspace(0, sample_rate // 2, F_size // 2),
        stft_db[: F_size // 2],
        color="red",
        label="Spectrogram",
    )
    ax2.plot(
        np.linspace(0, sample_rate // 2, F_size // 2),
        env_quef[: F_size // 2],
        color="blue",
        label="By cepstrum",
    )

    # LPC法でスペクトル包絡を推定
    lpc_parameter = 32  # LPC係数
    ac = auto_correlation(
        sample_data[extract_frame * F_size : (extract_frame + 1) * F_size]
        * np.hamming(F_size)
    )  # 対象フレームに対し自己相関関数を計算
    a, e = LevinsonDurbin(ac, lpc_parameter)
    w, h = scipy.signal.freqz(np.sqrt(e), a, F_size, "whole")
    ax2.plot(
        np.linspace(0, sample_rate // 2, F_size // 2),
        20 * np.log10(np.abs(h))[: F_size // 2],
        color="orange",
        label="By LPC",
    )

    ax.legend()
    ax1.legend()
    ax2.legend()
    plt.show()
    fig.savefig("Spectrogram_f0.png")
    fig1.savefig("f0.png")
    fig2.savefig("Env.png")
    plt.clf()
    plt.close()


if "__main__" == __name__:
    main()
