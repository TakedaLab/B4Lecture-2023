import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.fftpack import dct

def load_sound_file(filename):
    """Load sound file.

    Args:
        filename (str): file name
        
    Returns:
        data (ndarray): sound data
        sr (int): sample rate
    """
    data, sr = librosa.load(filename, sr=None)
    return data, sr

def delta(data):
    """Calculate delta of data.

    Args:
        data (ndarray): Data to calculate

    Returns:
        ndarray: Delta parameter
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

def hz2mel(f):
    """Hzをmelに変換"""
    return 2595 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
    """melをhzに変換"""
    return 700 * (np.exp(m / 2595) - 1.0)

def mel_filter_bank(fs, N, num_channels):
    """メルフィルタバンクを作成"""
    # ナイキスト周波数（Hz）
    fmax = fs / 2
    # ナイキスト周波数（mel）
    melmax = hz2mel(fmax)
    # 周波数インデックスの最大数
    nmax = N // 2
    # 周波数解像度（周波数インデックス1あたりのHz幅）
    df = fs / N
    # メル尺度における各フィルタの中心周波数を求める
    dmel = melmax / (num_channels + 1)
    melcenters = np.arange(1, num_channels + 1) * dmel
    # 各フィルタの中心周波数をHzに変換
    fcenters = mel2hz(melcenters)
    # 各フィルタの中心周波数を周波数インデックスに変換
    indexcenter = np.round(fcenters / df)
    # 各フィルタの開始位置のインデックス
    indexstart = np.hstack(([0], indexcenter[0 : num_channels - 1]))
    # 各フィルタの終了位置のインデックス
    indexstop = np.hstack((indexcenter[1 : num_channels], [nmax]))
    filterbank = np.zeros((num_channels, nmax))
    for c in range(0, num_channels):
        # 三角フィルタの左の直線の傾きから点を求める
        increment= 1.0 / (indexcenter[c] - indexstart[c])
        for i in range(int(indexstart[c]), int(indexcenter[c])):
            filterbank[c, i] = (i - indexstart[c]) * increment
        # 三角フィルタの右の直線の傾きから点を求める
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in range(int(indexcenter[c]), int(indexstop[c])):
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank


def calc_mfcc(data, sr, win_length=1024, hop_length=512, mfcc_dim=12):
    data_length = data.shape[0]
    window = np.hamming(win_length)

    mfcc = []
    for i in range(int((data_length - hop_length) / hop_length)):
        # データの切り取り
        tmp = data[i * hop_length: i * hop_length + win_length]
        # 窓関数を適用
        tmp = tmp * window
        # FFTの適用
        tmp = np.fft.rfft(tmp)
        # パワースペクトルの取得
        tmp = np.abs(tmp)
        tmp = tmp[:win_length//2]

        # フィルタバンク
        channels_n = 20
        filterbank = mel_filter_bank(sr, win_length, channels_n)
        # フィルタバンクの適用
        tmp = np.dot(filterbank, tmp)
        # log
        tmp = 20 * np.log10(tmp)
        # 離散コサイン変換
        tmp = dct(tmp, norm='ortho')
        # リフタの適用
        tmp = tmp[1:mfcc_dim+1]

        mfcc.append(tmp)

    mfcc = np.transpose(mfcc)
    return mfcc

def main():
    data, fs = librosa.load("sample.wav")

    win_length = 512
    hop_length = 256

    spectrogram = librosa.stft(data, win_length=win_length, hop_length=hop_length)
    spectrogram_db = 20 * np.log10(np.abs(spectrogram))

    fig = plt.figure(figsize=(12,10))
    
    ax0 = fig.add_subplot(411)
    img = librosa.display.specshow(
        spectrogram_db,
        y_axis="log",
        sr=fs,
        cmap="rainbow",
        ax=ax0
        )
    ax0.set_title("Spectrogram")
    ax0.set_ylabel("frequency [Hz]")
    fig.colorbar(
        img,
        aspect=10,
        pad=0.01,
        extend="both",
        ax=ax0,
        format="%+2.f dB"
        )

    # mfcc 表示
    mfcc_dim = 12
    ax1 = fig.add_subplot(412)
    mfcc = calc_mfcc(data, fs, win_length, hop_length, mfcc_dim)
    wav_time = data.shape[0] // fs
    extent = [0, wav_time, 0, mfcc_dim]
    img1 = ax1.imshow(
        np.flipud(mfcc),
        aspect="auto",
        extent=extent,
        cmap="rainbow"
        )
    ax1.set_title("MFCC sequence")
    ax1.set_ylabel("MFCC")
    ax1.set_yticks(range(0, 13, 2))
    fig.colorbar(
        img1,
        aspect=10,
        pad=0.01,
        extend="both",
        ax=ax1,
        format="%+2.f dB"
        )

    # Δmfcc 表示
    ax2 = fig.add_subplot(413)
    dmfcc = delta(mfcc)
    img2 = ax2.imshow(
        np.flipud(dmfcc),
        aspect="auto",
        extent=extent,
        cmap="rainbow"
        )
    ax2.set(
        title="ΔMFCC sequence",
        ylabel="ΔMFCC",
        yticks=range(0, 13, 2)
        )
    fig.colorbar(
        img2,
        aspect=10,
        pad=0.01,
        extend="both",
        ax=ax2,
        format="%+2.f dB"
        )

    # ΔΔmfcc 表示
    ax3 = fig.add_subplot(414)
    ddmfcc = delta(dmfcc)
    img3 = ax3.imshow(
        np.flipud(ddmfcc),
        aspect="auto",
        extent=extent,
        cmap="rainbow"
        )
    ax3.set(
        title="ΔΔMFCC sequence",
        xlabel="time[s]",
        ylabel="ΔΔMFCC",
        yticks=range(0, 13, 2)
        )
    fig.colorbar(img3,
                aspect=10,
                pad=0.01,
                extend="both",
                ax=ax3,
                format="%+2.f dB"
                )

    fig.tight_layout()
    fig.savefig("mfcc.png")

if __name__ == "__main__":
    main()