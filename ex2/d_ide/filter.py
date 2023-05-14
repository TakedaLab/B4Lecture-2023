"""filter file."""

import matplotlib.pyplot as plt
import numpy as np

"""def convolve(x, h):
    # 出力 y[0] ... y[M+N-2]
    y = np.zeros(len(h) + len(x) - 1, dtype=np.float32)

    # ゼロづめによる拡張
    hzero = np.hstack([h, np.zeros(len(x) - 1)])
    xzero = np.hstack([x, np.zeros(len(h) - 1)])

    for n in range(0, len(y)):
        for k in range(0, n + 1):
            import pdb; pdb.set_trace()
            y[n] = y[n] + hzero[k] * xzero[n - k]
            print(n,k)

    return y"""


def convolve(data, filter):
    """Convolutional operation of voice data.

    Args:
        filter (np.ndarray): filter matrix
        data (np.ndarray): input matrix
    Returns:
        out (np.ndarray): output
    """
    filter_len = len(filter)
    data_len = len(data)
    out = np.zeros(data_len - filter_len + 1)
    out = np.concatenate([
        np.zeros(filter_len - 1),
        data,
        np.zeros(filter_len - 1)
    ])

    filter = filter[::-1].T

    for i in range(data_len - filter_len + 1):
        out[i] = np.dot(data[i: i + filter_len], filter)

    return out


def design_hpf(fc, fs, N, window):
    """Create hpf.

    Args:
        fc (int): cutoff frequency
        fs (int): sampling frequency
        N (int): number of tap
        window (str): window name
    Returns:
        filtered (np.ndarray): Data after applying HPF
    """
    fc_norm = fc / (fs / 2)

    # 理想的なHPFのインパルス応答の計算
    ideal_hpf = (
        2 * fc_norm * np.sinc(2 * fc_norm * (np.arange(N) - (N - 1) / 2))
    )  # HPFのインパルス応答
    ideal_hpf = -ideal_hpf
    ideal_hpf[(N - 1) // 2] += 1

    window = np.hamming(N)

    filtered = ideal_hpf * window

    return filtered


def main():
    """Amplitude and phase after applying HPF."""
    fs = 48000
    fc = 4000
    N = 101
    window = "hamming"

    filter = design_hpf(fc, fs, N, window)
    mag = np.abs(np.fft.fft(filter))
    phase = np.angle(np.fft.fft(filter))
    mag_db = 20 * np.log10(mag)
    phase_deg = phase * 180.0 / np.pi

    x = np.linspace(0, fs, len(mag_db))

    plt.plot(x, mag_db)
    plt.xlabel("filter")
    plt.ylabel("Magnitude [dB]")
    plt.xlim(0, fs // 2)
    plt.savefig("result/mag.png")
    plt.show()

    plt.plot(x, phase_deg)
    plt.xlabel("filter")
    plt.ylabel("Phase[rad]")
    plt.xlim(0, fs // 2)
    plt.savefig("result/phase.png")
    plt.show()


if __name__ == "__main__":
    main()
