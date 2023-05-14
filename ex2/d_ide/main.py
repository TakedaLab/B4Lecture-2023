"""main file."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftpack
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import soundfile as sf
import filter


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(
        description="Generate spectrogram and inverse transform"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="input wav file"
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=1024,
        help="number of FFT points"
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="number of samples between successive STFT columns",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="hamming",
        help="window function type"
    )
    parser.add_argument(
        "--sampling",
        type=int,
        default=48000,
        help="sampling frequency"
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=4000,
        help="cutoff frequency"
    )
    parser.add_argument(
        "--tap",
        type=int,
        default=101,
        help="number of filter taps"
    )
    return parser.parse_args()


def main():
    """Calculate the spectrogram."""
    args = parse_args()

    # 音声ファイルを読み込む
    rate, data = wavfile.read(args.input_file)
    data = np.array(data, dtype=float)

    # 波形をプロットする
    time = np.arange(0, len(data)) / rate
    plt.plot(time, data)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.savefig("result/input-wave.png")
    plt.show()

    # STFTのそれぞれのパラメータ
    nfft = args.nfft
    hop_length = args.hop_length
    window = args.window
    window_func = signal.get_window(window, nfft)
    fs = args.sampling
    fc = args.cutoff
    N = args.tap

    hpf = filter.design_hpf(fc, fs, N, window)

    filtered = filter.convolve(data, hpf)

    time = np.arange(0, len(filtered)) / rate
    plt.plot(time, filtered)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.savefig("result/filtered-wave.png")
    plt.show()

    # スペクトログラムの計算
    spectrogram_f = np.zeros(
        (1 + nfft // 2, (len(filtered) - nfft) // hop_length + 1),
        dtype=np.complex128
    )
    for i in range(spectrogram_f.shape[1]):
        start = i * hop_length
        finish = i * hop_length + nfft
        segment = filtered[start: finish] * window_func
        spectrum = fftpack.fft(segment, n=nfft, axis=0)[: 1 + nfft // 2]
        spectrogram_f[:, i] = spectrum

    # スペクトログラムの描画
    plt.figure()
    plt.imshow(
        20 * np.log10(np.abs(spectrogram_f)),
        origin="lower",
        aspect="auto",
        cmap="jet"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar()
    plt.title("Spectrogram_f")
    plt.savefig("result/filtered-spectrogram.png")
    plt.show()

    # 変換後の音声ファイルの書き出し
    sf.write("soundfile.wav", data=filtered, samplerate=fs)
    filtered_int = np.int16(filtered * 32767)
    wavfile.write("wavfile.wav", fs, filtered_int)


if __name__ == "__main__":
    main()
