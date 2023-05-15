import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def loadfile(filename):
    data, sr = sf.read(filename)
    return data, sr


def bpf(size, fc_low, fc_high, sr):
    """generate band pass filter

    Args:
        size (int): the size of filter
        fc_low (int): ower cutoff frequency
        fc_high (int): higher cutoff frequency
        sr (int): sample rate

    Returns:
        ndarray: band pass filter
    """
    # Conversion to angular frequency
    afc_low = 2 * np.pi * fc_low / sr
    afc_high = 2 * np.pi * fc_high / sr

    filter = []
    for i in range(-size // 2, size // 2 + 1):
        filter.append(
            (
                afc_high * np.sinc(afc_high * i / np.pi)
                - afc_low * np.sinc(afc_low * i / np.pi)
            )
            / np.pi
        )

    # use a hamming window
    window = np.hamming(size + 1)

    return np.array(filter) * window


def conv(left, right):
    """convolution

    Args:
        l (ndarray): left arg
        r (ndarray): rignt arg

    Returns:
        ndarray: convolution result
    """
    result = np.zeros(len(left) + len(right))

    for i in range(len(left)):
        result[i : i + len(right)] += left[i] * right

    return result


def main():
    parser = argparse.ArgumentParser("Design digital filters to filter audio")

    parser.add_argument("name", help="file name")
    parser.add_argument("-s", "--size", help="the size of filter", default=100)
    parser.add_argument("-fcl", "--fc_low", help="lower cutoff frequency", default=50)
    parser.add_argument(
        "-fch", "--fc_high", help="higher cutoff frequency", default=2000
    )

    args = parser.parse_args()

    filename = args.name
    data, sr = loadfile(filename)
    time = np.arange(0, len(data)) / sr

    size = args.size
    fc_low = args.fc_low
    fc_high = args.fc_high

    filter = bpf(size, fc_low, fc_high, sr)

    filtered_data = conv(data, filter)
    filtered_time = np.arange(0, len(filtered_data)) / sr

    filter_freq = np.fft.rfft(filter)
    amp = np.abs(filter_freq)[: size // 2 + 1]
    fil_phase = np.unwrap(np.angle(filter_freq))[: size // 2 + 1] * 180 / np.pi
    frequency = np.linspace(0, sr / 2, len(fil_phase)) / 1000

    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 3), tight_layout=True)
    ax1.plot(time, data, label="original")
    ax1.plot(filtered_time, filtered_data, label="filtered")
    ax1.set(
        title="Signal comparison",
        xlabel="Time [s]",
        ylabel="Magnitude",
        xlim=(0, time[-1]),
        ylim=(-1, 1),
    )
    ax1.legend()
    fig1.savefig("wave_comparison.png")

    fig2, ax2 = plt.subplots(2, 1, figsize=(6, 6), tight_layout=True, sharex=True)
    ax2[0].plot(frequency, amp)
    ax2[0].set(
        title="Frequency response of BPF",
        ylabel="Amplitude [dB]",
    )
    ax2[1].plot(frequency, fil_phase)
    ax2[1].set(
        xlabel="Frequency [kHz]",
        ylabel="Phase [rad]",
    )
    fig2.savefig("frequency_response.png")

    fig3, ax3 = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True, sharey=True)

    data_stft = librosa.stft(data)
    spectrogram, phase = librosa.magphase(data_stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    librosa.display.specshow(
        spectrogram_db, sr=sr, ax=ax3[0], x_axis="time", y_axis="log"
    )

    f_data_stft = librosa.stft(filtered_data)
    f_spectrogram, f_phase = librosa.magphase(f_data_stft)
    f_spectrogram_db = librosa.amplitude_to_db(f_spectrogram)
    librosa.display.specshow(
        f_spectrogram_db, sr=sr, ax=ax3[1], x_axis="time", y_axis="log"
    )

    ax3[0].set(title="Original Spectrogram", xlabel="Time [s]", ylabel="Frequency")
    ax3[1].set(title="Filtered spectrogram", xlabel="Time [s]", ylabel=None)
    fig3.savefig("spectrogram_comparison.png")

    plt.show()


if __name__ == "__main__":
    main()
