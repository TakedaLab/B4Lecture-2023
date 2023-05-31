import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def loadfile(filename):
    """load sound file

    Args:
        filename (str): file name

    Returns:
        data (ndarray): sound data
        sr (int): sample rate
    """
    data, sr = sf.read(filename)
    return data, sr


def bpf(size, fc_low, fc_high, sr):
    """generate band pass filter

    Args:
        size (int): the size of filter
        fc_low (int): lower cutoff frequency
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
        left (ndarray): left arg
        right (ndarray): rignt arg

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
    parser.add_argument(
        "-s", "--size", type=int, help="the size of filter", default=100
    )
    parser.add_argument(
        "-fcl", "--fc_low", type=int, help="lower cutoff frequency", default=50
    )
    parser.add_argument(
        "-fch", "--fc_high", type=int, help="higher cutoff frequency", default=2000
    )

    args = parser.parse_args()

    filename = args.name
    data, sr = loadfile(filename)
    time = np.arange(0, len(data)) / sr

    size = args.size
    fc_low = args.fc_low
    fc_high = args.fc_high

    # create band pass filter
    filter = bpf(size, fc_low, fc_high, sr)

    # convolution of data and filter
    filtered_data = conv(data, filter)
    filtered_time = np.arange(0, len(filtered_data)) / sr

    # convert filter to frequency domain
    filter_freq = np.fft.rfft(filter)
    # take the absolute value of filter_freq to get amplitude
    amp = np.abs(filter_freq)
    # unwrap phases
    fil_phase = np.unwrap(np.angle(filter_freq))
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
    im1 = librosa.display.specshow(
        spectrogram_db, sr=sr, ax=ax3[0], x_axis="time", y_axis="log", cmap="plasma"
    )

    f_data_stft = librosa.stft(filtered_data)
    f_spectrogram, f_phase = librosa.magphase(f_data_stft)
    f_spectrogram_db = librosa.amplitude_to_db(f_spectrogram)
    im2 = librosa.display.specshow(
        f_spectrogram_db, sr=sr, ax=ax3[1], x_axis="time", y_axis="log", cmap="plasma"
    )

    ax3[0].set(
        title="Original Spectrogram",
        xlabel="Time [s]",
        ylabel="Frequency",
        xticks=(np.arange(0, time[-1], 1)),
    )
    ax3[1].set(
        title="Filtered spectrogram",
        xlabel="Time [s]",
        ylabel=None,
        xticks=(np.arange(0, time[-1], 1)),
    )
    fig3.colorbar(im1, ax=ax3[0], format="%+2.f dB")
    fig3.colorbar(im2, ax=ax3[1], format="%+2.f dB")
    fig3.savefig("spectrogram_comparison.png")

    data_istft = librosa.istft(f_data_stft)

    sf.write("re-sample.wav", data_istft, sr)

    plt.show()


if __name__ == "__main__":
    main()
