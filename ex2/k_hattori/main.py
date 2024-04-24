"""Reads and filters audio."""
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import myfunc as mf


def convolution(x, h):
    """Receives arrays x and h, and returns result fo the convolution y."""
    y = np.zeros(len(x) + len(h) - 1)
    width = len(h)
    for i in range(len(x)):
        y[i : i + width] += x[i] * h

    return y


def LPF_window(length, cutoff, size, samplerate):
    """
    Create a filter and Returns FIR impulse respoonse.

    Args:
        length (int):   Order of the filter. it must be odd number.
        cutoff (float): Cutoff frequency of the filter.
        size (int):     Size of filter.
        samplerate(int):samplerate of filtered signal.

    Returns:
        ndarray[int]:   Impulse response of the FIR filter
    """
    N = int((length - 1) / 2)  # tap number
    nyq_sample = int(size / 2)  # nyquist frequency [sample]
    window = np.kaiser(length, 6)  # kaiser window
    freq = np.linspace(0, samplerate, size)
    F_filter = np.zeros(size)

    # Filter Design
    F_filter[freq < cutoff] = 1
    im_response = np.fft.ifft(F_filter)
    im_response[N:-N] = 0  # Censoring of impulse response
    im_response = np.fft.ifftshift(im_response)
    # multiplying window function
    windowed = window * im_response[nyq_sample - N : nyq_sample + N + 1]
    im_response[nyq_sample - N : nyq_sample + N + 1] = windowed
    im_response = np.fft.fftshift(im_response)
    F_filter = np.fft.fft(im_response)
    h = np.roll(im_response, N)  # FIR filter's impulse response
    filter = np.fft.fft(h)

    # Show Filter Properties
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.plot(window)
    plt.show()
    plt.subplot(211)
    plt.title("FIR properties")
    plt.plot(freq, 20 * np.log(np.abs(filter)))
    plt.ylabel("Amplitude [dB]")
    plt.subplot(212)
    plt.plot(freq, np.arctan2(np.imag(filter), np.real(filter)))
    plt.ylabel("Phase [rad]")
    plt.xlabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()

    return np.real(h[:length])


def main():
    """
    Read an audio file and create a filter to save filtered sound file.

    Returns:
        None
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    path = r"ex2\k_hattori\ONSEI.wav"
    data, samplerate = mf.wavload(path)  # Reads wavfile
    TOTAL_TIME = len(data) / samplerate
    time = np.linspace(0, TOTAL_TIME, len(data))

    # FIR filter's impulse response
    h = LPF_window(81, 10000, len(data), samplerate)
    plt.plot(h)
    plt.title("FIR impulse response")
    plt.xlabel("Time [sample]")
    plt.ylabel("Magnitude")
    plt.show()

    # convolute data and impulse response
    filter_data = convolution(data, h)
    FILTER_TOTAL_TIME = len(filter_data) / samplerate
    filter_time = np.linspace(0, FILTER_TOTAL_TIME, len(filter_data))

    # Show Original and Filtered Sound
    plt.subplot(211)
    plt.title("Original Sound")
    plt.plot(time, data)
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.subplot(212)
    plt.title("Filtered Sound")
    plt.plot(filter_time, filter_data)
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.show()

    # Show spectrogram
    spec = mf.STFT(data, 1024)
    spec_f = mf.STFT(filter_data, 1024)
    mf.spectrogram_double(TOTAL_TIME, FILTER_TOTAL_TIME, samplerate, spec, spec_f)

    # Save Filtered Soundfile
    sf.write(file="Fiter_ONSEI.wav", data=filter_data, samplerate=samplerate)


if __name__ == "__main__":
    main()
    exit(1)
