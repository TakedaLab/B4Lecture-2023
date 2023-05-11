import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from mpl_toolkits.axes_grid1 import make_axes_locatable

def wavload(path):
    """"Reads audio from the path argument and
      returns data and sampling rate"""
    data, samplerate = sf.read(path)
    return data, samplerate

def STFT(data, WIDTH):
    """Receives data and window widths and then
      returns STFT results in complex numbers"""
    OVERLAP = int(WIDTH / 2)
    # Number of audio segments
    split_number = len(np.arange((WIDTH/2), data.shape[0], (WIDTH - OVERLAP)))
    # Size of Fourier transformed data with splited
    fframe_size = len(np.fft.fft(data[:WIDTH]))

    spec = np.zeros([split_number, fframe_size], dtype=complex)
    window = np.hamming(WIDTH)
    pos = 0 # Position of audio segment

    # STFT
    for i in range(split_number):
        frame = data[pos:pos + WIDTH]
        if len(frame) >= WIDTH:
            windowed = window * frame
            # Fourier transform of segmented audio
            fft_result = np.fft.fft(windowed)
            spec[i] = fft_result
            pos += OVERLAP
    return spec

def spectrogram(TOTAL_TIME, samplerate, data):
    """Show Spectrogram"""
    amp = np.abs(data[:,int(data.shape[1]/2)::-1])
    amp = np.log(amp** 2)

    plt.rcParams["image.cmap"] = "jet"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.imshow(amp.T, extent=[0, TOTAL_TIME, 0, samplerate/2]
               , aspect='auto')
    plt.colorbar()
    plt.xlim(0, TOTAL_TIME)
    plt.ylim(0, samplerate/2)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()