"""Main."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read

from modules import envelope as ev
from modules import f0_estimate as f0
from modules import spectrogram as sp


if __name__ == "__main__":
    file = "audio.wav"
    # file = "doremi.mp3"
    sr, data = read(file)
    # data, sr = librosa.load(file, sr=None)
    win_len = 2048
    ol = 0.75

    # calculate stectrogram and estimate f0
    stft, total_time, total_frame = sp.stft(data, sr, win_len=win_len, ol=ol)
    mag, _ = sp.magphase(stft)
    mag_db = sp.mag_to_db(mag)
    f0_auto = f0.f0_autocorrelation(data, sr, win_len=win_len, ol=ol)
    f0_cept = f0.f0_cepstrum(data, sr, win_len=win_len, ol=ol)

    # calculate spectrum-envelope
    left = int(sr * 0.5)  # 500ms point
    segment = data[left: left + win_len]
    envelope_cepstrum, log_spectrum, freq = \
        ev.spectrum_envelope_cepstrum(segment, sr)
    envelope_lpc, freq_lpc = ev.spectrum_envelope_lpc(segment, sr, 32)

    # draw graph
    fig = plt.figure(figsize=(12, 8))

    # f0 by autocorrelation
    ax1 = fig.add_subplot(221)
    ax1.set_title("f0(autocorrelation)")
    im = ax1.imshow(
        mag_db,
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[0, total_time, 0, sr // 2])
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Frequency [Hz]")
    fig.colorbar(im, ax=ax1)
    ax1.plot(np.linspace(0, total_time, len(f0_auto)), f0_auto,
             label="f0", color="black")
    ax1.legend()

    # f0 by cepstrum
    ax2 = fig.add_subplot(222)
    ax2.set_title("f0(cepstrum)")
    im = ax2.imshow(
        mag_db,
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[0, total_time, 0, sr // 2])
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Frequency [Hz]")
    fig.colorbar(im, ax=ax2)
    ax2.plot(np.linspace(0, total_time, len(f0_auto)), f0_cept,
             label="f0", color="black")
    ax2.legend()

    # envelope by cepstrum
    ax3 = fig.add_subplot(223)
    ax3.set_title("envelope based on cepstrum")
    ax3.set_xlabel("Frequency [Hz]")
    ax3.set_ylabel("Amplitude [dB]")
    ax3.plot(freq, log_spectrum,
             label="log amplitude")
    ax3.plot(freq, envelope_cepstrum,
             label="spectrum envelope")
    ax3.set_xlim(0, sr//2)
    ax3.legend()

    # envelope by lpc
    ax4 = fig.add_subplot(224)
    ax4.set_title("envelope based on lpc")
    ax4.set_xlabel("Frequency [Hz]")
    ax4.set_ylabel("Amplitude [dB]")
    ax4.plot(freq, log_spectrum,
             label="log amplitude")
    ax4.plot(freq_lpc, envelope_lpc,
             label="spectrum envelope")
    ax4.set_xlim(0, sr//2)
    ax4.legend()

    plt.tight_layout()
    plt.show()
