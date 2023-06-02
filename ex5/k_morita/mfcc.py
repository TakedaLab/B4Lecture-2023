import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fftpack import dct


def seg(data, sr, win_len, overlap):
    """Calculate segmented data with overlap."""
    hop = win_len - overlap
    nsteps = (len(data) - win_len) // hop
    total_frame = nsteps * hop + win_len
    total_time = total_frame / sr
    seged = [data[hop*step: hop*step + win_len] for step in range(nsteps)]
    return np.array(seged), total_frame, total_time


def mel_filter_bank(sr, n_fft, n_channels):
    """Generate mel-filter-bank."""
    fmax = sr // 2
    melmax = hz2mel(fmax)
    nmax = n_fft // 2
    df = sr / n_fft
    dmel = melmax / (n_channels + 1)
    melcenters = np.arange(1, n_channels+1) * dmel
    fcenters = mel2hz(melcenters)
    indexcenter = np.round(fcenters / df)
    indexstart = np.hstack(([0], indexcenter[0:n_channels-1]))
    indexstop = np.hstack((indexcenter[1:n_channels], [nmax]))
    filterbank = np.zeros((n_channels, nmax))

    for c in range(0, n_channels):
        increment = 1. / (indexcenter[c] - indexstart[c])
        for i in range(int(indexstart[c]), int(indexcenter[c])):
            filterbank[c, i] = (i - indexstart[c]) * increment

        decrement = 1. / (indexstop[c] - indexcenter[c])
        for i in range(int(indexcenter[c]), int(indexstop[c])):
            filterbank[c, i] = 1. - ((i - indexcenter[c]) * decrement)

    return filterbank


def delta(data):
    """Calculate delta."""
    delt = np.zeros_like(data)
    for i in range(1, data.shape[0]):
        delt[i] = data[i] - data[i-1]
    delt[0] = delt[1]
    return delt


def hz2mel(f):
    """Convert hz to mel."""
    return 2595 * np.log(f / 700. + 1.)


def mel2hz(m):
    """Convert mel to hz."""
    return 700 * (np.exp(m / 2595.) - 1.)


def mfcc_(data, sr, n_mel=20, n_mfcc=13):
    """Calculate MFCC."""
    win_len = n_fft = 1024
    overlap = 512
    n_channels = 20
    n_ceps = 12

    seged_data, total_frame, total_time = seg(data, sr, win_len, overlap)
    windowed = seged_data * np.hamming(win_len)
    spec = np.abs(np.fft.fft(windowed, n_fft))[:, n_fft//2:]
    mel_filter = mel_filter_bank(sr, n_fft, n_channels)
    mel_spec = np.dot(spec, mel_filter.T)
    ceps = dct(20 * np.log10(mel_spec), type=2, axis=1, norm='ortho')
    return ceps[:, :n_ceps][:, ::-1], spec, total_time


if __name__ == "__main__":
    file = "audio.wav"
    data, sr = sf.read(file)

    mfcc, spec, total_time = mfcc_(data, sr)
    delt = delta(mfcc)
    deltdelt = delta(delt)

    # draw
    fig = plt.figure(figsize=(8, 9))

    ax = fig.add_subplot(411)
    ax.set_title("Spectrogram")
    im = ax.imshow(
        (20 * np.log10(spec)).T,
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[0, total_time, 0, sr // 2])
    ax.set_ylabel("f [Hz]")
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(412)
    ax.set_title("MFCC")
    im = ax.imshow(
        mfcc.T,
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[0, total_time, 0, sr // 2])
    ax.set_ylabel("MFCC")
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(413)
    ax.set_title("$\Delta$MFCC")
    im = ax.imshow(
        delt.T,
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[0, total_time, 0, sr // 2])
    ax.set_ylabel("$\Delta$MFCC")
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(414)
    ax.set_title("$\Delta\Delta$MFCC")
    im = ax.imshow(
        deltdelt.T,
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[0, total_time, 0, sr // 2])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$\Delta\Delta$MFCC")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()
