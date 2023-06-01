import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def seg(data, sr, win_len, overlap):
    hop = win_len - overlap
    nsteps = (len(data) - win_len) // hop
    total_frame = nsteps * hop + win_len
    total_time = total_frame / sr
    seged = [data[hop*step: hop*step + win_len] for step in range(nsteps)]
    return np.array(seged), total_frame, total_time


def mel_frequency_array(n_fft, n_mel, sr):
    f0 = 700
    m0 = 1127
    high_limit = sr
    high_limit_mel = m0 * np.log10(high_limit / f0 + 1)
    mel_array = np.linspace(0, high_limit_mel, n_mel + 2)
    freq_array = f0 * (10 ** (mel_array / m0) - 1)
    mel_freq_array = ((n_fft+1) * freq_array) // sr
    return mel_freq_array


def mel_filter_bank(n_fft, n_mel, sr):
    h = mel_frequency_array(n_fft, n_mel, sr)
    mel_fil_bank = np.zeros((n_mel, n_fft // 2))
    for m in range(1, n_mel + 1):
        left = int(h[m - 1])
        mid = int(h[m])
        right = int(h[m + 1])
        for k in range(left, mid):
            if k >= mel_fil_bank.shape[1]-1:
                continue
            mel_fil_bank[m - 1, k] = (k - left) / (mid - left)
        for k in range(mid, right):
            if k >= mel_fil_bank.shape[1]-1:
                continue
            mel_fil_bank[m - 1, k] = (right - k) / (right - mid)
    return mel_fil_bank


def dct(n_fil, n_mfcc):
    n = np.arange(n_fil)
    dct_filter = np.zeros((n_mfcc, n_fil))
    for i in range(n_mfcc):
        dct_filter[i, :] = np.cos(i * n * np.pi / n_fil)
    return dct_filter


def mfcc(data, sr, n_mel=20, n_mfcc=13):
    win_len = 1024
    overlap = 512

    # step 1
    seged_data, total_frame, total_time = seg(data, sr, win_len, overlap)

    # step 2
    seged_data *= np.hamming(win_len)
    fft = np.fft.fft(seged_data)
    spec = np.abs(fft)

    # step 3
    log_spec = 20 * np.log10(spec)
    log_spec = log_spec[:, log_spec.shape[1] // 2:]

    # step 4
    n_fft = fft.shape[1]
    mel_filter = mel_filter_bank(n_fft, n_mel, sr)
    mel_spec = np.dot(log_spec, mel_filter.T)

    # step 5
    mfcc_filter = dct(n_mel, n_mfcc)
    mfcc = np.dot(mel_spec, mfcc_filter.T)
    log_mfcc = 20 * np.log10(np.abs(mfcc) + np.finfo(float).eps)

    # draw
    fig = plt.figure()

    ax = fig.add_subplot(211)
    ax.set_title("Spectrogram")
    im = ax.imshow(
        log_spec.T,
        cmap=plt.cm.jet,
        aspect="auto",
        extent=[0, total_time, 0, sr // 2])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(212)
    ax.set_title("MFCC")
    im = ax.imshow(
        -1 * log_mfcc.T,
        cmap=plt.cm.jet,
        aspect="auto",
        vmax=0,
        vmin=-80,
        extent=[0, total_time, 0, sr // 2])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file = "audio.wav"
    data, sr = sf.read(file)
    mfcc(data, sr)
