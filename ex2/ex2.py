import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd

# 課題1: 畳み込み演算を実装
def my_conv(x, h):
    y = np.zeros(len(x) + len(h) - 1)
    for n in range(len(y)):
        for k in range(len(h)):
            if n-k < 0 or n-k >= len(x):
                continue
            y[n] += h[k] * x[n-k]
    return y

# 課題2: ディジタルフィルタの設計
fs = 44100  # サンプリング周波数
fc = 4000   # カットオフ周波数
M = 511     # フィルタの長さ（奇数）
h = signal.firwin(M, fc/(fs/2), window='hamming')

# 時間領域でのフィルタの定義を表示
plt.plot(h)
plt.title('Impulse Response')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# 周波数特性・振幅特性の表示
w, H = signal.freqz(h)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(w/np.pi*fs/2, 20*np.log10(np.abs(H)), 'b')
ax2.plot(w/np.pi*fs/2, np.unwrap(np.angle(H))*180/np.pi, 'r')
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('Magnitude [dB]', color='b')
ax2.set_ylabel('Phase [deg]', color='r')
plt.title('Frequency Response')
plt.show()

# 課題3: スペクトログラムを表示してフィルタリングの影響を確認
# 元の音声の読み込み
data, fs = sf.read('./B4Lecture-2023/ex2/sample.wav')

fig, (ax1, ax2) = plt.subplots(1, 2)
# スペクトログラムの表示（フィルタリング前）
ax1.specgram(data, Fs=fs)
ax1.set_title('Before Filtering')
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Frequency [Hz]')

# 音声信号にフィルタをかける
filtered = my_conv(data, h)

# スペクトログラムの表示（フィルタリング後）
ax2.specgram(filtered, Fs=fs)
ax2.set_title('After Filtering')
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Frequency [Hz]')

plt.tight_layout()
plt.show()

# 課題4: フィルタリングされた音声を再生
sf.write('./B4Lecture-2023/ex2/filtered.wav', filtered, fs)  # フィルタリングされた音声の書き出し
filtered, fs = sf.read('./B4Lecture-2023/ex2/filtered.wav')  # フィルタリングされた音声の読み込み
sd.play(filtered, fs)  # 再生