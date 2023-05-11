import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def bandpass_filter(fs, f_low, f_high, taps=101):
    nyq = 0.5 * fs
    f = np.zeros(taps)
    f[int(taps * f_low / nyq):int(taps * f_high / nyq)] = 1
    h = np.sinc(2 * f / fs)
    h *= np.blackman(taps)
    h /= np.sum(h)
    return h

def conv(input, filter):
    filter_len = len(filter)
    zero_pad = np.zeros(filter_len-1)
    input_pad = np.concatenate([zero_pad, input, zero_pad])
    output_len = len(input_pad) - len(zero_pad)
    output = np.zeros(output_len)
    filter = filter
    for i in range(len(input)-(filter_len-1)):
        start, end = i, i+filter_len
        output[i] = np.dot(filter[start:end], input_pad)


def main():
    fs = 1000  # サンプリング周波数
    f_low = 20  # 帯域通過フィルタの下限周波数
    f_high = 200  # 帯域通過フィルタの上限周波数
    h = bandpass_filter(fs, f_low, f_high, taps=101)
    b = h
    a = 1
    print("b = ", b)
    print("a = ", a)
    plt.plot(b)
    plt.show()


if __name__ == '__main__':
    main()