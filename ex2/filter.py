import numpy as np

def conv(x,h){
    # 出力 y[0] ... y[M+N-2]
    y = np.zeros(len(h) + len(x) - 1, dtype=np.float32)

    # ゼロづめによる拡張
    hzero = np.hstack([h, np.zeros(len(x) - 1)])
    xzero = np.hstack([x, np.zeros(len(h) - 1)])

    for n in range(0, len(y)):
        for k in range(0, n + 1):
            y[n] = y[n] + hzero[k] * xzero[n - k]
    return y
}


def design_hpf(fs, fc, N, window){
    #カットオフ周波数の正規化
    fc_norm = fc / (fs / 2)

    #理想的なHPFのインパルス応答の計算
    ideal_hpf = np.sinc(np.pi * (np.arange(N) - (N - 1) / 2)) - 2 * fc_norm * np.sinc(2 * fc_norm * (np.arange(N) - (N - 1) / 2))

    #窓関数を適用
    if window == 'hamming':
        win = np.hamming(N)
    elif window == 'hanning':
        win = np.hanning(N)
    elif window == 'blackman':
        win = np.blackman(N)
    else:
        raise ValueError("Unknown window type: {}".format(window))

    return ideal_hpf * win
}