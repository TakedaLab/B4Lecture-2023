def conv(x,h){ 
    # 出力 y[0] ... y[M+N-2]
    y = np.zeros(len(h) + len(x) - 1, dtype=np.float32)

    # ゼロづめによる拡張
    hzero = np.hstack([h, np.zeros(len(x) - 1)])
    xzero = np.hstack([x, np.zeros(len(h) - 1)])

    for n in range(0, len(y)):
        for k in range(0, n + 1):
            y[n] = y[n] + hzero[k] * xzero[n - k]
}