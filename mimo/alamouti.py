import numpy as np


class AlamoutiEncoder:
    def __init__(self):
        pass

    def encode(self, symbols):
        n = len(symbols)
        if n % 2 != 0:
            symbols = np.pad(symbols, (0, 1))
            n = len(symbols)
        tx1 = np.zeros(n, dtype=complex)
        tx2 = np.zeros(n, dtype=complex)
        tx1[0::2] = symbols[0::2]
        tx1[1::2] = -np.conj(symbols[1::2])
        tx2[0::2] = symbols[1::2]
        tx2[1::2] = np.conj(symbols[0::2])
        return tx1, tx2


class AlamoutiDecoder:
    def __init__(self):
        pass

    def decode(self, rx1, rx2, h1, h2):
        n = len(rx1)
        symbols = np.zeros(n, dtype=complex)
        for i in range(0, n, 2):
            r0 = rx1[i]
            r1 = rx1[i + 1] if i + 1 < n else 0
            h0 = h1 if np.isscalar(h1) else h1[i]
            h1_val = h2 if np.isscalar(h2) else h2[i]
            s0 = (np.conj(h0) * r0 + h1_val * np.conj(r1)) / (
                np.abs(h0) ** 2 + np.abs(h1_val) ** 2 + 1e-12
            )
            s1 = (np.conj(h1_val) * r0 - h0 * np.conj(r1)) / (
                np.abs(h0) ** 2 + np.abs(h1_val) ** 2 + 1e-12
            )
            symbols[i] = s0
            if i + 1 < n:
                symbols[i + 1] = s1
        return symbols
