import numpy as np


class Demodulator:
    def __init__(self, scheme="QPSK"):
        self.scheme = scheme.upper()
        self._setup()

    def _setup(self):
        from transmitter.modulation import Modulator
        mod = Modulator(self.scheme)
        self.constellation = mod.get_constellation()
        self.order = mod.order
        self.bits_per_symbol = mod.bits_per_symbol

    def demodulate_soft(self, symbols):
        bits = np.zeros(len(symbols) * self.bits_per_symbol, dtype=float)
        for i, sym in enumerate(symbols):
            dists = np.abs(sym - self.constellation) ** 2
            for b in range(self.bits_per_symbol):
                llr_ones = dists[
                    [j for j in range(self.order) if (j >> b) & 1]
                ]
                llr_zeros = dists[
                    [j for j in range(self.order) if not ((j >> b) & 1)]
                ]
                bits[i * self.bits_per_symbol + b] = (
                    np.min(llr_zeros) - np.min(llr_ones)
                )
        return bits

    def demodulate_hard(self, symbols):
        indices = np.argmin(
            np.abs(symbols[:, np.newaxis] - self.constellation[np.newaxis, :]) ** 2,
            axis=1,
        )
        bits = np.zeros(len(symbols) * self.bits_per_symbol, dtype=np.int8)
        for i, idx in enumerate(indices):
            for b in range(self.bits_per_symbol):
                bits[i * self.bits_per_symbol + b] = (idx >> b) & 1
        return bits

    def symbol_to_llr(self, symbol, noise_variance):
        bits = np.zeros(self.bits_per_symbol, dtype=float)
        for b in range(self.bits_per_symbol):
            ones = self.constellation[
                [j for j in range(self.order) if (j >> b) & 1]
            ]
            zeros = self.constellation[
                [j for j in range(self.order) if not ((j >> b) & 1)]
            ]
            d1 = np.min(np.abs(symbol - ones) ** 2) / noise_variance
            d0 = np.min(np.abs(symbol - zeros) ** 2) / noise_variance
            bits[b] = d0 - d1
        return bits
