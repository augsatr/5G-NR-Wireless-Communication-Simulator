import numpy as np


class Modulator:
    def __init__(self, scheme="QPSK"):
        self.scheme = scheme.upper()
        self._setup()

    def _setup(self):
        if self.scheme == "BPSK":
            self.order = 2
            self.bits_per_symbol = 1
            self._constellation = np.array([-1 + 0j, 1 + 0j])
        elif self.scheme == "QPSK":
            self.order = 4
            self.bits_per_symbol = 2
            self._constellation = np.array([
                1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j
            ]) / np.sqrt(2)
        elif self.scheme == "16QAM":
            self.order = 16
            self.bits_per_symbol = 4
            self._build_qam(4)
        elif self.scheme == "64QAM":
            self.order = 64
            self.bits_per_symbol = 6
            self._build_qam(8)
        elif self.scheme == "256QAM":
            self.order = 256
            self.bits_per_symbol = 8
            self._build_qam(16)
        else:
            raise ValueError(f"Unsupported modulation: {self.scheme}")

    def _build_qam(self, m):
        indices = np.arange(m)
        gray = indices ^ (indices >> 1)
        levels = 2 * gray - m + 1
        a, b = np.meshgrid(levels, levels)
        constellation = a.ravel() + 1j * b.ravel()
        power = np.mean(np.abs(constellation) ** 2)
        self._constellation = constellation / np.sqrt(power)

    def modulate(self, bits):
        if len(bits) % self.bits_per_symbol != 0:
            pad = self.bits_per_symbol - (len(bits) % self.bits_per_symbol)
            bits = np.pad(bits, (0, pad))
        n_symbols = len(bits) // self.bits_per_symbol
        bits_reshaped = bits.reshape(n_symbols, self.bits_per_symbol)
        if self.scheme == "BPSK":
            indices = bits_reshaped[:, 0]
        else:
            indices = np.zeros(n_symbols, dtype=int)
            for b in range(self.bits_per_symbol):
                indices = indices | (bits_reshaped[:, b].astype(int) << b)
        return self._constellation[indices]

    def get_constellation(self):
        return self._constellation

    def get_avg_power(self):
        return np.mean(np.abs(self._constellation) ** 2)
