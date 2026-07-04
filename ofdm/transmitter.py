import numpy as np
from .fft_utils import FFTHandler
from .cyclic_prefix import CyclicPrefix


class OFDMTransmitter:
    def __init__(self, n_subcarriers=512, n_data_subcarriers=300,
                 cp_length=128):
        self.n_subcarriers = n_subcarriers
        self.n_data_subcarriers = n_data_subcarriers
        self.fft = FFTHandler(n_subcarriers)
        self.cp = CyclicPrefix(cp_length)

    def modulate(self, data_symbols):
        n_symbols = len(data_symbols) // self.n_data_subcarriers
        if len(data_symbols) % self.n_data_subcarriers != 0:
            n_symbols += 1
            pad = n_symbols * self.n_data_subcarriers - len(data_symbols)
            data_symbols = np.pad(data_symbols, (0, pad))

        ofdm_symbols = data_symbols.reshape(n_symbols, self.n_data_subcarriers)

        time_signal = []
        dc_idx = self.n_subcarriers // 2
        half_data = self.n_data_subcarriers // 2

        for sym in ofdm_symbols:
            freq_grid = np.zeros(self.n_subcarriers, dtype=complex)
            start = dc_idx - half_data
            freq_grid[start:start + self.n_data_subcarriers] = sym
            time_sym = self.fft.ifft(freq_grid)
            time_sym_cp = self.cp.add_cp(time_sym)
            time_signal.append(time_sym_cp)

        return np.concatenate(time_signal)

    def get_pilot_pattern(self):
        grid = np.zeros((self.n_subcarriers, 14), dtype=complex)
        for sym in range(14):
            if sym % 2 == 0:
                for sc in range(0, self.n_subcarriers, 4):
                    grid[sc, sym] = 1 + 0j
        return grid
