import numpy as np
from .fft_utils import FFTHandler
from .cyclic_prefix import CyclicPrefix


class OFDMReceiver:
    def __init__(self, n_subcarriers=512, n_data_subcarriers=300,
                 cp_length=128):
        self.n_subcarriers = n_subcarriers
        self.n_data_subcarriers = n_data_subcarriers
        self.fft = FFTHandler(n_subcarriers)
        self.cp = CyclicPrefix(cp_length)

    def demodulate(self, time_signal, n_ofdm_symbols=None):
        sym_len = self.n_subcarriers + self.cp.cp_length
        if n_ofdm_symbols is None:
            n_ofdm_symbols = len(time_signal) // sym_len

        data_symbols = []
        dc_idx = self.n_subcarriers // 2
        half_data = self.n_data_subcarriers // 2
        start = dc_idx - half_data

        for i in range(n_ofdm_symbols):
            sym_start = i * sym_len
            sym_end = sym_start + sym_len
            sym_with_cp = time_signal[sym_start:sym_end]
            sym_no_cp = self.cp.remove_cp(sym_with_cp)
            freq_sym = self.fft.fft(sym_no_cp)
            data_sym = freq_sym[start:start + self.n_data_subcarriers]
            data_symbols.append(data_sym)

        return np.concatenate(data_symbols)
