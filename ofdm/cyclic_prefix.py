import numpy as np


class CyclicPrefix:
    def __init__(self, cp_length=128):
        self.cp_length = cp_length

    def add_cp(self, ofdm_symbol):
        if ofdm_symbol.ndim == 1:
            return np.concatenate([ofdm_symbol[-self.cp_length:], ofdm_symbol])
        return np.concatenate([ofdm_symbol[..., -self.cp_length:], ofdm_symbol], axis=-1)

    def remove_cp(self, signal_with_cp):
        if signal_with_cp.ndim == 1:
            return signal_with_cp[self.cp_length:]
        return signal_with_cp[..., self.cp_length:]
