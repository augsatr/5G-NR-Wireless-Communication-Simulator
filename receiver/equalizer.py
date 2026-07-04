import numpy as np


class Equalizer:
    def __init__(self, method="zero_forcing"):
        self.method = method

    def equalize(self, received_symbols, channel_estimate, noise_variance=1e-10):
        if self.method == "zero_forcing":
            return self._zero_forcing(received_symbols, channel_estimate)
        elif self.method == "mmse":
            return self._mmse(received_symbols, channel_estimate, noise_variance)
        elif self.method == "none":
            return received_symbols
        raise ValueError(f"Unknown equalizer method: {self.method}")

    def _zero_forcing(self, received, channel):
        return received / (channel + 1e-12)

    def _mmse(self, received, channel, noise_var):
        channel_power = np.abs(channel) ** 2 + 1e-12
        mmse_filter = np.conj(channel) / (channel_power + noise_var)
        return received * mmse_filter
