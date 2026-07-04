import numpy as np


class AWGNChannel:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def add_noise(self, signal, snr_db):
        snr_linear = 10 ** (snr_db / 10)
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / snr_linear
        noise = (
            np.sqrt(noise_power / 2)
            * (self.rng.standard_normal(signal.shape)
               + 1j * self.rng.standard_normal(signal.shape))
        )
        return signal + noise

    def get_channel_response(self, n_subcarriers):
        return np.ones(n_subcarriers, dtype=complex)
