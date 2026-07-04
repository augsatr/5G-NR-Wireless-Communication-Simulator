import numpy as np


class DopplerChannel:
    def __init__(self, doppler_frequency=100.0, sampling_rate=1e6,
                 n_paths=6, seed=None):
        self.fd = doppler_frequency
        self.fs = sampling_rate
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def _jakes_spectrum(self, n_samples):
        t = np.arange(n_samples) / self.fs
        theta = self.rng.uniform(0, 2 * np.pi, self.n_paths)
        phi = self.rng.uniform(0, 2 * np.pi, self.n_paths)
        h = np.zeros(n_samples, dtype=complex)
        for p in range(self.n_paths):
            h += np.exp(1j * (2 * np.pi * self.fd * t * np.cos(theta[p]) + phi[p]))
        return h / np.sqrt(self.n_paths)

    def apply_channel(self, signal, snr_db=None):
        n = len(signal)
        h = self._jakes_spectrum(n)
        output = signal * h
        if snr_db is not None:
            snr_linear = 10 ** (snr_db / 10)
            signal_power = np.mean(np.abs(output) ** 2)
            noise_power = signal_power / snr_linear
            noise = (
                np.sqrt(noise_power / 2)
                * (self.rng.standard_normal(output.shape)
                   + 1j * self.rng.standard_normal(output.shape))
            )
            output += noise
        return output

    def add_noise(self, signal, snr_db):
        return self.apply_channel(signal, snr_db)

    def get_channel_response(self, n_subcarriers):
        h = self._jakes_spectrum(n_subcarriers)
        return h
