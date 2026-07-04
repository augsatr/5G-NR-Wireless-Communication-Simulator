import numpy as np


class RicianChannel:
    def __init__(self, k_factor=5.0, n_taps=6, delay_spread=1e-6,
                 sampling_rate=1e6, seed=None):
        self.k_factor = k_factor
        self.n_taps = n_taps
        self.delay_spread = delay_spread
        self.sampling_rate = sampling_rate
        self.rng = np.random.default_rng(seed)

    def apply_channel(self, signal, snr_db=None):
        n = len(signal)
        los_power = self.k_factor / (self.k_factor + 1)
        nlos_power = 1 / (self.k_factor + 1)

        los_component = np.sqrt(los_power) * np.ones(n, dtype=complex)

        nlos = (self.rng.standard_normal(n)
                + 1j * self.rng.standard_normal(n)) * np.sqrt(nlos_power / 2)

        channel = los_component + nlos
        output = signal * channel

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
        los = np.sqrt(self.k_factor / (self.k_factor + 1))
        nlos = (self.rng.standard_normal(n_subcarriers)
                + 1j * self.rng.standard_normal(n_subcarriers)) * np.sqrt(
            1 / (2 * (self.k_factor + 1))
        )
        return (los + nlos).astype(complex)
