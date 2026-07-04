import numpy as np


class RayleighChannel:
    def __init__(self, n_taps=6, delay_spread=1e-6, sampling_rate=1e6,
                 flat_fading=True, seed=None):
        self.flat_fading = flat_fading
        self.n_taps = n_taps
        self.delay_spread = delay_spread
        self.sampling_rate = sampling_rate
        self.rng = np.random.default_rng(seed)
        if not flat_fading:
            self._generate_taps()

    def _generate_taps(self):
        delays = np.arange(self.n_taps) * (self.delay_spread / self.n_taps)
        power_profile = np.exp(-delays / self.delay_spread)
        power_profile /= np.sum(power_profile)
        self.taps_power = power_profile
        self.taps_delays = delays

    def apply_channel(self, signal, snr_db=None):
        n = len(signal)
        if self.flat_fading:
            h = (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)) / np.sqrt(2)
            output = signal * h
        else:
            h = (self.rng.standard_normal(self.n_taps)
                 + 1j * self.rng.standard_normal(self.n_taps))
            h *= np.sqrt(self.taps_power / 2)
            output = np.zeros(n, dtype=complex)
            for i in range(self.n_taps):
                tap_delay = int(self.taps_delays[i] * self.sampling_rate)
                if tap_delay < n:
                    output[tap_delay:] += h[i] * signal[:n - tap_delay]
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
        if self.flat_fading:
            return (self.rng.standard_normal(n_subcarriers)
                    + 1j * self.rng.standard_normal(n_subcarriers)) / np.sqrt(2)
        h_time = np.zeros(1024, dtype=complex)
        for i in range(self.n_taps):
            tap_delay = int(self.taps_delays[i] * self.sampling_rate)
            if tap_delay < len(h_time):
                h_time[tap_delay] = (
                    self.rng.standard_normal()
                    + 1j * self.rng.standard_normal()
                ) * np.sqrt(self.taps_power[i] / 2)
        h_freq = np.fft.fft(h_time)[:n_subcarriers]
        return h_freq
