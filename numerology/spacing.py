import numpy as np


class NumerologyManager:
    NR_SCS = {0: 15e3, 1: 30e3, 2: 60e3, 3: 120e3, 4: 240e3}
    NR_FFT_SIZES = {0: 512, 1: 1024, 2: 2048, 3: 4096, 4: 4096}

    def __init__(self, numerology=0):
        self.mu = numerology
        self.scs = self.NR_SCS[self.mu]
        self.n_fft = self.NR_FFT_SIZES[self.mu]
        self.slot_duration = 0.5 / (2 ** self.mu) * 1e-3
        self.symbol_duration = self.slot_duration / 14
        self.cp_duration = self.symbol_duration * 0.07

    @property
    def sampling_rate(self):
        return self.n_fft * self.scs

    def get_slot_structure(self):
        return {
            "mu": self.mu,
            "scs_khz": self.scs / 1e3,
            "fft_size": self.n_fft,
            "slot_duration_ms": self.slot_duration * 1e3,
            "symbol_duration_us": self.symbol_duration * 1e6,
            "cp_duration_us": self.cp_duration * 1e6,
            "sampling_rate_mhz": self.sampling_rate / 1e6,
            "symbols_per_slot": 14,
            "slots_per_subframe": 2 ** self.mu,
        }

    def get_bandwidth(self, n_rb=52):
        return n_rb * 12 * self.scs

    @staticmethod
    def describe_tradeoffs(mu):
        descriptions = {
            0: "15 kHz SCS: Large coverage, tolerant to delay spread, suitable for sub-6 GHz",
            1: "30 kHz SCS: Balance of coverage and throughput, common in FR1",
            2: "60 kHz SCS: Lower latency, higher throughput, used in FR2/mmWave",
            3: "120 kHz SCS: Very low latency, high Doppler tolerance, mmWave",
            4: "240 kHz SCS: Ultra-low latency, extreme throughput, specialized mmWave",
        }
        base = descriptions.get(mu, "Unknown numerology")
        if mu < 2:
            base += " - Better coverage, more robust to multipath"
        else:
            base += " - Better for high mobility and low latency"
        return base
