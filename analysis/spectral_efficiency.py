import numpy as np


class SpectralEfficiencyCalculator:
    def __init__(self):
        self.results = {}

    def calculate(self, ber, modulation_order, bandwidth, code_rate=1.0):
        max_se = np.log2(modulation_order) * code_rate
        actual_se = max_se * (1 - ber)
        return actual_se

    def shannon_capacity(self, snr_linear, bandwidth):
        return bandwidth * np.log2(1 + snr_linear)

    def analyze_modulation(self, snr_range, bers_dict, bandwidth):
        results = {}
        for mod_name, bers in bers_dict.items():
            mod_orders = {"BPSK": 2, "QPSK": 4, "16QAM": 16, "64QAM": 64}
            order = mod_orders.get(mod_name, 4)
            efficiencies = np.array([
                self.calculate(ber, order, bandwidth)
                for ber in bers
            ])
            results[mod_name] = efficiencies

        shannon_se = np.array([
            self.shannon_capacity(10 ** (snr / 10), bandwidth) / bandwidth
            for snr in snr_range
        ])

        self.results = {
            "snr_range": snr_range,
            "modulation_results": results,
            "shannon_limit": shannon_se,
        }
        return self.results

    def analyze_channel(self, snr_range, bers_dict, bandwidth):
        return self.analyze_modulation(snr_range, bers_dict, bandwidth)
