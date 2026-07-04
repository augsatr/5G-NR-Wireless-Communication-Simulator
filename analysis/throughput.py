import numpy as np


class ThroughputCalculator:
    def __init__(self):
        self.results = {}

    def calculate_throughput(self, ber, data_rate, overhead=0.1):
        error_free_rate = (1 - ber) ** 8
        return data_rate * error_free_rate * (1 - overhead)

    def calculate_data_rate(self, n_subcarriers, n_symbols, bits_per_symbol,
                            cp_overhead=0.07, code_rate=1.0):
        raw_rate = n_subcarriers * n_symbols * bits_per_symbol
        effective_rate = raw_rate * (1 - cp_overhead) * code_rate
        return effective_rate

    def spectral_efficiency_from_throughput(self, throughput, bandwidth):
        return throughput / bandwidth if bandwidth > 0 else 0

    def analyze(self, snr_range, bers, modulation_order, bandwidth,
                n_subcarriers=300, n_symbols=14000, cp_overhead=0.07):
        data_rates = np.zeros(len(snr_range))
        throughputs = np.zeros(len(snr_range))
        spec_effs = np.zeros(len(snr_range))

        for i, (snr, ber) in enumerate(zip(snr_range, bers)):
            data_rate = self.calculate_data_rate(
                n_subcarriers, n_symbols,
                int(np.log2(modulation_order)),
                cp_overhead
            )
            data_rates[i] = data_rate
            throughputs[i] = self.calculate_throughput(ber, data_rate)
            spec_effs[i] = self.spectral_efficiency_from_throughput(
                throughputs[i], bandwidth
            )

        self.results = {
            "snr_range": snr_range,
            "data_rate": data_rates,
            "throughput": throughputs,
            "spectral_efficiency": spec_effs,
        }
        return self.results
