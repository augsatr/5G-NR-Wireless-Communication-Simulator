import numpy as np


class AdaptiveModulationCoding:
    MODE_TABLE = [
        {"mod": "BPSK",  "bps": 1,  "min_snr": -5, "code_rate": "1"},
        {"mod": "QPSK",  "bps": 2,  "min_snr": 3,  "code_rate": "1"},
        {"mod": "16QAM", "bps": 4,  "min_snr": 9,  "code_rate": "1"},
        {"mod": "64QAM", "bps": 6,  "min_snr": 16, "code_rate": "1"},
        {"mod": "256QAM","bps": 8,  "min_snr": 22, "code_rate": "1"},
    ]

    def __init__(self, target_ber=1e-3, use_cqi_feedback=True):
        self.target_ber = target_ber
        self.use_cqi_feedback = use_cqi_feedback
        self.current_mode = 0
        self.cqi_table = np.arange(0, 16)

    def select_mode(self, snr_estimate_db):
        mode = 0
        for i, m in enumerate(self.MODE_TABLE):
            if snr_estimate_db >= m["min_snr"]:
                mode = i
        self.current_mode = mode
        return self.MODE_TABLE[mode]

    def get_cqi(self, snr_db):
        cqi = max(0, min(15, int((snr_db + 5) / 2)))
        return cqi

    def adapt_to_cqi(self, cqi):
        mode_idx = min(cqi // 4, len(self.MODE_TABLE) - 1)
        self.current_mode = mode_idx
        return self.MODE_TABLE[mode_idx]

    def estimate_snr_from_ber(self, ber):
        if ber < 1e-6:
            return 30
        snr_map = {
            1: {1e-1: 0, 1e-2: 4, 1e-3: 6, 1e-4: 8, 1e-5: 10},
            2: {1e-1: 2, 1e-2: 6, 1e-3: 8, 1e-4: 11, 1e-5: 13},
            4: {1e-1: 6, 1e-2: 10, 1e-3: 13, 1e-4: 16, 1e-5: 18},
            6: {1e-1: 10, 1e-2: 15, 1e-3: 18, 1e-4: 21, 1e-5: 23},
        }
        return 10

    def get_throughput_for_snr(self, snr_db):
        mode = self.select_mode(snr_db)
        ber_est = max(1e-6, 0.5 * np.exp(-0.5 * snr_db))
        eff = mode["bps"] * (1 - ber_est)
        return eff, mode["mod"]

    def get_mode_table(self):
        return self.MODE_TABLE
