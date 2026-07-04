import numpy as np


class SimulationConfig:
    def __init__(self):
        # Simulation parameters
        self.num_bits = 100000
        self.snr_db_range = np.arange(0, 31, 2)

        # Modulation
        self.modulation_scheme = "QPSK"  # BPSK, QPSK, 16QAM, 64QAM

        # Channel
        self.channel_model = "AWGN"  # AWGN, Rayleigh, Rician, Doppler
        self.rician_k_factor = 5.0
        self.doppler_frequency = 100.0
        self.sampling_rate = 1e6

        # OFDM
        self.n_subcarriers = 512
        self.n_data_subcarriers = 300
        self.cp_length = 128
        self.n_ofdm_symbols = 14

        # Numerology (5G NR)
        self.numerology = 0  # 0=15kHz, 1=30kHz, 2=60kHz
        self.subcarrier_spacing = 15e3  # Hz

        # MIMO
        self.n_tx = 2
        self.n_rx = 2
        self.mimo_scheme = "spatial_multiplexing"  # spatial_multiplexing, alamouti

        # Pilot
        self.pilot_spacing = 4  # pilot every N subcarriers

        # Results
        self.num_trials = 5

    def update_numerology(self):
        spacings = {0: 15e3, 1: 30e3, 2: 60e3}
        self.subcarrier_spacing = spacings.get(self.numerology, 15e3)

    def get_sampling_rate(self):
        return self.n_subcarriers * self.subcarrier_spacing
