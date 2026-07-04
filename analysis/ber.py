import numpy as np


class BERCalculator:
    def __init__(self):
        self.results = {}

    def calculate_ber(self, original_bits, received_bits):
        min_len = min(len(original_bits), len(received_bits))
        errors = np.sum(original_bits[:min_len] != received_bits[:min_len])
        return errors / min_len

    def calculate_ser(self, original_symbols, received_symbols):
        min_len = min(len(original_symbols), len(received_symbols))
        errors = np.sum(original_symbols[:min_len] != received_symbols[:min_len])
        return errors / min_len

    def run_monte_carlo(self, transmit_fn, num_bits, snr_range,
                        modulation_scheme="QPSK", channel_type="AWGN",
                        num_trials=3):
        from transmitter.modulation import Modulator
        from transmitter.bit_generator import BitGenerator
        from receiver.demodulation import Demodulator
        from channel.channel_models import ChannelFactory

        bg = BitGenerator()
        modulator = Modulator(modulation_scheme)
        demodulator = Demodulator(modulation_scheme)
        bpss = modulator.bits_per_symbol

        bers = np.zeros(len(snr_range))
        sers = np.zeros(len(snr_range))

        for idx, snr_db in enumerate(snr_range):
            ber_trials = []
            ser_trials = []
            for _ in range(num_trials):
                bits = bg.generate_bits(num_bits)
                symbols = modulator.modulate(bits)
                channel = ChannelFactory.create(channel_type)
                rx_symbols = channel.add_noise(symbols, snr_db)
                rx_bits = demodulator.demodulate_hard(rx_symbols)
                ber = self.calculate_ber(bits, rx_bits)
                ber_trials.append(ber)
            bers[idx] = np.mean(ber_trials)
            sers[idx] = 1 - (1 - bers[idx]) ** bpss

        self.results = {
            "snr_range": snr_range,
            "ber": bers,
            "ser": sers,
            "modulation": modulation_scheme,
            "channel": channel_type,
        }
        return self.results

    def theoretical_ber(self, snr_db, modulation="BPSK"):
        snr_linear = 10 ** (snr_db / 10)
        if modulation == "BPSK":
            return 0.5 * np.erfc(np.sqrt(snr_linear))
        elif modulation == "QPSK":
            return 0.5 * np.erfc(np.sqrt(snr_linear / 2))
        elif modulation == "16QAM":
            return 0.75 * np.erfc(np.sqrt(0.2 * snr_linear))
        elif modulation == "64QAM":
            return 0.875 * np.erfc(np.sqrt(0.05 * snr_linear))
        return 0
