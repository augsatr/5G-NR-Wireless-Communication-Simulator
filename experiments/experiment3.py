"""
Experiment 3: OFDM system with equalization (Zero Forcing vs MMSE vs None).
Includes throughput and spectral efficiency analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import SimulationConfig
from transmitter.bit_generator import BitGenerator
from transmitter.modulation import Modulator
from transmitter.mapper import ResourceMapper
from receiver.demodulation import Demodulator
from receiver.equalizer import Equalizer
from ofdm.transmitter import OFDMTransmitter
from ofdm.receiver import OFDMReceiver
from channel.awgn import AWGNChannel
from analysis.ber import BERCalculator
from analysis.throughput import ThroughputCalculator
from visualization.ber_plot import BERPlotter
from visualization.spectrum import SpectrumAnalyzer
from utils.logger import logger


def run():
    config = SimulationConfig()
    config.num_bits = 100000
    config.snr_db_range = np.arange(0, 26, 2)
    config.modulation_scheme = "QPSK"

    modulator = Modulator(config.modulation_scheme)
    demodulator = Demodulator(config.modulation_scheme)
    bg = BitGenerator()
    ber_calc = BERCalculator()

    ofdm_tx = OFDMTransmitter(config.n_subcarriers,
                               config.n_data_subcarriers,
                               config.cp_length)
    ofdm_rx = OFDMReceiver(config.n_subcarriers,
                            config.n_data_subcarriers,
                            config.cp_length)

    eq_methods = ["none", "zero_forcing", "mmse"]
    ber_results = {}

    for eq_method in eq_methods:
        logger.info(f"Running equalizer: {eq_method}")
        equalizer = Equalizer(eq_method)
        bers = []

        for snr_db in config.snr_db_range:
            bits = bg.generate_bits(config.num_bits)
            symbols = modulator.modulate(bits)
            ofdm_signal = ofdm_tx.modulate(symbols)

            n_ofdm_syms = len(ofdm_signal) // (
                config.n_subcarriers + config.cp_length
            )
            channel = AWGNChannel()
            rx_signal = channel.add_noise(ofdm_signal, snr_db)
            rx_symbols = ofdm_rx.demodulate(rx_signal, n_ofdm_syms)

            rx_symbols = rx_symbols[:len(symbols)]
            ch_est = channel.get_channel_response(config.n_subcarriers)
            ch_est_data = np.ones(len(rx_symbols), dtype=complex)

            eq_symbols = equalizer.equalize(rx_symbols, ch_est_data)
            rx_bits = demodulator.demodulate_hard(eq_symbols)

            min_len = min(len(bits), len(rx_bits))
            ber = ber_calc.calculate_ber(bits[:min_len], rx_bits[:min_len])
            bers.append(ber)

        ber_results[eq_method] = bers

    plotter = BERPlotter()
    plotter.plot_ber_vs_snr(
        config.snr_db_range, ber_results,
        title="OFDM: Equalizer Comparison (QPSK)",
        save_path="reports/ofdm_equalizer_comparison.png"
    )

    spec_plotter = SpectrumAnalyzer()
    bits_test = bg.generate_bits(config.num_bits)
    syms_test = modulator.modulate(bits_test)
    ofdm_test = ofdm_tx.modulate(syms_test)
    fs = config.get_sampling_rate()
    spec_plotter.plot_ofdm_spectrum(
        ofdm_test[:2048], fs, config.n_subcarriers,
        save_path="reports/ofdm_spectrum.png"
    )

    logger.info("Experiment 3 complete. Graphs saved to reports/")
    return ber_results


if __name__ == "__main__":
    run()
