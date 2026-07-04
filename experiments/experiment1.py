"""
Experiment 1: BER vs SNR for BPSK, QPSK, 16-QAM, 64-QAM over AWGN.
Compares simulated BER with theoretical BER curves.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import SimulationConfig
from transmitter.bit_generator import BitGenerator
from transmitter.modulation import Modulator
from receiver.demodulation import Demodulator
from channel.awgn import AWGNChannel
from analysis.ber import BERCalculator
from visualization.ber_plot import BERPlotter
from visualization.constellation import ConstellationPlotter
from utils.logger import logger


def run():
    config = SimulationConfig()
    config.num_bits = 50000
    config.snr_db_range = np.arange(0, 26, 2)

    modulations = ["BPSK", "QPSK", "16QAM", "64QAM"]
    ber_results = {}
    ser_results = {}
    bg = BitGenerator()
    ber_calc = BERCalculator()
    channel = AWGNChannel()

    for mod_name in modulations:
        logger.info(f"Running {mod_name}...")
        modulator = Modulator(mod_name)
        demodulator = Demodulator(mod_name)
        bers = []
        sers = []

        for snr_db in config.snr_db_range:
            bits = bg.generate_bits(config.num_bits)
            symbols = modulator.modulate(bits)
            rx_symbols = channel.add_noise(symbols, snr_db)
            rx_bits = demodulator.demodulate_hard(rx_symbols)

            ber = ber_calc.calculate_ber(bits, rx_bits)
            ser = ber_calc.calculate_ser(symbols, rx_symbols)
            bers.append(ber)
            sers.append(ser)

        ber_results[mod_name] = bers
        ser_results[mod_name] = sers

    plotter = BERPlotter()
    plotter.plot_ber_comparison_modulation(
        config.snr_db_range, ber_results,
        save_path="reports/ber_modulation_comparison.png"
    )

    for mod_name in modulations:
        modulator = Modulator(mod_name)
        constellation = modulator.get_constellation()
        const_plotter = ConstellationPlotter()
        const_plotter.plot_constellation(
            constellation,
            title=f"{mod_name} Constellation",
            save_path=f"reports/constellation_{mod_name}.png"
        )

    logger.info("Experiment 1 complete. Graphs saved to reports/")
    return ber_results, ser_results


if __name__ == "__main__":
    run()
