"""
Experiment 2: BER comparison across channel models (AWGN, Rayleigh, Rician).
Uses perfect CSI at receiver for fading channels.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import SimulationConfig
from transmitter.bit_generator import BitGenerator
from transmitter.modulation import Modulator
from receiver.demodulation import Demodulator
from receiver.equalizer import Equalizer
from channel.channel_models import ChannelFactory
from channel.awgn import AWGNChannel
from analysis.ber import BERCalculator
from visualization.ber_plot import BERPlotter
from utils.logger import logger


def run():
    config = SimulationConfig()
    config.num_bits = 50000
    config.snr_db_range = np.arange(0, 31, 2)
    config.modulation_scheme = "QPSK"

    channels = ["AWGN", "Rayleigh", "Rician"]
    ber_results = {}
    bg = BitGenerator()
    modulator = Modulator(config.modulation_scheme)
    demodulator = Demodulator(config.modulation_scheme)
    ber_calc = BERCalculator()

    for ch_name in channels:
        logger.info(f"Running {ch_name} channel...")
        ch_kwargs = {}
        if ch_name == "Rician":
            ch_kwargs["k_factor"] = 5.0
        channel = ChannelFactory.create(ch_name, **ch_kwargs)
        bers = []

        for snr_db in config.snr_db_range:
            bits = bg.generate_bits(config.num_bits)
            symbols = modulator.modulate(bits)

            if ch_name == "AWGN":
                rx_symbols = channel.add_noise(symbols, snr_db)
            else:
                ch_resp = channel.get_channel_response(len(symbols))
                faded = symbols * ch_resp
                rx_symbols = AWGNChannel().add_noise(faded, snr_db)
                eq = Equalizer("zero_forcing")
                rx_symbols = eq.equalize(rx_symbols, ch_resp)

            rx_bits = demodulator.demodulate_hard(rx_symbols)
            ber = ber_calc.calculate_ber(bits, rx_bits)
            bers.append(ber)
            logger.info(f"  SNR={snr_db:2d} dB, BER={ber:.2e}")

        ber_results[ch_name] = bers

    plotter = BERPlotter()
    plotter.plot_ber_comparison_channel(
        config.snr_db_range, ber_results,
        modulation=config.modulation_scheme,
        save_path="reports/ber_channel_comparison.png"
    )
    logger.info("Experiment 2 complete. Graph saved to reports/")
    return ber_results


if __name__ == "__main__":
    run()
