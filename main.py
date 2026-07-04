#!/usr/bin/env python3
"""
5G-NR Wireless Communication Simulator
A modular PHY layer simulation framework.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import SimulationConfig
from simulation_controller import SimulationController
from experiments import experiment1, experiment2, experiment3
from visualization.ber_plot import BERPlotter
from visualization.constellation import ConstellationPlotter
from visualization.spectrum import SpectrumAnalyzer
from numerology.frame_structure import FrameStructure
from utils.logger import logger


def run_experiment(num):
    if num == 1:
        return experiment1.run()
    elif num == 2:
        return experiment2.run()
    elif num == 3:
        return experiment3.run()
    raise ValueError(f"Unknown experiment: {num}")


def run_full_analysis(config):
    controller = SimulationController(config)
    results = controller.run_full_analysis()

    snr_range = results["snr_range"]
    ber_data = results["ber"]

    plotter = BERPlotter()
    plotter.plot_ber_vs_snr(
        snr_range, {"Simulated": ber_data},
        title=f"BER vs SNR ({config.modulation_scheme}, {config.channel_model})",
        save_path="reports/full_analysis_ber.png"
    )

    logger.info("Full analysis complete. Results:")
    logger.info(f"  Modulation: {config.modulation_scheme}")
    logger.info(f"  Channel: {config.channel_model}")
    for i, snr in enumerate(snr_range):
        logger.info(f"  SNR={snr:2d} dB, BER={ber_data[i]:.2e}")

    return results


def show_numerology_info(mu=0):
    fs = FrameStructure(mu)
    logger.info("\n" + fs.describe_frame_structure())
    return fs


def main():
    parser = argparse.ArgumentParser(
        description="5G-NR Wireless Communication Simulator"
    )
    parser.add_argument(
        "--experiment", "-e", type=int, choices=[1, 2, 3],
        help="Run predefined experiment (1, 2, or 3)"
    )
    parser.add_argument(
        "--modulation", "-m", type=str,
        choices=["BPSK", "QPSK", "16QAM", "64QAM"],
        default="QPSK", help="Modulation scheme"
    )
    parser.add_argument(
        "--channel", "-c", type=str,
        choices=["AWGN", "Rayleigh", "Rician", "Doppler"],
        default="AWGN", help="Channel model"
    )
    parser.add_argument(
        "--snr-range", type=str, default="0,26,2",
        help="SNR range: start,stop,step"
    )
    parser.add_argument(
        "--num-bits", type=int, default=50000,
        help="Number of bits to simulate"
    )
    parser.add_argument(
        "--numerology", type=int, choices=[0, 1, 2, 3, 4],
        default=0, help="5G NR numerology (0-4)"
    )
    parser.add_argument(
        "--full-analysis", "-f", action="store_true",
        help="Run full PHY layer analysis"
    )
    parser.add_argument(
        "--constellation", action="store_true",
        help="Plot constellation diagram"
    )
    args = parser.parse_args()

    config = SimulationConfig()
    config.modulation_scheme = args.modulation
    config.channel_model = args.channel
    config.num_bits = args.num_bits
    config.numerology = args.numerology
    config.update_numerology()

    snr_parts = [int(x) for x in args.snr_range.split(",")]
    config.snr_db_range = np.arange(snr_parts[0], snr_parts[1], snr_parts[2])

    if args.experiment:
        logger.info(f"Running Experiment {args.experiment}...")
        run_experiment(args.experiment)
        return

    if args.numerology is not None:
        show_numerology_info(args.numerology)

    if args.constellation:
        from transmitter.modulation import Modulator
        mod = Modulator(args.modulation)
        const = mod.get_constellation()
        plotter = ConstellationPlotter()
        plotter.plot_constellation(
            const,
            title=f"{args.modulation} Constellation",
            save_path=f"reports/constellation_{args.modulation}.png"
        )
        logger.info(f"Constellation saved to reports/constellation_{args.modulation}.png")

    if args.full_analysis:
        run_full_analysis(config)
        return

    controller = SimulationController(config)
    results = controller.run_sweep()

    plotter = BERPlotter()
    plotter.plot_ber_vs_snr(
        results["snr_range"],
        {f"{args.modulation} ({args.channel})": results["ber"]},
        save_path="reports/ber_curve.png"
    )

    logger.info("Simulation complete. BER curve saved to reports/ber_curve.png")


if __name__ == "__main__":
    main()
