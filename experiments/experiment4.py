"""
Experiment 4: Advanced 5G NR Features
  - LDPC coding gain
  - Pilot-based channel estimation (LS vs MMSE vs Linear)
  - Adaptive Modulation & Coding (AMC)
  - Beamforming pattern and DOA estimation
  - MU-MIMO precoding (ZF, MMSE, BD)
  - HARQ with chase combining
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SimulationConfig
from transmitter.bit_generator import BitGenerator
from transmitter.modulation import Modulator
from transmitter.mapper import ResourceMapper
from transmitter.adaptive import AdaptiveModulationCoding
from receiver.demodulation import Demodulator
from receiver.equalizer import Equalizer
from receiver.channel_estimation import ChannelEstimator
from receiver.harq import HARQ, HybridARQController
from channel.rayleigh import RayleighChannel
from channel.awgn import AWGNChannel
from analysis.ber import BERCalculator
from receiver.channel_estimation import ChannelEstimator
from receiver.harq import HARQ, HybridARQController
from mimo.beamforming import Beamformer
from mimo.mumimo import MUMIMO
from utils.logger import logger


def demo_ldpc():
    logger.info("--- LDPC Coding ---")
    from coding.ldpc import LDPCEncoder, LDPCDecoder
    bg = BitGenerator(42)
    bits = bg.generate_bits(324)
    encoder = LDPCEncoder(n=648, rate=0.5, seed=42)
    codeword = encoder.encode(bits)
    decoder = LDPCDecoder(encoder.get_H(), max_iter=30)
    rx = codeword.copy()
    flip = np.random.randint(0, len(rx), 30)
    rx[flip] ^= 1
    decoded = decoder.decode_hard(rx)
    ber_before = np.mean(codeword != rx)
    ber_after = np.mean(codeword[:len(decoded)] != decoded[:len(codeword)])
    logger.info(f"  Uncorrected BER: {ber_before:.4f}")
    logger.info(f"  LDPC decoded BER: {ber_after:.4f}")
    logger.info(f"  Coding gain: {10 * np.log10(ber_before / max(ber_after, 1e-10)):.1f} dB")
    return ber_before, ber_after


def demo_channel_estimation():
    logger.info("--- Channel Estimation ---")
    n_sc = 512
    n_pilots = 64
    pilot_idx = np.linspace(0, n_sc - 1, n_pilots, dtype=int)
    tx_pilots = np.ones(n_pilots, dtype=complex)
    ch = RayleighChannel(flat_fading=False, seed=42)
    true_H = ch.get_channel_response(n_sc)
    rx_pilots = true_H[pilot_idx] * tx_pilots
    noise = (np.random.randn(n_pilots) + 1j * np.random.randn(n_pilots)) * 0.05
    rx_pilots += noise
    estimators = {"LS": ChannelEstimator("ls"),
                  "Linear": ChannelEstimator("linear"),
                  "MMSE": ChannelEstimator("mmse")}
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(20 * np.log10(np.abs(true_H) + 1e-12), "k-", label="True", linewidth=2)
    for name, est in estimators.items():
        H_est = est.estimate_from_pilots(rx_pilots, tx_pilots, pilot_idx, n_sc, 0.01)
        mse = np.mean(np.abs(H_est - true_H) ** 2)
        ax.plot(20 * np.log10(np.abs(H_est) + 1e-12), "--", label=f"{name} (MSE={mse:.4f})", linewidth=1)
        logger.info(f"  {name}: MSE = {mse:.6f}")
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Channel Estimation Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("reports/channel_estimation.png", dpi=150)
    plt.close(fig)
    logger.info("  Plot saved to reports/channel_estimation.png")


def demo_amc():
    logger.info("--- Adaptive Modulation & Coding ---")
    amc = AdaptiveModulationCoding(target_ber=1e-3)
    snr_range = np.arange(-5, 25, 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    modes = []
    bps_list = []
    for snr in snr_range:
        mode = amc.select_mode(snr)
        modes.append(mode["mod"])
        bps_list.append(mode["bps"])
        logger.info(f"  SNR={snr:2d} dB -> {mode['mod']} ({mode['bps']} bps/Hz)")
    ax1.step(snr_range, bps_list, "b-", linewidth=2, where="post")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("Bits per Symbol")
    ax1.set_title("AMC: Throughput per SNR")
    ax1.grid(True, alpha=0.3)
    ax2.step(snr_range, modes, "r-", linewidth=2, where="post")
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Modulation Scheme")
    ax2.set_title("AMC: Modulation Selection")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("reports/amc_selection.png", dpi=150)
    plt.close(fig)
    logger.info("  Plot saved to reports/amc_selection.png")
    return modes, bps_list


def demo_beamforming():
    logger.info("--- Beamforming ---")
    bf = Beamformer(n_antennas=8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    angles = np.linspace(-90, 90, 361)
    pattern = np.zeros(361)
    for i, theta in enumerate(angles):
        w = bf.steering_vector(0)
        a = bf.get_array_response(theta)
        pattern[i] = np.abs(np.conj(w) @ a) ** 2
    ax1.plot(angles, 20 * np.log10(pattern + 1e-12))
    ax1.set_xlabel("Angle (deg)")
    ax1.set_ylabel("Gain (dB)")
    ax1.set_title("Beam Pattern (steered to 0°)")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-40, 5)
    ax2.plot(angles, pattern)
    ax2.set_xlabel("Angle (deg)")
    ax2.set_ylabel("|Gain|")
    ax2.set_title("Beam Pattern (linear)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("reports/beam_pattern.png", dpi=150)
    plt.close(fig)
    logger.info("  Plot saved to reports/beam_pattern.png")


def demo_mumimo():
    logger.info("--- MU-MIMO ---")
    mu = MUMIMO(n_tx=8, n_users=2, seed=42)
    H = mu.generate_channels()
    snr_range = np.arange(0, 21, 2)
    fig, ax = plt.subplots(figsize=(10, 6))
    for prec in ["zf", "mmse", "bd"]:
        rates = []
        for snr in snr_range:
            rate = mu.sum_rate(H, snr, prec)
            rates.append(rate)
        ax.plot(snr_range, rates, "o-", label=prec.upper(), linewidth=2, markersize=6)
        logger.info(f"  {prec.upper()}: {rates[-1]:.1f} bps/Hz @ {snr_range[-1]}dB")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Sum Rate (bps/Hz)")
    ax.set_title("MU-MIMO Precoding Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("reports/mumimo_sumrate.png", dpi=150)
    plt.close(fig)
    logger.info("  Plot saved to reports/mumimo_sumrate.png")


def demo_harq():
    logger.info("--- HARQ ---")
    data = np.random.randint(0, 2, 1000)
    mod = Modulator("BPSK")
    fig, ax = plt.subplots(figsize=(10, 5))
    for snr_db in [0, 5, 10]:
        original = data.copy()
        symbols = mod.modulate(original)
        ch = AWGNChannel()
        rx = ch.add_noise(symbols, snr_db)
        ber_no_harq = BERCalculator().calculate_ber(original, (rx.real > 0).astype(int))
        bers_harq = [ber_no_harq]
        combined_rx = rx.copy()
        for tx in range(1, 5):
            new_rx = ch.add_noise(symbols, snr_db + 3 * np.log2(tx + 1))
            combined_rx = (combined_rx * tx + new_rx) / (tx + 1)
            decoded = (combined_rx.real > 0).astype(int)
            ber = BERCalculator().calculate_ber(original, decoded)
            bers_harq.append(ber)
        label = f"HARQ (SNR={snr_db}dB)"
        ax.semilogy(range(len(bers_harq)), bers_harq, "o-", label=label,
                     linewidth=2, markersize=8)
        eff_gain = 3 * np.log2(len(bers_harq))
        logger.info(f"  SNR={snr_db:2d}dB: Final BER={bers_harq[-1]:.2e}, "
                    f"eff. SNR gain={eff_gain:.1f}dB after {len(bers_harq)} TX")
    ax.set_xlabel("Transmission Number")
    ax.set_ylabel("BER")
    ax.set_title("HARQ Chase Combining Performance (BPSK, AWGN)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("reports/harq_performance.png", dpi=150)
    plt.close(fig)
    logger.info("  Plot saved to reports/harq_performance.png")


def run():
    logger.info("===== Advanced 5G NR Features =====")
    logger.info("")

    demo_ldpc()
    logger.info("")

    demo_channel_estimation()
    logger.info("")

    demo_amc()
    logger.info("")

    demo_beamforming()
    logger.info("")

    demo_mumimo()
    logger.info("")

    demo_harq()
    logger.info("")

    logger.info("===== All advanced demos complete =====")
    logger.info("Graphs saved to reports/")
    return True


if __name__ == "__main__":
    run()
