"""5G-NR Simulator Web Application"""

import sys, os, io, base64, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file

from config import SimulationConfig
from transmitter.bit_generator import BitGenerator
from transmitter.modulation import Modulator
from receiver.demodulation import Demodulator
from receiver.equalizer import Equalizer
from channel.channel_models import ChannelFactory
from channel.awgn import AWGNChannel
from ofdm.transmitter import OFDMTransmitter
from ofdm.receiver import OFDMReceiver
from analysis.ber import BERCalculator
from analysis.throughput import ThroughputCalculator
from analysis.latency import LatencyAnalyzer
from analysis.spectral_efficiency import SpectralEfficiencyCalculator
from numerology.frame_structure import FrameStructure
from mimo.beamforming import Beamformer
from mimo.mumimo import MUMIMO
from coding.ldpc import LDPCEncoder, LDPCDecoder

app = Flask(__name__)
app.config["SECRET_KEY"] = "5g-nr-simulator-2024"


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ber-curve", methods=["POST"])
def api_ber_curve():
    params = request.get_json()
    mod = params.get("modulation", "QPSK")
    channel = params.get("channel", "AWGN")
    snr_start = float(params.get("snr_start", 0))
    snr_end = float(params.get("snr_end", 25))
    snr_step = float(params.get("snr_step", 2))
    num_bits = int(params.get("num_bits", 50000))

    snr_range = np.arange(snr_start, snr_end + 0.1, snr_step)
    bg = BitGenerator()
    modulator = Modulator(mod)
    demodulator = Demodulator(mod)
    bers = []

    ch_kwargs = {"k_factor": 5.0} if channel == "Rician" else {}
    channel_obj = ChannelFactory.create(channel, **ch_kwargs)
    ber_calc = BERCalculator()

    for snr in snr_range:
        bits = bg.generate_bits(num_bits)
        symbols = modulator.modulate(bits)
        if channel == "AWGN":
            rx = channel_obj.add_noise(symbols, snr)
        else:
            ch_resp = channel_obj.get_channel_response(len(symbols))
            faded = symbols * ch_resp
            rx = AWGNChannel().add_noise(faded, snr)
            eq = Equalizer("zero_forcing")
            rx = eq.equalize(rx, ch_resp)
        rx_bits = demodulator.demodulate_hard(rx)
        ber = ber_calc.calculate_ber(bits, rx_bits)
        bers.append(float(ber))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(snr_range, bers, "bo-", linewidth=2, markersize=6)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(f"BER vs SNR ({mod}, {channel})")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(bottom=1e-6, top=1)
    fig.tight_layout()

    return jsonify({
        "image": fig_to_base64(fig),
        "snr": snr_range.tolist(),
        "ber": bers
    })


@app.route("/api/constellation", methods=["POST"])
def api_constellation():
    params = request.get_json()
    mod = params.get("modulation", "QPSK")
    snr = float(params.get("snr", 15))

    modulator = Modulator(mod)
    bg = BitGenerator()
    bits = bg.generate_bits(10000)
    symbols = modulator.modulate(bits)
    rx = AWGNChannel().add_noise(symbols, snr)
    demodulator = Demodulator(mod)
    rx_bits = demodulator.demodulate_hard(rx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    const = modulator.get_constellation()
    ax1.scatter(const.real, const.imag, s=80, c="red", marker="o")
    for i, pt in enumerate(const):
        bits_str = format(i, f"0{modulator.bits_per_symbol}b")
        ax1.annotate(bits_str, (pt.real, pt.imag),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax1.set_title(f"{mod} Ideal Constellation")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(rx[:2000].real, rx[:2000].imag, s=10, alpha=0.5, c="blue")
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax2.set_title(f"{mod} Noisy (SNR={snr} dB)")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return jsonify({"image": fig_to_base64(fig)})


@app.route("/api/ofdm-spectrum", methods=["POST"])
def api_ofdm_spectrum():
    n_sc = 512
    cp = 128
    ofdm_tx = OFDMTransmitter(n_sc, 300, cp)
    modulator = Modulator("QPSK")
    bg = BitGenerator()
    bits = bg.generate_bits(10000)
    symbols = modulator.modulate(bits)
    ofdm_signal = ofdm_tx.modulate(symbols)
    fs = n_sc * 15e3

    fig, ax = plt.subplots(figsize=(10, 5))
    freq = np.fft.fftshift(np.fft.fftfreq(len(ofdm_signal), 1 / fs))
    spectrum = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(ofdm_signal))) + 1e-12)
    ax.plot(freq / 1e6, spectrum, "b-", linewidth=1)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("OFDM Transmit Spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return jsonify({"image": fig_to_base64(fig)})


@app.route("/api/mimo-sumrate", methods=["POST"])
def api_mimo_sumrate():
    params = request.get_json()
    n_tx = int(params.get("n_tx", 8))
    n_users = int(params.get("n_users", 2))

    mu = MUMIMO(n_tx=n_tx, n_users=n_users, seed=42)
    H = mu.generate_channels()
    snr_range = np.arange(0, 21, 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    for prec in ["zf", "mmse", "bd"]:
        rates = []
        for snr in snr_range:
            rate = mu.sum_rate(H, snr, prec)
            rates.append(rate)
        ax.plot(snr_range, rates, "o-", label=prec.upper(), linewidth=2, markersize=6)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Sum Rate (bps/Hz)")
    ax.set_title(f"MU-MIMO ({n_tx}×{n_users}) Precoding")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return jsonify({"image": fig_to_base64(fig)})


@app.route("/api/beam-pattern", methods=["POST"])
def api_beam_pattern():
    params = request.get_json()
    n_ant = int(params.get("n_antennas", 8))

    bf = Beamformer(n_antennas=n_ant)
    angles = np.linspace(-90, 90, 361)
    pattern = np.zeros(361)
    for i, theta in enumerate(angles):
        w = bf.steering_vector(0)
        a = bf.get_array_response(theta)
        pattern[i] = np.abs(np.conj(w) @ a) ** 2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(angles, 20 * np.log10(pattern + 1e-12), "b-", linewidth=2)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title(f"ULA Beam Pattern ({n_ant} elements, steered to 0°)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-40, 5)
    fig.tight_layout()
    return jsonify({"image": fig_to_base64(fig)})


@app.route("/api/numerology", methods=["POST"])
def api_numerology():
    params = request.get_json()
    mu = int(params.get("mu", 0))
    fs = FrameStructure(mu)
    info = fs.describe_frame_structure()
    return jsonify({"info": info})


@app.route("/api/modulation-compare", methods=["POST"])
def api_modulation_compare():
    channel = request.get_json().get("channel", "AWGN")
    snr_range = np.arange(0, 25, 2)
    mods = ["BPSK", "QPSK", "16QAM", "64QAM"]
    bg = BitGenerator()
    results = {}

    ch_kwargs = {"k_factor": 5.0} if channel == "Rician" else {}
    channel_obj = ChannelFactory.create(channel, **ch_kwargs)
    ber_calc = BERCalculator()

    for mod in mods:
        modulator = Modulator(mod)
        demodulator = Demodulator(mod)
        bers = []
        for snr in snr_range:
            bits = bg.generate_bits(50000)
            symbols = modulator.modulate(bits)
            if channel == "AWGN":
                rx = channel_obj.add_noise(symbols, snr)
            else:
                ch_resp = channel_obj.get_channel_response(len(symbols))
                faded = symbols * ch_resp
                rx = AWGNChannel().add_noise(faded, snr)
                rx = Equalizer("zero_forcing").equalize(rx, ch_resp)
            rx_bits = demodulator.demodulate_hard(rx)
            ber = ber_calc.calculate_ber(bits, rx_bits)
            bers.append(float(ber))
        results[mod] = bers

    fig, ax = plt.subplots(figsize=(9, 6))
    for mod, bers in results.items():
        ax.semilogy(snr_range, bers, "o-", label=mod, linewidth=2, markersize=5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(f"Modulation Comparison ({channel})")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(bottom=1e-6, top=1)
    fig.tight_layout()
    return jsonify({"image": fig_to_base64(fig)})


@app.route("/api/full-analysis", methods=["POST"])
def api_full_analysis():
    params = request.get_json()
    mod = params.get("modulation", "QPSK")
    channel = params.get("channel", "AWGN")

    config = SimulationConfig()
    config.modulation_scheme = mod
    config.channel_model = channel
    config.num_bits = 100000

    snr_range = np.arange(0, 26, 2)
    bg = BitGenerator()
    modulator = Modulator(mod)
    demodulator = Demodulator(mod)
    ch_kwargs = {"k_factor": 5.0} if channel == "Rician" else {}
    channel_obj = ChannelFactory.create(channel, **ch_kwargs)
    ber_calc = BERCalculator()
    bers = []

    for snr in snr_range:
        bits = bg.generate_bits(config.num_bits)
        symbols = modulator.modulate(bits)
        if channel == "AWGN":
            rx = channel_obj.add_noise(symbols, snr)
        else:
            ch_resp = channel_obj.get_channel_response(len(symbols))
            faded = symbols * ch_resp
            rx = AWGNChannel().add_noise(faded, snr)
            rx = Equalizer("zero_forcing").equalize(rx, ch_resp)
        rx_bits = demodulator.demodulate_hard(rx)
        ber = ber_calc.calculate_ber(bits, rx_bits)
        bers.append(float(ber))

    bandwidth = config.n_subcarriers * config.subcarrier_spacing
    mod_order = {"BPSK": 2, "QPSK": 4, "16QAM": 16, "64QAM": 64}.get(mod, 4)
    bps = int(np.log2(mod_order))

    tc = ThroughputCalculator()
    thr = tc.analyze(snr_range, bers, mod_order, bandwidth, n_subcarriers=300,
                     n_symbols=config.num_bits // bps)

    la = LatencyAnalyzer()
    lat = la.simulate_latency(snr_range, bers, packet_size_bits=1000,
                               data_rate_bps=thr["data_rate"][0] if len(thr["data_rate"]) > 0 else 1e6)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].semilogy(snr_range, bers, "bo-", linewidth=2)
    axes[0, 0].set_title("BER vs SNR")
    axes[0, 0].set_xlabel("SNR (dB)")
    axes[0, 0].set_ylabel("BER")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(snr_range, thr["throughput"], "rs-", linewidth=2)
    axes[0, 1].set_title("Throughput vs SNR")
    axes[0, 1].set_xlabel("SNR (dB)")
    axes[0, 1].set_ylabel("Throughput (bps)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(snr_range, lat["total_latency_ms"], "g^-", linewidth=2)
    axes[1, 0].set_title("Latency vs SNR")
    axes[1, 0].set_xlabel("SNR (dB)")
    axes[1, 0].set_ylabel("Latency (ms)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(snr_range, thr["spectral_efficiency"], "md-", linewidth=2)
    axes[1, 1].set_title("Spectral Efficiency vs SNR")
    axes[1, 1].set_xlabel("SNR (dB)")
    axes[1, 1].set_ylabel("bps/Hz")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Full PHY Analysis ({mod}, {channel})", fontsize=14)
    fig.tight_layout()

    return jsonify({
        "image": fig_to_base64(fig),
        "ber": bers,
        "snr": snr_range.tolist(),
    })


@app.route("/api/ldpc-demo", methods=["POST"])
def api_ldpc_demo():
    bits = np.random.randint(0, 2, 64)
    enc = LDPCEncoder(n=128, rate=0.5, seed=42)
    cw = enc.encode(bits)
    decoder = LDPCDecoder(enc.get_H(), max_iter=50)

    results = []
    for n_err in [0, 2, 5, 8, 10, 12, 15]:
        rx = cw.copy()
        if n_err > 0:
            idx = np.random.choice(len(rx), min(n_err, len(rx)), replace=False)
            rx[idx] ^= 1
        dec = decoder.decode_hard(rx)
        ber_before = float(np.mean(cw != rx))
        ber_after = float(np.mean(cw != dec[:len(cw)]))
        results.append({
            "errors": n_err,
            "ber_before": ber_before,
            "ber_after": ber_after,
        })

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
