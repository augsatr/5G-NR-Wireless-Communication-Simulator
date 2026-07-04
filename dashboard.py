"""
5G-NR Simulator Interactive Dashboard (Streamlit)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SimulationConfig
from transmitter.bit_generator import BitGenerator
from transmitter.modulation import Modulator
from receiver.demodulation import Demodulator
from channel.channel_models import ChannelFactory
from analysis.ber import BERCalculator
from visualization.ber_plot import BERPlotter
from visualization.constellation import ConstellationPlotter
from visualization.spectrum import SpectrumAnalyzer
from ofdm.transmitter import OFDMTransmitter
from ofdm.receiver import OFDMReceiver
from numerology.frame_structure import FrameStructure
from utils.logger import logger

st.set_page_config(
    page_title="5G-NR Simulator",
    page_icon="📡",
    layout="wide",
)

st.title("5G-NR Wireless Communication Simulator")
st.markdown("Interactive PHY Layer Simulation Dashboard")

with st.sidebar:
    st.header("Configuration")

    modulation = st.selectbox(
        "Modulation Scheme",
        ["BPSK", "QPSK", "16QAM", "64QAM"],
        index=1,
    )

    channel_model = st.selectbox(
        "Channel Model",
        ["AWGN", "Rayleigh", "Rician", "Doppler"],
        index=0,
    )

    snr_min = st.slider("SNR Min (dB)", -5, 20, 0)
    snr_max = st.slider("SNR Max (dB)", 5, 40, 25)
    snr_step = st.slider("SNR Step (dB)", 1, 5, 2)

    num_bits = st.number_input("Number of Bits", 1000, 200000, 50000, step=1000)

    use_ofdm = st.checkbox("Enable OFDM", value=False)

    if use_ofdm:
        col1, col2 = st.columns(2)
        with col1:
            n_subcarriers = st.selectbox("FFT Size", [256, 512, 1024, 2048], index=1)
        with col2:
            cp_length = st.selectbox("CP Length", [32, 64, 128, 256], index=2)

    show_numerology = st.checkbox("Show Numerology Info", value=False)

    run_btn = st.button("Run Simulation", type="primary")

    st.header("Quick Plots")
    plot_const_btn = st.button("Plot Constellation")
    plot_spectrum_btn = st.button("Plot OFDM Spectrum")


def run_simulation(modulation, channel_model, snr_range, num_bits):
    snr_db_range = np.arange(snr_range[0], snr_range[1], snr_range[2])
    bg = BitGenerator()
    modulator = Modulator(modulation)
    demodulator = Demodulator(modulation)
    ch_kwargs = {}
    if channel_model == "Rician":
        ch_kwargs["k_factor"] = 5.0

    bers = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, snr_db in enumerate(snr_db_range):
        bits = bg.generate_bits(num_bits)
        symbols = modulator.modulate(bits)
        channel = ChannelFactory.create(channel_model, **ch_kwargs)
        rx_symbols = channel.add_noise(symbols, snr_db)
        rx_bits = demodulator.demodulate_hard(rx_symbols)

        ber_calc = BERCalculator()
        ber = ber_calc.calculate_ber(bits, rx_bits)
        bers.append(ber)

        progress_bar.progress((idx + 1) / len(snr_db_range))
        status_text.text(f"SNR={snr_db} dB, BER={ber:.2e}")

    status_text.text("Simulation complete!")
    return snr_db_range, bers


if run_btn:
    st.header("Results")
    snr_range = (snr_min, snr_max, snr_step)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("BER vs SNR")
        with st.spinner("Running simulation..."):
            snr_vals, bers = run_simulation(
                modulation, channel_model, snr_range, num_bits
            )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(snr_vals, bers, "bo-", linewidth=2, markersize=6)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.set_title(f"BER vs SNR ({modulation}, {channel_model})")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=1e-6, top=1)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Constellation (Last SNR Point)")
        bg = BitGenerator()
        modulator = Modulator(modulation)
        demodulator = Demodulator(modulation)
        bits = bg.generate_bits(num_bits)
        symbols = modulator.modulate(bits)
        ch_kwargs = {}
        if channel_model == "Rician":
            ch_kwargs["k_factor"] = 5.0
        channel = ChannelFactory.create(channel_model, **ch_kwargs)
        rx_symbols = channel.add_noise(symbols, snr_max - snr_step)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.scatter(
            rx_symbols[:2000].real,
            rx_symbols[:2000].imag,
            s=10, alpha=0.5, c="blue"
        )
        ax2.set_xlabel("In-Phase")
        ax2.set_ylabel("Quadrature")
        ax2.set_title(f"Received Constellation (SNR={snr_max - snr_step} dB)")
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal")
        st.pyplot(fig2)
        plt.close(fig2)

if plot_const_btn:
    st.header("Constellation Diagram")
    modulator = Modulator(modulation)
    const = modulator.get_constellation()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(const.real, const.imag, s=100, c="red", marker="o")
    for i, pt in enumerate(const):
        ax.annotate(
            f"{i:0{modulator.bits_per_symbol}b}",
            (pt.real, pt.imag),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(f"{modulation} Constellation (Gray-coded)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    st.pyplot(fig)
    plt.close(fig)

if plot_spectrum_btn:
    st.header("OFDM Spectrum")
    n_sc = st.session_state.get("n_subcarriers", 512)
    cp = st.session_state.get("cp_length", 128)
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
    spectrum = 20 * np.log10(
        np.abs(np.fft.fftshift(np.fft.fft(ofdm_signal))) + 1e-12
    )
    ax.plot(freq / 1e6, spectrum, "b-", linewidth=1)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("OFDM Transmit Spectrum")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

if show_numerology:
    st.header("5G NR Numerology")
    mu = st.selectbox("Numerology (μ)", [0, 1, 2, 3, 4], index=0)
    fs = FrameStructure(mu)
    st.text(fs.describe_frame_structure())

st.sidebar.markdown("---")
st.sidebar.markdown("**5G-NR Simulator v1.0**")
