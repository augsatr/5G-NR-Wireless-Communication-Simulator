import numpy as np
from config import SimulationConfig
from transmitter.bit_generator import BitGenerator
from transmitter.modulation import Modulator
from transmitter.mapper import ResourceMapper
from transmitter.coding import Encoder
from receiver.demodulation import Demodulator
from receiver.decoder import Decoder
from receiver.equalizer import Equalizer
from ofdm.transmitter import OFDMTransmitter
from ofdm.receiver import OFDMReceiver
from channel.channel_models import ChannelFactory
from analysis.ber import BERCalculator
from analysis.throughput import ThroughputCalculator
from analysis.latency import LatencyAnalyzer
from analysis.spectral_efficiency import SpectralEfficiencyCalculator
from utils.logger import logger
import time


class SimulationController:
    def __init__(self, config=None):
        self.config = config or SimulationConfig()
        self.results = {}

    def run_single(self, snr_db):
        bg = BitGenerator()
        modulator = Modulator(self.config.modulation_scheme)
        demodulator = Demodulator(self.config.modulation_scheme)
        encoder = Encoder(code_rate="1")
        decoder = Decoder(code_rate="1")

        bits = bg.generate_bits(self.config.num_bits)
        encoded = encoder.encode(bits)
        symbols = modulator.modulate(encoded)

        channel = ChannelFactory.create(
            self.config.channel_model,
            sampling_rate=self.config.sampling_rate
        )
        rx_symbols = channel.add_noise(symbols, snr_db)

        equalizer = Equalizer("none")
        ch_est = channel.get_channel_response(len(rx_symbols))
        eq_symbols = equalizer.equalize(rx_symbols, ch_est)

        rx_bits = demodulator.demodulate_hard(eq_symbols)
        decoded = decoder.decode(rx_bits)

        ber_calc = BERCalculator()
        ber = ber_calc.calculate_ber(bits, decoded)

        return {"ber": ber, "bits_sent": len(bits)}

    def run_sweep(self):
        snr_range = self.config.snr_db_range
        bers = []

        for snr_db in snr_range:
            t_start = time.time()
            result = self.run_single(snr_db)
            elapsed = time.time() - t_start
            bers.append(result["ber"])
            logger.info(
                f"SNR={snr_db:2d} dB, BER={result['ber']:.2e}, "
                f"time={elapsed:.2f}s"
            )

        self.results["snr_range"] = snr_range
        self.results["ber"] = bers
        return self.results

    def run_mimo(self, snr_db):
        from mimo.mimo2x2 import MIMO2x2
        from mimo.alamouti import AlamoutiEncoder, AlamoutiDecoder

        config = self.config
        bg = BitGenerator()
        modulator = Modulator(config.modulation_scheme)
        demodulator = Demodulator(config.modulation_scheme)

        bits = bg.generate_bits(config.num_bits)
        symbols = modulator.modulate(bits)

        if config.mimo_scheme == "alamouti":
            enc = AlamoutiEncoder()
            tx1, tx2 = enc.encode(symbols)
            mimo = MIMO2x2()
            H = mimo.generate_channel()
            rx1 = H[0, 0] * tx1[:len(tx1)] + H[0, 1] * tx2[:len(tx2)]
            rx2 = H[1, 0] * tx1[:len(tx1)] + H[1, 1] * tx2[:len(tx2)]
            noise_pwr = 10 ** (-snr_db / 10)
            rx1 += np.sqrt(noise_pwr / 2) * (
                np.random.randn(len(rx1)) + 1j * np.random.randn(len(rx1))
            )
            rx2 += np.sqrt(noise_pwr / 2) * (
                np.random.randn(len(rx2)) + 1j * np.random.randn(len(rx2))
            )
            dec = AlamoutiDecoder()
            rx_symbols = dec.decode(rx1, rx2, H[0, 0], H[0, 1])
        else:
            mimo = MIMO2x2()
            H = mimo.generate_channel()
            rx1, rx2, H = mimo.transmit(symbols, H, snr_db)
            s1, s2 = mimo.zero_forcing_detector(rx1, rx2, H)
            rx_symbols = np.zeros(len(s1) + len(s2), dtype=complex)
            rx_symbols[0::2] = s1
            rx_symbols[1::2] = s2

        rx_bits = demodulator.demodulate_hard(rx_symbols)
        ber_calc = BERCalculator()
        ber = ber_calc.calculate_ber(bits, rx_bits)
        return {"ber": ber, "mimo_scheme": config.mimo_scheme}

    def run_full_analysis(self):
        logger.info("=== Running Full PHY Layer Analysis ===")
        self.run_sweep()

        throughput_calc = ThroughputCalculator()
        bandwidth = self.config.n_subcarriers * self.config.subcarrier_spacing
        mod_order = {
            "BPSK": 2, "QPSK": 4, "16QAM": 16, "64QAM": 64
        }.get(self.config.modulation_scheme, 4)

        throughput_results = throughput_calc.analyze(
            self.results["snr_range"],
            self.results["ber"],
            mod_order,
            bandwidth,
            n_subcarriers=self.config.n_subcarriers,
            n_symbols=self.config.num_bits // int(np.log2(mod_order)),
        )
        self.results["throughput"] = throughput_results

        latency_analyzer = LatencyAnalyzer()
        latency_results = latency_analyzer.simulate_latency(
            self.results["snr_range"],
            self.results["ber"],
            packet_size_bits=1000,
            data_rate_bps=throughput_results["data_rate"][0]
            if len(throughput_results["data_rate"]) > 0 else 1e6,
        )
        self.results["latency"] = latency_results

        spec_eff = SpectralEfficiencyCalculator()
        se_results = spec_eff.calculate(
            self.results["ber"][0] if len(self.results["ber"]) > 0 else 1,
            mod_order,
            bandwidth,
        )
        self.results["spectral_efficiency"] = se_results

        return self.results
