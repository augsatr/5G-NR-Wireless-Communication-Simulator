import numpy as np


class LatencyAnalyzer:
    def __init__(self):
        self.results = {}

    def simulate_latency(self, snr_range, bers, packet_size_bits=1000,
                         data_rate_bps=1e6, processing_delay_us=100,
                         propagation_distance_km=1):
        propagation_speed = 3e8
        propagation_delay = propagation_distance_km * 1000 / propagation_speed
        transmission_delays = []
        retransmission_counts = []
        total_latencies = []

        for ber in bers:
            per = 1 - (1 - ber) ** packet_size_bits
            expected_retx = per / (1 - per) if per < 1 else 100
            tx_delay = packet_size_bits / data_rate_bps
            total_delay = (
                (1 + expected_retx) * (tx_delay + 2 * propagation_delay)
                + (1 + expected_retx) * processing_delay_us * 1e-6
            )
            transmission_delays.append(tx_delay * 1e6)
            retransmission_counts.append(expected_retx)
            total_latencies.append(total_delay * 1e3)

        self.results = {
            "snr_range": snr_range,
            "transmission_delay_us": np.array(transmission_delays),
            "expected_retransmissions": np.array(retransmission_counts),
            "total_latency_ms": np.array(total_latencies),
        }
        return self.results

    def get_tti_duration(self, numerology=0):
        slot_duration = 1 / (2 ** numerology)
        return slot_duration
