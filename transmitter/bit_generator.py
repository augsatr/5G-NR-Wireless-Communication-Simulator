import numpy as np


class BitGenerator:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def generate_bits(self, num_bits):
        return self.rng.integers(0, 2, size=num_bits).astype(np.int8)

    def generate_packet(self, payload_bytes):
        bits = np.unpackbits(
            np.frombuffer(payload_bytes, dtype=np.uint8)
        )
        return bits
