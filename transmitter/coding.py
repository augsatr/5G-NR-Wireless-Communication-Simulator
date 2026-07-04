import numpy as np


class Encoder:
    def __init__(self, code_rate="1/2", use_crc=False):
        self.code_rate = code_rate
        self.use_crc = use_crc

    def encode(self, bits):
        if self.code_rate == "1":
            return bits.copy()
        rate = float(self.code_rate)
        if rate == 0.5:
            return self._simple_conv_encode(bits)
        elif rate == 0.75:
            encoded = self._simple_conv_encode(bits)
            return self._puncture(encoded, 3, 4)
        return bits.copy()

    def _simple_conv_encode(self, bits):
        n = len(bits)
        gen1 = np.array([1, 0, 1, 1, 0, 1, 1])
        gen2 = np.array([1, 1, 1, 1, 0, 0, 1])
        state = np.zeros(6, dtype=int)
        encoded = []
        for b in bits:
            state = np.roll(state, 1)
            state[0] = b
            encoded.append(int(np.dot(state, gen1[1:]) % 2))
            encoded.append(int(np.dot(state, gen2[1:]) % 2))
        return np.array(encoded, dtype=np.int8)

    def _puncture(self, bits, p, q):
        mask = np.ones(len(bits), dtype=bool)
        mask[p - 1::p] = False
        return bits[mask]

    def add_crc(self, bits):
        if not self.use_crc:
            return bits
        poly = 0x1021
        crc = 0xFFFF
        data = np.packbits(bits.astype(np.uint8))
        for b in data:
            crc ^= b << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ poly
                else:
                    crc <<= 1
                crc &= 0xFFFF
        crc_bits = np.unpackbits(np.array([crc >> 8, crc & 0xFF], dtype=np.uint8))
        return np.concatenate([bits, crc_bits])
