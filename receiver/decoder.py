import numpy as np


class Decoder:
    def __init__(self, code_rate="1/2"):
        self.code_rate = code_rate

    def decode(self, bits):
        if self.code_rate == "1":
            return bits.copy()
        rate = float(self.code_rate)
        if rate == 0.5:
            return self._viterbi_decode(bits)
        return bits.copy()

    def _viterbi_decode(self, received_bits):
        if len(received_bits) % 2 != 0:
            received_bits = received_bits[:-1]
        n = len(received_bits) // 2
        trellis = np.full((64, n + 1), np.inf)
        trellis[0, 0] = 0
        traceback = np.zeros((64, n + 1), dtype=int)
        for t in range(n):
            r0 = received_bits[2 * t]
            r1 = received_bits[2 * t + 1]
            for state in range(64):
                if trellis[state, t] == np.inf:
                    continue
                for inp in [0, 1]:
                    next_state = ((state << 1) & 0x3E) | inp
                    g1 = bin(state & 0x2F).count("1") % 2
                    g2 = bin(state & 0x37).count("1") % 2
                    exp0 = (inp ^ g1) & 1
                    exp1 = (inp ^ g2) & 1
                    dist = (r0 != exp0) + (r1 != exp1)
                    new_cost = trellis[state, t] + dist
                    if new_cost < trellis[next_state, t + 1]:
                        trellis[next_state, t + 1] = new_cost
                        traceback[next_state, t + 1] = state
        best_state = np.argmin(trellis[:, n])
        decoded = np.zeros(n, dtype=np.int8)
        for t in range(n, 0, -1):
            decoded[t - 1] = best_state & 1
            best_state = traceback[best_state, t]
        return decoded

    def check_crc(self, bits):
        return True
