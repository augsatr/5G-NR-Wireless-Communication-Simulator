import numpy as np


class MIMO2x2:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def generate_channel(self):
        H = (self.rng.standard_normal((2, 2))
             + 1j * self.rng.standard_normal((2, 2))) / np.sqrt(2)
        return H

    def transmit(self, symbols, H, snr_db):
        tx1 = symbols[0::2]
        tx2 = symbols[1::2]
        min_len = min(len(tx1), len(tx2))
        tx1 = tx1[:min_len]
        tx2 = tx2[:min_len]
        rx1 = H[0, 0] * tx1 + H[0, 1] * tx2
        rx2 = H[1, 0] * tx1 + H[1, 1] * tx2
        noise = self._generate_noise(len(tx1), snr_db)
        rx1 += noise[0]
        rx2 += noise[1]
        return rx1, rx2, H

    def _generate_noise(self, n, snr_db):
        snr_linear = 10 ** (snr_db / 10)
        noise_power = 1.0 / snr_linear
        noise = (
            np.sqrt(noise_power / 2)
            * (self.rng.standard_normal((2, n))
               + 1j * self.rng.standard_normal((2, n)))
        )
        return noise

    def zero_forcing_detector(self, rx1, rx2, H):
        y = np.array([rx1, rx2])
        H_inv = np.linalg.pinv(H)
        estimated = H_inv @ y
        return estimated[0], estimated[1]

    def mmse_detector(self, rx1, rx2, H, snr_db):
        y = np.array([rx1, rx2])
        snr_linear = 10 ** (snr_db / 10)
        I = np.eye(2)
        H_H = H.conj().T
        W = np.linalg.inv(H_H @ H + (1 / snr_linear) * I) @ H_H
        estimated = W @ y
        return estimated[0], estimated[1]
