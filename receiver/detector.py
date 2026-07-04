import numpy as np


class Detector:
    def __init__(self, method="maximum_likelihood"):
        self.method = method

    def detect(self, received, channel_matrix, constellation, noise_variance=1.0):
        if self.method == "maximum_likelihood":
            return self._ml_detect(received, channel_matrix, constellation)
        elif self.method == "zero_forcing":
            return self._zf_detect(received, channel_matrix, constellation)
        elif self.method == "mmse":
            return self._mmse_detect(received, channel_matrix, constellation, noise_variance)
        raise ValueError(f"Unknown detection method: {self.method}")

    def _ml_detect(self, received, H, constellation):
        best = np.zeros(received.shape[0], dtype=complex)
        for i in range(received.shape[0]):
            dists = np.abs(received[i] - H[i] * constellation) ** 2
            best[i] = constellation[np.argmin(dists)]
        return best

    def _zf_detect(self, received, H, constellation):
        estimated = received / (H + 1e-12)
        return self._slice(estimated, constellation)

    def _mmse_detect(self, received, H, constellation, noise_var):
        H_pow = np.abs(H) ** 2 + 1e-12
        estimated = received * np.conj(H) / (H_pow + noise_var)
        return self._slice(estimated, constellation)

    def _slice(self, symbols, constellation):
        indices = np.argmin(
            np.abs(symbols[:, np.newaxis] - constellation[np.newaxis, :]) ** 2,
            axis=1,
        )
        return constellation[indices]
