import numpy as np


class ChannelEstimator:
    def __init__(self, method="ls"):
        self.method = method
        self.estimate = None

    def estimate_from_pilots(self, received_pilots, known_pilots, pilot_indices,
                             n_subcarriers, noise_variance=1e-10):
        if self.method == "ls":
            return self._ls_estimate(received_pilots, known_pilots,
                                     pilot_indices, n_subcarriers)
        elif self.method == "mmse":
            return self._mmse_estimate(received_pilots, known_pilots,
                                       pilot_indices, n_subcarriers,
                                       noise_variance)
        elif self.method == "linear":
            return self._linear_interp(received_pilots, known_pilots,
                                       pilot_indices, n_subcarriers)
        raise ValueError(f"Unknown estimator: {self.method}")

    def _ls_estimate(self, rx_pilots, tx_pilots, pilot_idx, n_sc):
        H_pilot = rx_pilots / (tx_pilots + 1e-12)
        H = np.ones(n_sc, dtype=complex)
        for i, idx in enumerate(pilot_idx):
            if idx < n_sc:
                H[idx] = H_pilot[i]
        for i in range(len(pilot_idx) - 1):
            start = pilot_idx[i]
            end = pilot_idx[i + 1]
            if end > start:
                H[start:end] = np.linspace(H[start], H[end], end - start)
        return H

    def _mmse_estimate(self, rx_pilots, tx_pilots, pilot_idx, n_sc, noise_var):
        H_ls = rx_pilots / (tx_pilots + 1e-12)
        K = len(pilot_idx)
        R_hh = np.zeros((K, K), dtype=complex)
        for i in range(K):
            for j in range(K):
                delta = abs(pilot_idx[i] - pilot_idx[j])
                R_hh[i, j] = np.sinc(delta / n_sc) * np.exp(
                    -1j * 2 * np.pi * delta / n_sc
                )
        R_hh += 1e-6 * np.eye(K)
        W = R_hh @ np.linalg.inv(R_hh + noise_var * np.eye(K))
        H_mmse_pilot = W @ H_ls
        H = np.ones(n_sc, dtype=complex)
        for i, idx in enumerate(pilot_idx):
            if idx < n_sc:
                H[idx] = H_mmse_pilot[i]
        for i in range(len(pilot_idx) - 1):
            start = pilot_idx[i]
            end = pilot_idx[i + 1]
            if end > start:
                H[start:end] = np.linspace(H[start], H[end], end - start)
        return H

    def _linear_interp(self, rx_pilots, tx_pilots, pilot_idx, n_sc):
        H_pilot = rx_pilots / (tx_pilots + 1e-12)
        H = np.ones(n_sc, dtype=complex)
        for i, idx in enumerate(pilot_idx):
            H[idx] = H_pilot[i]
        known = np.array(pilot_idx)
        values = H_pilot
        for i in range(n_sc):
            if i not in known:
                left_idx = np.searchsorted(known, i) - 1
                right_idx = left_idx + 1
                if left_idx < 0:
                    H[i] = values[0]
                elif right_idx >= len(known):
                    H[i] = values[-1]
                else:
                    x0, x1 = known[left_idx], known[right_idx]
                    ratio = (i - x0) / (x1 - x0) if x1 > x0 else 0
                    H[i] = (1 - ratio) * values[left_idx] + ratio * values[right_idx]
        return H

    def get_channel_estimate(self):
        return self.estimate
