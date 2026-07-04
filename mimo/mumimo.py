import numpy as np


class MUMIMO:
    def __init__(self, n_tx=8, n_users=2, n_rx_per_user=1, seed=None):
        self.Nt = n_tx
        self.K = n_users
        self.Nr = n_rx_per_user
        self.Nr_total = n_users * n_rx_per_user
        self.rng = np.random.default_rng(seed)

    def generate_channels(self):
        H = (self.rng.standard_normal((self.K * self.Nr, self.Nt))
             + 1j * self.rng.standard_normal((self.K * self.Nr, self.Nt))) / np.sqrt(2)
        return H

    def zf_precoder(self, H):
        H_H = H.conj().T
        return H_H @ np.linalg.inv(H @ H_H + 1e-12 * np.eye(self.K * self.Nr))

    def mmse_precoder(self, H, snr_db):
        snr_linear = 10 ** (snr_db / 10)
        H_H = H.conj().T
        F = H_H @ np.linalg.inv(
            H @ H_H + (self.K * self.Nr / snr_linear) * np.eye(self.K * self.Nr)
        )
        return F

    def bd_precoder(self, H):
        F = np.zeros((self.Nt, self.K * self.Nr), dtype=complex)
        for k in range(self.K):
            idx = list(range(k * self.Nr)) + list(range((k + 1) * self.Nr, self.K * self.Nr))
            H_bar = H[idx, :]
            U, S, Vh = np.linalg.svd(H_bar, full_matrices=True)
            V_null = Vh[self.Nt - self.Nr:, :].conj().T
            H_k = H[k * self.Nr:(k + 1) * self.Nr, :]
            U_k, S_k, Vh_k = np.linalg.svd(H_k @ V_null, full_matrices=True)
            F[:, k * self.Nr:(k + 1) * self.Nr] = V_null @ Vh_k.conj().T
        return F

    def transmit(self, symbols_list, H, snr_db, precoder_type="zf"):
        K = self.K
        all_symbols = np.concatenate(symbols_list)
        if precoder_type == "zf":
            F = self.zf_precoder(H)
        elif precoder_type == "mmse":
            F = self.mmse_precoder(H, snr_db)
        elif precoder_type == "bd":
            F = self.bd_precoder(H)
        else:
            raise ValueError(f"Unknown precoder: {precoder_type}")
        power = np.trace(F.conj().T @ F) / (K * self.Nr)
        F = F / np.sqrt(power + 1e-12)
        tx = F @ all_symbols
        noise_power = 10 ** (-snr_db / 10) / 2
        noise = np.sqrt(noise_power) * (
            self.rng.standard_normal(self.Nt)
            + 1j * self.rng.standard_normal(self.Nt)
        )
        rx = H @ tx + noise
        rx_per_user = np.split(rx, K)
        F_per_user = [F[:, k * self.Nr:(k + 1) * self.Nr] for k in range(K)]
        return rx_per_user, F_per_user

    def user_receive(self, rx, H_k, F_k):
        H_eff = H_k @ F_k
        return rx / (np.diag(H_eff).mean() + 1e-12)

    def sum_rate(self, H, snr_db, precoder_type="zf"):
        K = self.K
        if precoder_type == "zf":
            F = self.zf_precoder(H)
        elif precoder_type == "mmse":
            F = self.mmse_precoder(H, snr_db)
        elif precoder_type == "bd":
            F = self.bd_precoder(H)
        else:
            raise ValueError(f"Unknown precoder: {precoder_type}")
        snr_linear = 10 ** (snr_db / 10)
        rate = 0
        for k in range(K):
            H_k = H[k * self.Nr:(k + 1) * self.Nr, :]
            F_k = F[:, k * self.Nr:(k + 1) * self.Nr]
            F_interf = np.delete(F, range(k * self.Nr, (k + 1) * self.Nr), axis=1)
            sig = H_k @ F_k
            sig_pow = np.trace(sig.conj().T @ sig)
            interf = H_k @ F_interf
            interf_pow = np.trace(interf.conj().T @ interf) + 1e-12
            sinr = snr_linear * sig_pow / (snr_linear * interf_pow + 1)
            rate += float(np.real(np.log2(1 + sinr)))
        return float(np.real(rate))
