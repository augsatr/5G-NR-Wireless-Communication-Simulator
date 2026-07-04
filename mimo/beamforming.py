import numpy as np


class Beamformer:
    def __init__(self, n_antennas=8, d_half_wavelength=True, seed=None):
        self.N = n_antennas
        self.d = 0.5 if d_half_wavelength else 1.0
        self.rng = np.random.default_rng(seed)

    def steering_vector(self, theta_deg):
        theta = np.deg2rad(theta_deg)
        k = 2 * np.pi * self.d
        n = np.arange(self.N)
        return np.exp(-1j * k * n * np.sin(theta)) / np.sqrt(self.N)

    def beamform_tx(self, symbols, theta_deg):
        w = self.steering_vector(theta_deg)
        return np.outer(w, symbols)

    def beamform_rx(self, received_signal, theta_deg):
        w = self.steering_vector(theta_deg)
        return received_signal.T @ w.conj()

    def get_array_response(self, theta_deg):
        theta = np.deg2rad(theta_deg)
        k = 2 * np.pi * self.d
        n = np.arange(self.N)
        return np.exp(-1j * k * n * np.sin(theta))

    def pattern(self, theta_range_deg=(-90, 90), n_points=361):
        angles = np.linspace(theta_range_deg[0], theta_range_deg[1], n_points)
        patterns = np.zeros((n_points, n_points), dtype=complex)
        for i, t1 in enumerate(angles):
            a_t1 = self.get_array_response(t1)
            for j, t2 in enumerate(angles):
                a_t2 = self.get_array_response(t2)
                patterns[i, j] = np.abs(np.conj(a_t1) @ a_t2) ** 2
        return angles, patterns

    def beamscan(self, received_signal, theta_range_deg=(-90, 90), n_points=361):
        angles = np.linspace(theta_range_deg[0], theta_range_deg[1], n_points)
        power = np.zeros(n_points)
        for i, theta in enumerate(angles):
            w = self.steering_vector(theta)
            power[i] = np.abs(np.conj(w) @ received_signal) ** 2
        best_theta = angles[np.argmax(power)]
        return best_theta, angles, power

    def mmse_beamformer(self, received, channel_matrix, noise_variance=1.0):
        R = channel_matrix @ channel_matrix.conj().T + noise_variance * np.eye(self.N)
        r = channel_matrix
        w = np.linalg.solve(R, r)[:, 0]
        return w / np.linalg.norm(w)

    def lcmv_beamformer(self, received, steering, constraint=1.0):
        R = received @ received.conj().T
        R_inv = np.linalg.pinv(R)
        w = R_inv @ steering
        w = w / (np.conj(steering) @ w + 1e-12)
        return w[:, 0]
