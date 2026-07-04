import numpy as np


class DiversityCombiner:
    def __init__(self, method="maximal_ratio"):
        self.method = method

    def combine(self, received_signals, channel_gains=None):
        if self.method == "selection":
            return self._selection_combine(received_signals)
        elif self.method == "equal_gain":
            return self._equal_gain_combine(received_signals)
        elif self.method == "maximal_ratio":
            return self._mrc_combine(received_signals, channel_gains)
        raise ValueError(f"Unknown combining method: {self.method}")

    def _selection_combine(self, signals):
        powers = [np.abs(s) ** 2 for s in signals]
        best_idx = np.argmax(np.mean(powers, axis=1))
        return signals[best_idx]

    def _equal_gain_combine(self, signals):
        return np.sum(signals, axis=0) / len(signals)

    def _mrc_combine(self, signals, gains):
        if gains is None:
            return self._equal_gain_combine(signals)
        weights = np.conj(gains)
        combined = np.sum(weights * signals, axis=0)
        norm = np.sum(np.abs(weights) ** 2, axis=0) + 1e-12
        return combined / norm
