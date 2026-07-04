import numpy as np


class FFTHandler:
    def __init__(self, n_fft=512):
        self.n_fft = n_fft

    def ifft(self, frequency_signal):
        return np.fft.ifft(np.fft.ifftshift(frequency_signal)) * np.sqrt(self.n_fft)

    def fft(self, time_signal):
        return np.fft.fftshift(np.fft.fft(time_signal)) / np.sqrt(self.n_fft)

    def get_frequency_bins(self, fs):
        return np.fft.fftshift(np.fft.fftfreq(self.n_fft, 1 / fs))
