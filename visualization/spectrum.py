import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SpectrumAnalyzer:
    def __init__(self):
        self.figsize = (10, 6)

    def plot_spectrum(self, signal, fs, title="Power Spectrum",
                      save_path=None):
        n = len(signal)
        freq = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
        spectrum = 20 * np.log10(
            np.abs(np.fft.fftshift(np.fft.fft(signal))) + 1e-12
        )
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(freq / 1e6, spectrum, "b-", linewidth=1.5)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Power (dB)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig

    def plot_ofdm_spectrum(self, signal, fs, n_subcarriers,
                           save_path=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        n = len(signal)
        freq = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
        spectrum = 20 * np.log10(
            np.abs(np.fft.fftshift(np.fft.fft(signal))) + 1e-12
        )
        ax1.plot(freq / 1e6, spectrum, "b-", linewidth=1.5)
        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_ylabel("Power (dB)")
        ax1.set_title("OFDM Transmit Spectrum")
        ax1.grid(True, alpha=0.3)
        scs = fs / n_subcarriers
        ax2.stem(np.arange(-n_subcarriers // 2, n_subcarriers // 2) * scs / 1e6,
                 np.ones(n_subcarriers), linefmt="r-", markerfmt="ro",
                 basefmt="k-")
        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("|H(f)|")
        ax2.set_title("Subcarrier Allocation")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig

    def plot_waterfall(self, signal, fs, n_fft=256, hop_size=64,
                       save_path=None):
        nperseg = n_fft
        n_overlap = n_fft - hop_size
        from scipy import signal as sg
        f, t, Sxx = sg.spectrogram(signal, fs, nperseg=nperseg,
                                    noverlap=n_overlap)
        fig, ax = plt.subplots(figsize=self.figsize)
        c = ax.pcolormesh(t * 1e3, f / 1e6, 10 * np.log10(Sxx + 1e-12),
                          shading="gouraud")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title("Spectrogram (Waterfall)")
        plt.colorbar(c, ax=ax, label="Power (dB)")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig
