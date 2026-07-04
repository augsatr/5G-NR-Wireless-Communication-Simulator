import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ChannelPlotter:
    def __init__(self):
        self.figsize = (10, 7)

    def plot_channel_response(self, freq_response, fs, title="Channel Response",
                              save_path=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        freq_axis = np.linspace(-fs / 2, fs / 2, len(freq_response))
        ax1.plot(freq_axis, 20 * np.log10(np.abs(freq_response) + 1e-12),
                 "b-", linewidth=1.5)
        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax2.plot(freq_axis, np.angle(freq_response, deg=True),
                 "r-", linewidth=1.5)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (degrees)")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig

    def plot_impulse_response(self, taps, delays, save_path=None):
        fig, ax = plt.subplots(figsize=self.figsize)
        markerline, _, _ = ax.stem(delays * 1e6, np.abs(taps), linefmt="b-",
                                    markerfmt="bo", basefmt="r-")
        ax.set_xlabel("Delay (µs)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Channel Impulse Response")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig

    def plot_doppler_spectrum(self, doppler_frequencies, spectrum,
                              save_path=None):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(doppler_frequencies, spectrum, "b-", linewidth=1.5)
        ax.set_xlabel("Doppler Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density")
        ax.set_title("Doppler Spectrum")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig
