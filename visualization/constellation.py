import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ConstellationPlotter:
    def __init__(self):
        self.figsize = (10, 8)

    def plot_constellation(self, symbols, title="Constellation Diagram",
                           save_path=None):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(symbols.real, symbols.imag, s=20, alpha=0.6, c="blue")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("In-Phase")
        ax.set_ylabel("Quadrature")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig

    def plot_constellation_comparison(self, clean_symbols, noisy_symbols,
                                      snr_db, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.scatter(clean_symbols.real, clean_symbols.imag, s=15, alpha=0.5, c="green")
        ax1.set_title("Clean Constellation")
        ax1.set_xlabel("In-Phase")
        ax1.set_ylabel("Quadrature")
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal")

        ax2.scatter(noisy_symbols.real, noisy_symbols.imag, s=15, alpha=0.5, c="red")
        ax2.set_title(f"Noisy Constellation (SNR={snr_db} dB)")
        ax2.set_xlabel("In-Phase")
        ax2.set_ylabel("Quadrature")
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig

    def plot_grid(self, modulations, snr_db, save_path=None):
        n = len(modulations)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, (name, syms) in zip(axes, modulations):
            ax.scatter(syms.real, syms.imag, s=15, alpha=0.5)
            ax.set_title(f"{name} (SNR={snr_db} dB)")
            ax.set_xlabel("In-Phase")
            ax.set_ylabel("Quadrature")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig
