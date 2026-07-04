import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class BERPlotter:
    def __init__(self):
        self.figsize = (10, 7)

    def plot_ber_vs_snr(self, snr_range, bers_dict, title="BER vs SNR",
                        save_path=None):
        fig, ax = plt.subplots(figsize=self.figsize)
        colors = plt.cm.tab10(np.linspace(0, 1, len(bers_dict)))
        for (label, bers), color in zip(bers_dict.items(), colors):
            ax.semilogy(snr_range, bers, "o-", label=label, color=color,
                        linewidth=2, markersize=6)
        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=11)
        ax.set_ylim(bottom=1e-6, top=1)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig

    def plot_ber_comparison_modulation(self, snr_range, results_dict,
                                       save_path=None):
        fig, ax = plt.subplots(figsize=self.figsize)
        for mod_name, bers in results_dict.items():
            ax.semilogy(snr_range, bers, "o-", label=mod_name, linewidth=2,
                        markersize=5)
        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("BER", fontsize=12)
        ax.set_title("BER Comparison across Modulation Schemes", fontsize=14)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=11)
        ax.set_ylim(bottom=1e-6, top=1)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig

    def plot_ber_comparison_channel(self, snr_range, results_dict,
                                    modulation="QPSK", save_path=None):
        fig, ax = plt.subplots(figsize=self.figsize)
        for ch_name, bers in results_dict.items():
            ax.semilogy(snr_range, bers, "o-", label=ch_name, linewidth=2,
                        markersize=5)
        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("BER", fontsize=12)
        ax.set_title(f"BER Comparison across Channels ({modulation})",
                     fontsize=14)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=11)
        ax.set_ylim(bottom=1e-6, top=1)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig

    def plot_comprehensive(self, snr_range, results, save_path=None):
        n_plots = len(results)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        for ax, (title, data) in zip(axes, results.items()):
            if isinstance(data, dict):
                for label, bers in data.items():
                    ax.semilogy(snr_range, bers, "o-", label=label,
                                linewidth=2, markersize=4)
            else:
                ax.semilogy(snr_range, data, "o-", linewidth=2, markersize=4)
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel("BER")
            ax.set_title(title)
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=9)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        return fig
