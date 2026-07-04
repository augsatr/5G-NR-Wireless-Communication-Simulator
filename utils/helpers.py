import numpy as np


def bits_to_symbols(bits):
    return 2 * bits - 1


def symbols_to_bits(symbols):
    return (symbols > 0).astype(int)


def compute_energy_per_bit(symbols, modulation_order):
    return np.mean(np.abs(symbols) ** 2) / modulation_order


def compute_noise_variance(snr_db, energy_per_bit, modulation_order):
    snr_linear = 10 ** (snr_db / 10)
    return energy_per_bit / (modulation_order * snr_linear)


def compute_noise_power(snr_db, signal_power):
    snr_linear = 10 ** (snr_db / 10)
    return signal_power / snr_linear


def quantize_to_bits(values, n_bits):
    max_val = np.max(np.abs(values))
    if max_val == 0:
        return values
    scaled = values / max_val
    quant = np.round(scaled * (2 ** (n_bits - 1) - 1))
    return quant / (2 ** (n_bits - 1) - 1) * max_val


def gray_encode(bits):
    return bits ^ (bits >> 1)


def gray_decode(gray):
    mask = gray >> 1
    while mask.any():
        gray ^= mask
        mask >>= 1
    return gray
