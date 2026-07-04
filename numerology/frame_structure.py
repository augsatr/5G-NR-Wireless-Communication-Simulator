import numpy as np


class FrameStructure:
    def __init__(self, numerology=0):
        self.mu = numerology
        self.scs = 15e3 * (2 ** self.mu)
        self.slots_per_subframe = 2 ** self.mu
        self.slots_per_frame = 10 * self.slots_per_subframe
        self.symbols_per_slot = 14

    def create_resource_grid(self, n_rb=52):
        n_subcarriers = n_rb * 12
        n_symbols = self.symbols_per_slot
        return np.zeros((n_subcarriers, n_symbols), dtype=complex)

    def get_frame_duration_ms(self):
        return 10.0

    def get_subframe_duration_ms(self):
        return 1.0

    def get_slot_duration_ms(self):
        return 1.0 / (2 ** self.mu)

    def get_symbol_duration_us(self):
        slot_us = self.get_slot_duration_ms() * 1000
        return slot_us / self.symbols_per_slot

    def describe_frame_structure(self):
        return (
            f"Numerology μ={self.mu}:\n"
            f"  Subcarrier Spacing: {self.scs / 1e3:.0f} kHz\n"
            f"  Frame Duration: {self.get_frame_duration_ms():.1f} ms\n"
            f"  Subframe Duration: {self.get_subframe_duration_ms():.1f} ms\n"
            f"  Slot Duration: {self.get_slot_duration_ms():.3f} ms\n"
            f"  Symbol Duration: {self.get_symbol_duration_us():.2f} µs\n"
            f"  Slots per Subframe: {self.slots_per_subframe}\n"
            f"  Slots per Frame: {self.slots_per_frame}\n"
            f"  Symbols per Slot: {self.symbols_per_slot}"
        )
