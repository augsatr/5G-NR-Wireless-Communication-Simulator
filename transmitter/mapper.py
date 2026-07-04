import numpy as np


class ResourceMapper:
    def __init__(self, n_subcarriers, n_ofdm_symbols, pilot_spacing=4):
        self.n_subcarriers = n_subcarriers
        self.n_ofdm_symbols = n_ofdm_symbols
        self.pilot_spacing = pilot_spacing

    def map_to_resource_grid(self, data_symbols, pilots=None):
        grid = np.zeros((self.n_subcarriers, self.n_ofdm_symbols), dtype=complex)
        if pilots is None:
            pilots = np.ones(self.n_subcarriers // self.pilot_spacing + 1, dtype=complex)
        data_idx = 0
        pilot_idx = 0
        for sym in range(self.n_ofdm_symbols):
            for sc in range(self.n_subcarriers):
                if sym % 2 == 0 and sc % self.pilot_spacing == 0:
                    if pilot_idx < len(pilots):
                        grid[sc, sym] = pilots[pilot_idx]
                        pilot_idx += 1
                else:
                    if data_idx < len(data_symbols):
                        grid[sc, sym] = data_symbols[data_idx]
                        data_idx += 1
        return grid

    def extract_data_symbols(self, grid):
        data_symbols = []
        for sym in range(self.n_ofdm_symbols):
            for sc in range(self.n_subcarriers):
                if not (sym % 2 == 0 and sc % self.pilot_spacing == 0):
                    data_symbols.append(grid[sc, sym])
        return np.array(data_symbols)

    def extract_pilots(self, grid):
        pilots = []
        for sym in range(self.n_ofdm_symbols):
            for sc in range(self.n_subcarriers):
                if sym % 2 == 0 and sc % self.pilot_spacing == 0:
                    pilots.append(grid[sc, sym])
        return np.array(pilots)
