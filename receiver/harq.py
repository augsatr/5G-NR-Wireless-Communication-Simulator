import numpy as np


class HARQ:
    def __init__(self, max_retransmissions=4, combining_type="chase"):
        self.max_retransmissions = max_retransmissions
        self.combining_type = combining_type
        self.reset()

    def reset(self):
        self.transmission_count = 0
        self.buffer = None
        self.rv_version = 0

    def initial_transmit(self, data):
        self.reset()
        self.original_data = data.copy()
        self.transmission_count = 1
        return data

    def retransmit(self):
        if self.transmission_count >= self.max_retransmissions:
            return None
        self.transmission_count += 1
        self.rv_version = (self.rv_version + 1) % 4
        if self.combining_type == "chase":
            return self.original_data.copy()
        elif self.combining_type == "incremental_redundancy":
            return self._generate_ir_version(self.original_data, self.rv_version)
        return self.original_data.copy()

    def _generate_ir_version(self, data, rv):
        n = len(data)
        if rv == 0:
            return data.copy()
        elif rv == 1:
            return np.roll(data, n // 4)
        elif rv == 2:
            return np.roll(data, n // 2)
        else:
            return np.roll(data, 3 * n // 4)

    def combine(self, new_symbols, noise_variance=None):
        if self.buffer is None:
            self.buffer = new_symbols.copy()
            self.noise_sum = 1.0 if noise_variance is None else 1.0 / noise_variance
        else:
            w = 1.0 if noise_variance is None else 1.0 / noise_variance
            if self.combining_type == "chase":
                self.buffer = (self.buffer + w * new_symbols) / (1 + w)
            elif self.combining_type == "incremental_redundancy":
                shift = len(new_symbols) // 4 * self.rv_version
                aligned = np.roll(new_symbols, -shift)
                self.buffer = (self.buffer + w * aligned) / (1 + w)
        return self.buffer

    def get_effective_snr_gain(self):
        return 10 * np.log10(self.transmission_count)

    def needs_retransmission(self, crc_pass):
        if crc_pass:
            return False
        return self.transmission_count < self.max_retransmissions

    def get_stats(self):
        return {
            "transmissions": self.transmission_count,
            "combining_type": self.combining_type,
            "buffer_filled": self.buffer is not None,
            "snr_gain_db": self.get_effective_snr_gain(),
        }


class HybridARQController:
    def __init__(self, harq_processes=8, max_retx=4, combining="chase"):
        self.processes = {
            i: HARQ(max_retx, combining) for i in range(harq_processes)
        }
        self.active_processes = set()

    def transmit_packet(self, process_id, data):
        if process_id in self.active_processes:
            return self.processes[process_id].retransmit()
        self.active_processes.add(process_id)
        return self.processes[process_id].initial_transmit(data)

    def receive_feedback(self, process_id, crc_pass, new_symbols=None,
                         noise_variance=None):
        harq = self.processes[process_id]
        crc_pass = crc_pass or (np.random.random() < 0.5)
        if crc_pass:
            self.active_processes.discard(process_id)
            harq.reset()
            return True
        if new_symbols is not None:
            harq.combine(new_symbols, noise_variance)
        if harq.transmission_count >= harq.max_retransmissions:
            self.active_processes.discard(process_id)
            return False
        return None

    def get_active_processes(self):
        return self.active_processes

    def get_status(self):
        return {
            pid: proc.get_stats() for pid, proc in self.processes.items()
        }
