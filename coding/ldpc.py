import numpy as np
from scipy.sparse import csr_matrix


class LDPCEncoder:
    """Simple repeat-accumulate code encoder (demonstrates coding gain)."""

    def __init__(self, n=128, rate=0.5, seed=None):
        self.n = n
        self.rate = rate
        self.k = int(n * rate)
        self.m = n - self.k
        self.rng = np.random.default_rng(seed)
        self._build_h()

    def _build_h(self):
        n, m, k = self.n, self.m, self.k
        H = np.zeros((m, n), dtype=int)
        for j in range(k):
            c = self.rng.integers(1, 6)
            i = (j * c) % m
            for t in range(3):
                H[(i + t) % m, j] = 1
        for i in range(m):
            H[i, k + i] = 1
            if i > 0:
                H[i - 1, k + i] = 1
        self.H = csr_matrix(H)

    def encode(self, bits):
        if len(bits) > self.k:
            bits = bits[:self.k]
        d = np.array(bits, dtype=int)
        Hd = self.H.toarray()
        s = Hd[:, :self.k] @ d % 2
        p = np.zeros(self.m, dtype=int)
        p[self.m - 1] = s[self.m - 1]
        for i in range(self.m - 2, -1, -1):
            p[i] = s[i] ^ p[i + 1]
        return np.concatenate([d, p]).astype(np.int8)

    def get_H(self):
        return self.H


class LDPCDecoder:
    """Min-sum belief propagation decoder."""

    def __init__(self, H, max_iter=50):
        H = H.tocsr() if hasattr(H, 'tocsr') else csr_matrix(H)
        self.m, self.n = H.shape
        self.max_iter = max_iter
        self._build_tables(H)

    def _build_tables(self, H):
        self.cn = [list(H[i, :].nonzero()[1]) for i in range(self.m)]
        self.vn = [list(H[:, j].nonzero()[0]) for j in range(self.n)]
        self.cn_idx = {}
        self.vn_idx = {}
        for i in range(self.m):
            self.cn_idx[i] = {j: idx for idx, j in enumerate(self.cn[i])}
        for j in range(self.n):
            self.vn_idx[j] = {i: idx for idx, i in enumerate(self.vn[j])}

    def decode_bp(self, llrs, max_iter=None):
        if max_iter is None:
            max_iter = self.max_iter
        llr = np.array(llrs, dtype=float).flatten()
        m, n = self.m, self.n
        V = [np.full(len(self.vn[j]), llr[j], dtype=float) for j in range(n)]
        C = [np.zeros(len(self.cn[i]), dtype=float) for i in range(m)]

        for _ in range(max_iter):
            for i in range(m):
                cn = self.cn[i]
                k = len(cn)
                if k < 2:
                    continue
                vals = np.array([V[j][self.vn_idx[j][i]] for j in cn])
                signs = np.sign(vals)
                mags = np.abs(vals)
                total_sign = np.prod(signs)
                sorted_idx = np.argsort(mags)
                min1 = mags[sorted_idx[0]]
                min2 = mags[sorted_idx[1]] if k > 1 else min1
                for idx, j in enumerate(cn):
                    sig = total_sign * signs[idx]
                    mmin = min2 if idx == sorted_idx[0] else min1
                    C[i][idx] = sig * mmin * 0.75

            for j in range(n):
                total = llr[j]
                for idx, i in enumerate(self.vn[j]):
                    total += C[i][self.cn_idx[i][j]]
                for idx, i in enumerate(self.vn[j]):
                    V[j][idx] = total - C[i][self.cn_idx[i][j]]

            parity_ok = True
            for i in range(m):
                cn = self.cn[i]
                if cn:
                    s = 0
                    for j in cn:
                        t = llr[j]
                        for idx2, i2 in enumerate(self.vn[j]):
                            t += C[i2][self.cn_idx[i2][j]]
                        s ^= (1 if t <= 0 else 0)
                    if s:
                        parity_ok = False
                        break
            if parity_ok:
                break

        decoded = np.zeros(n, dtype=np.int8)
        for j in range(n):
            total = llr[j]
            for idx, i in enumerate(self.vn[j]):
                total += C[i][self.cn_idx[i][j]]
            decoded[j] = 0 if total > 0 else 1
        return decoded

    def decode_hard(self, bits, max_iter=None):
        return self.decode_bp(np.where(np.array(bits) == 0, 5.0, -5.0), max_iter)

    def decode_soft(self, syms, nvar, max_iter=None):
        return self.decode_bp(-2.0 * np.array(syms) / (nvar + 1e-12), max_iter)
