"""Sparse matrices in CSR (Compressed Sparse Row) format.

The web graph has millions of nodes but only ~10 links per page on
average, so a dense n x n matrix would waste ~99.9999% of its memory.
CSR stores only the nonzeros: three arrays (row pointers, column
indices, values), and the workhorse operation — sparse matrix times
dense vector — is a single pass over the nonzeros.

Why CSR and not COO or CSC?  CSR's matvec reads rows sequentially,
which is cache-friendly and needs no sorting at runtime; CSC is the
transpose operation, which is what we need for the 'follow links'
direction (rank flows *along* links, so we multiply by the
transpose).  We provide both matvec directions.
"""

from __future__ import annotations

import numpy as np


class CSR:
    """Compressed sparse row matrix (n x n square, but could be any)."""

    def __init__(self, n: int, rows: list[int] | None = None,
                 cols: list[int] | None = None,
                 vals: list[float] | None = None):
        self.n = n
        if rows is None:
            rows, cols, vals = [], [], []
        self.rowptr = np.zeros(n + 1, dtype=np.int64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.vals = np.asarray(vals, dtype=np.float64)
        # build rowptr from the row of each entry
        for r in rows:
            self.rowptr[r + 1] += 1
        np.cumsum(self.rowptr, out=self.rowptr)

    @classmethod
    def from_coo(cls, n: int, entries: list[tuple[int, int, float]]) -> "CSR":
        """entries: list of (i, j, value).  Values for duplicate (i,j)
        are summed."""
        if not entries:
            return cls(n)
        # sort by (row, col) so duplicates are adjacent
        entries = sorted(entries, key=lambda e: (e[0], e[1]))
        rows = [e[0] for e in entries]
        cols = [e[1] for e in entries]
        vals = [e[2] for e in entries]
        # merge duplicates
        merged_rows, merged_cols, merged_vals = [], [], []
        i = 0
        while i < len(entries):
            r, c, v = rows[i], cols[i], vals[i]
            j = i + 1
            while j < len(entries) and rows[j] == r and cols[j] == c:
                v += vals[j]
                j += 1
            merged_rows.append(r)
            merged_cols.append(c)
            merged_vals.append(v)
            i = j
        return cls(n, merged_rows, merged_cols, merged_vals)

    @classmethod
    def identity(cls, n: int) -> "CSR":
        return cls.from_coo(n, [(i, i, 1.0) for i in range(n)])

    # ------------------------------------------------------------------

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """y = A @ x  (dense result)."""
        assert x.shape == (self.n,)
        y = np.zeros(self.n)
        for i in range(self.n):
            s = 0.0
            for k in range(self.rowptr[i], self.rowptr[i + 1]):
                s += self.vals[k] * x[self.cols[k]]
            y[i] = s
        return y

    def matvec_T(self, x: np.ndarray) -> np.ndarray:
        """y = A^T @ x.  This is the direction PageRank needs: rank
        flows from page i to each outlink j, so we gather, for each j,
        the sum of x[i] * A[i][j] over all i that link to j."""
        assert x.shape == (self.n,)
        y = np.zeros(self.n)
        for i in range(self.n):
            xi = x[i]
            if xi == 0.0:
                continue
            for k in range(self.rowptr[i], self.rowptr[i + 1]):
                y[self.cols[k]] += self.vals[k] * xi
        return y

    def row_sums(self) -> np.ndarray:
        out = np.zeros(self.n)
        for i in range(self.n):
            out[i] = self.vals[self.rowptr[i]:self.rowptr[i + 1]].sum()
        return out

    def nnz(self) -> int:
        return int(self.rowptr[-1])

    def density(self) -> float:
        return self.nnz() / (self.n * self.n)

    def to_dense(self) -> np.ndarray:
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            for k in range(self.rowptr[i], self.rowptr[i + 1]):
                A[i, self.cols[k]] = self.vals[k]
        return A
