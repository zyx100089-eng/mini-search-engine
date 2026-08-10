"""PageRank: the random-surfer Markov chain, solved by power iteration.

Model.  A surfer either follows a random outgoing link of the current
page (probability d, the damping factor) or jumps to a uniformly
random page (probability 1-d).  PageRank is the stationary
distribution of this chain: the fraction of time the surfer spends on
each page.

Math.  With N pages and link matrix L (L[i][j] = 1/outdegree(i) if i
links to j), the update is

    pi_new[j] = (1-d)/N + d * sum_{i -> j} pi[i]/outdegree(i)

In matrix form pi_new = (1-d)/N * 1 + d * (L^T) pi.  The damping term
is not just a hack: it fixes both problems of the naive chain:

  1. sink pages (outdegree 0) would trap all rank;
  2. the chain would not be irreducible, so the stationary
     distribution need not be unique.

With 0 < d < 1 every page has a positive self-probability (1-d)/N, so
the chain is irreducible and aperiodic, and the stationary vector is
unique and positive.  Power iteration converges linearly with rate d
(the spectral gap is at least 1-d).

Implementation.  We use the CSR transpose-matvec (rank flows along
links), add the teleportation constant in O(1) per iteration instead
of materialising a dense matrix, and handle sinks by redistributing
their mass to all pages (the standard fix: a surfer on a sink jumps
uniformly).
"""

from __future__ import annotations

import numpy as np

from sparse import CSR


def pagerank(adjacency: list[tuple[int, int]], n: int, *,
             damping: float = 0.85, tol: float = 1e-12,
             max_iters: int = 2000, verbose: bool = False) -> np.ndarray:
    """PageRank by power iteration.

    adjacency: list of (i, j) directed edges (i links to j).
    Returns the rank vector pi (n,), normalised so sum(pi) == 1.
    """
    if n <= 0:
        raise ValueError("need at least one page")
    if not (0.0 < damping < 1.0):
        raise ValueError("damping must be strictly between 0 and 1")
    for (i, j) in adjacency:
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"edge ({i}, {j}) out of range for {n} pages")

    # out-degrees
    outdeg = np.zeros(n, dtype=np.int64)
    for (i, j) in adjacency:
        outdeg[i] += 1

    # build the stochastic matrix L: L[i][j] = 1/outdeg(i) for i -> j
    entries: list[tuple[int, int, float]] = []
    for (i, j) in adjacency:
        if outdeg[i] > 0:
            entries.append((i, j, 1.0 / outdeg[i]))
    L = CSR.from_coo(n, entries)

    teleport = (1.0 - damping) / n
    pi = np.full(n, 1.0 / n)

    for it in range(max_iters):
        new = L.matvec_T(pi)
        # sink redistribution: mass on sink pages is spread uniformly
        sink_mass = pi[outdeg == 0].sum()
        new += sink_mass / n
        new = damping * new + teleport
        diff = np.abs(new - pi).sum()  # L1 distance (sum of abs)
        pi = new
        if diff < tol:
            if verbose:
                print(f"  converged in {it + 1} iterations (L1 diff {diff:.2e})")
            return pi
    raise RuntimeError(f"power iteration did not converge in {max_iters} iters")


def check_convergence(pi: np.ndarray, tol: float = 1e-9) -> bool:
    """Sanity: the vector is a proper probability distribution."""
    return abs(pi.sum() - 1.0) < tol and np.all(pi >= 0)


def rank_scores(pi: np.ndarray, names: list[str]) -> list[tuple[str, float]]:
    """Sort pages by rank, descending."""
    order = np.argsort(-pi)
    return [(names[i], float(pi[i])) for i in order]
