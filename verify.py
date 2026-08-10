"""Verification suite for the mini search engine.

Checks:
1. CSR matvec/matvec_T match dense numpy on random matrices
2. PageRank is a proper distribution (sums to 1, non-negative)
3. PageRank matches an independent dense power-iteration reference on
   random graphs, including graphs with sinks
4. known-graph correctness: two-page cycle -> equal rank; a chain ->
   sink page accumulates the most (damped) rank
5. damping in (0,1) and sink redistribution both required for
   convergence to a unique positive vector
6. search: topic pages rank above others for their topic query; the
   pagerank prior changes ordering (beta > 0 differs from beta = 0);
   the index handles queries with no matches
"""

from __future__ import annotations

import numpy as np

from crawl import synthetic_web
from pagerank import check_convergence, pagerank
from search import Index
from sparse import CSR


def check_sparse() -> None:
    print("[1] CSR matvec vs dense numpy")
    rng = np.random.default_rng(0)
    for trial in range(10):
        n = int(rng.integers(2, 30))
        entries = []
        A = np.zeros((n, n))
        for _ in range(int(rng.integers(0, n * n // 2))):
            i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
            v = float(rng.normal())
            entries.append((i, j, v))
            A[i, j] += v
        csr = CSR.from_coo(n, entries)
        x = rng.normal(size=n)
        assert np.allclose(csr.matvec(x), A @ x, atol=1e-10)
        assert np.allclose(csr.matvec_T(x), A.T @ x, atol=1e-10)
    print("    matvec and matvec_T match dense numpy (10 random matrices)")


def dense_pagerank_ref(adjacency, n, damping=0.85):
    """Independent implementation with dense numpy (no CSR, no
    special-casing) - the ground truth for testing."""
    outdeg = np.zeros(n)
    for (i, j) in adjacency:
        outdeg[i] += 1
    L = np.zeros((n, n))
    for (i, j) in adjacency:
        if outdeg[i] > 0:
            L[i, j] = 1.0 / outdeg[i]
    teleport = (1 - damping) / n
    pi = np.full(n, 1.0 / n)
    for _ in range(20000):
        new = L.T @ pi
        sink = pi[outdeg == 0].sum()
        new = damping * (new + sink / n) + teleport
        if np.abs(new - pi).sum() < 1e-14:
            return new
        pi = new
    raise RuntimeError("reference did not converge")


def check_pagerank() -> None:
    print("[2] PageRank properties + dense reference")
    rng = np.random.default_rng(1)
    for trial in range(20):
        n = int(rng.integers(2, 20))
        adj = []
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() < 0.2:
                    adj.append((i, j))
        pi = pagerank(adj, n)
        assert check_convergence(pi), "not a proper distribution"
        ref = dense_pagerank_ref(adj, n)
        assert np.allclose(pi, ref, atol=1e-10), f"trial {trial}"
    print("    matches dense reference on 20 random graphs")

    # known graphs
    pi = pagerank([(0, 1), (1, 0)], 2)
    assert np.allclose(pi, [0.5, 0.5], atol=1e-9), "2-cycle not symmetric"
    pi = pagerank([(0, 1), (1, 2)], 3)
    assert pi[2] > pi[1] > pi[0], "chain ranks not ordered"
    print("    2-cycle symmetric, chain monotone OK")

    # sinks don't trap everything
    pi = pagerank([(0, 1), (1, 2)], 3, damping=0.85)
    assert pi[2] < 0.95, "sink swallowed all rank"
    # convergence speed depends on damping (spectral radius = d)
    chain = [(0, 1), (1, 2), (2, 3), (3, 4)]
    iters_low = _count_iters(chain, 5, damping=0.1)
    iters_high = _count_iters(chain, 5, damping=0.99)
    assert iters_high > iters_low, "higher damping must converge slower"
    print(f"    damping 0.1 -> {iters_low} iters, 0.99 -> {iters_high} iters")

    # input validation: out-of-range / negative node ids must raise
    for bad in ([(0, 5)], [(-1, 0)], [(5, 0)]):
        try:
            pagerank(bad, 3)
            raise AssertionError(f"accepted invalid edges {bad}")
        except ValueError:
            pass
    print("    out-of-range and negative node ids rejected")


def _count_iters(adj, n, damping, tol=1e-10):
    import numpy as _np
    outdeg = _np.zeros(n)
    for (i, j) in adj:
        outdeg[i] += 1
    L = CSR.from_coo(n, [(i, j, 1.0 / outdeg[i]) for (i, j) in adj if outdeg[i] > 0])
    teleport = (1 - damping) / n
    pi = _np.full(n, 1.0 / n)
    for it in range(100000):
        new = L.matvec_T(pi)
        new += pi[outdeg == 0].sum() / n
        new = damping * new + teleport
        if _np.abs(new - pi).sum() < tol:
            return it + 1
        pi = new
    return -1


def check_search() -> None:
    print("[3] search ranking")
    pages, links = synthetic_web()
    idx = Index(pages, links)
    # a topic query must return that topic's pages first
    for query, topic in [("quantum wavefunction", "Quantum Mechanics"),
                         ("caesar legion", "Ancient Rome"),
                         ("neural backpropagation", "Deep Learning")]:
        top = idx.search(query, top_k=5)
        assert top, f"no results for {query}"
        for r in top[:3]:
            assert topic in r["title"], f"'{query}': {r['title']} not {topic}"
    print("    topic queries rank the right topics")

    # PageRank prior changes the ordering
    q = "quantum"
    idx0 = Index(pages, links, beta=0.0)
    idx1 = Index(pages, links, beta=1.0)
    assert idx0.search(q, 5) != idx1.search(q, 5), "beta must change ordering"
    print("    pagerank prior (beta) changes ordering")

    # no matches -> empty
    assert idx.search("zzzznotaword") == []
    print("    unmatched query returns nothing")

    # repeated query terms must not double-count
    one = idx.search("quantum", top_k=3)
    two = idx.search("quantum quantum", top_k=3)
    assert [r["id"] for r in one] == [r["id"] for r in two]
    assert all(abs(x["score"] - y["score"]) < 1e-12
               for x, y in zip(one, two))
    print("    duplicate query terms do not double-count")

    # negative top_k must raise
    try:
        idx.search("quantum", top_k=-1)
        raise AssertionError("negative top_k accepted")
    except ValueError:
        pass
    print("    negative top_k rejected")


def main() -> None:
    check_sparse()
    check_pagerank()
    check_search()
    print("\nAll verification passed.")


if __name__ == "__main__":
    main()
