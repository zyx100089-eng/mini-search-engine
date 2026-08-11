"""The personal-statement story for the mini search engine.

Walkthrough:
1. The random-surfer model and why PageRank is a Markov-chain
   stationary distribution.
2. CSR sparse matrix: why the web graph can't be dense.
3. Power iteration: convergence, damping as the convergence-rate dial.
4. The graph: communities, hubs, authorities, sinks (synthetic web).
5. Search: TF-IDF + PageRank blend, and a query where the two disagree.
"""

from __future__ import annotations

import time

import numpy as np

from crawl import synthetic_web
from pagerank import pagerank, rank_scores
from search import Index
from sparse import CSR


def demo_model() -> None:
    print("=" * 70)
    print("1. THE RANDOM SURFER")
    print("   PageRank = stationary distribution of a Markov chain:")
    print("   a surfer follows a random link with prob d, or jumps to")
    print("   a random page with prob 1-d.  Rank(page) = long-run")
    print("   fraction of visits.")
    print("=" * 70)
    # tiny example: 0 -> 1 -> 2 chain
    pi = pagerank([(0, 1), (1, 2)], 3, damping=0.85)
    print("   chain 0 -> 1 -> 2 (d = 0.85):")
    for i, r in enumerate(pi):
        print(f"     page {i}: rank {r:.4f}")
    print("   the sink (page 2) collects rank but does NOT get all of")
    print("   it: the teleport term guarantees every page keeps (1-d)/N.")


def demo_sparse() -> None:
    print("=" * 70)
    print("2. WHY CSR")
    print("   The web has ~10 links per page.  A dense N x N matrix")
    print("   would waste ~99.9999% of its memory on zeros; CSR stores")
    print("   only the nonzeros (3 arrays) and matvec is one pass.")
    print("=" * 70)
    pages, links = synthetic_web()
    outdeg = [0] * len(pages)
    for (i, j) in links:
        outdeg[i] += 1
    n = len(pages)
    nnz = len(links)
    print(f"   synthetic web: {n} pages, {nnz} links, "
          f"mean out-degree {nnz / n:.1f}")
    print(f"   dense: {n * n:7d} cells   CSR: {nnz:5d} entries "
          f"({100 * nnz / (n * n):.2f}% of dense)")
    L = CSR.from_coo(n, [(i, j, 1.0 / outdeg[i]) for (i, j) in links])
    print(f"   CSR matrix: {L.nnz()} nonzeros, density {L.density():.4f}")
    print(f"   ({nnz - L.nnz()} duplicate link pairs merged by from_coo)")


def demo_power_iteration() -> None:
    print("=" * 70)
    print("3. POWER ITERATION")
    print("   pi_new = d * L^T pi + (1-d)/N.  Converges linearly with")
    print("   rate d (the spectral radius).  Lower d = faster.")
    print("=" * 70)
    pages, links = synthetic_web()
    for d in (0.5, 0.85, 0.99):
        t0 = time.time()
        pi = pagerank(links, len(pages), damping=d, tol=1e-12)
        print(f"   d={d}: sum(pi) = {pi.sum():.12f} in {time.time() - t0:.3f}s")
    # count iterations
    def iters(d, tol=1e-10):
        outdeg = [0] * len(pages)
        for (i, j) in links:
            outdeg[i] += 1
        L = CSR.from_coo(len(pages), [(i, j, 1.0 / outdeg[i]) for (i, j) in links])
        teleport = (1 - d) / len(pages)
        pi = np.full(len(pages), 1.0 / len(pages))
        for it in range(100000):
            new = L.matvec_T(pi)
            new += pi[np.array(outdeg) == 0].sum() / len(pages)
            new = d * new + teleport
            if np.abs(new - pi).sum() < tol:
                return it + 1
            pi = new
        return -1
    print(f"   iterations to 1e-10: d=0.5 -> {iters(0.5)}, "
          f"d=0.85 -> {iters(0.85)}, d=0.99 -> {iters(0.99)}")


def demo_graph() -> None:
    print("=" * 70)
    print("4. THE GRAPH")
    print("   Synthetic web: 10 topic communities, 4 hubs, 3")
    print("   authorities, 3 sinks - the structures real webs have.")
    print("=" * 70)
    pages, links = synthetic_web()
    pi = pagerank(links, len(pages))
    top = rank_scores(pi, [p["title"] for p in pages])
    print("   top-5 by PageRank (authorities: many pages link to them):")
    for title, r in top[:5]:
        print(f"     {r:8.4f}  {title}")
    print("   bottom-3 (isolated or sink pages):")
    for title, r in top[-3:]:
        print(f"     {r:8.4f}  {title}")
    print("   note: rank flows INTO a page - hubs give rank away, so")
    print("   they rank low despite their out-degree.")


def demo_search() -> None:
    print("=" * 70)
    print("5. SEARCH: RELEVANCE x IMPORTANCE")
    print("   tf-idf finds pages that talk about the topic; PageRank")
    print("   finds pages the network endorses.  Search blends both.")
    print("=" * 70)
    pages, links = synthetic_web()
    idx = Index(pages, links)
    for query in ("quantum wavefunction", "caesar legion", "neural gradient"):
        print(f"\n   query: '{query}'")
        for r in idx.search(query, top_k=3):
            print(f"     {r['title']:35s} tfidf={r['tfidf']:.3f} "
                  f"rank={r['rank']:.4f} score={r['score']:.3f}")
            print(f"       {r['snippet']}")

    print("\n   a query where PageRank changes the winner:")
    for q in ("wavefunction measurement",):
        plain = Index(pages, links, beta=0.0).search(q, 3)
        blend = Index(pages, links, beta=1.0).search(q, 3)
        print(f"     '{q}' without pagerank: {[r['title'] for r in plain]}")
        print(f"     '{q}' with pagerank:    {[r['title'] for r in blend]}")


def main() -> None:
    demo_model()
    demo_sparse()
    demo_power_iteration()
    demo_graph()
    demo_search()
    print("\nDone.")


if __name__ == "__main__":
    main()
