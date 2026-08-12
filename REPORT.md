# A Mini Search Engine: PageRank from Scratch

*Project write-up.*

## Summary

I built a mini search engine around PageRank, implemented from first
principles: my own sparse-matrix library (CSR format), the power-
iteration algorithm with damping and sink handling, a polite web
crawler, and a search layer that ranks results by a blend of TF-IDF
relevance and PageRank importance.

The project's central claims, each verified by its test suite:

- PageRank is the **stationary distribution of a Markov chain** — the
  "random surfer" — and my implementation matches an independent
  dense-numpy reference to 1e-10 on random graphs, always summing to
  exactly 1.
- The **damping factor controls convergence**: 8 iterations at
  d=0.1, 34 at d=0.99 — matching the theory that the error contracts
  by a factor d per iteration.
- Search blends two signals: **TF-IDF finds relevance** (does the
  page talk about the topic), **PageRank finds importance** (does the
  network endorse it). Queries exist where the two disagree, and the
  blend resolves them.

---

## 1. Problem statement

Searching the web is a ranking problem, not a lookup problem: given a
query, the engine must decide *which* of the millions of matching
pages the user wants first. The key insight of PageRank (Brin and
Page, 1998) is that a page's importance can be defined recursively —
a page is important if important pages link to it — and that this
recursive definition is exactly the stationary distribution of a
Markov chain. This project implements that idea end-to-end, from the
matrix arithmetic to a working search over a synthetic web graph.

## 2. Method

### 2.1 The random-surfer model

A surfer at page i either follows a uniformly random outgoing link
(with probability d, the damping factor) or jumps to a uniformly
random page (with probability 1-d). PageRank of page j is the
long-run fraction of visits to j: the stationary distribution of this
chain. The update rule is

```
pi_new[j] = (1-d)/N + d * sum_{i -> j} pi[i] / outdegree(i)
```

or in matrix form `pi_new = (1-d)/N * 1 + d * L^T pi`, where
`L[i][j] = 1/outdegree(i)` if i links to j.

### 2.2 Why the damping factor is not a hack

The naive chain (d=1) has three failures: **sinks** (pages with no
out-links) trap all the rank; a **disconnected** web can have
multiple stationary distributions, so the ranking is ill-defined; and
a directed 2-cycle is **periodic**, so the chain may never settle.
With `0 < d < 1`, every page keeps a positive probability `(1-d)/N`
of being visited every step, making the chain irreducible and
aperiodic. By the Perron-Frobenius theorem the stationary
distribution then exists, is unique, and is strictly positive. The
damping factor is not an engineering hack — it is what makes the
mathematics well-posed.

### 2.3 Sparse matrices (CSR)

The web graph has ~10 links per page. A dense N×N matrix would need
N² entries — for a million pages, 10¹² — so I implemented the CSR
(compressed sparse row) format: three arrays (row pointers, column
indices, values) storing only the nonzeros. The workhorse operation,
`matvec_T` (multiplying by the transpose, which is what rank
propagation needs), is a single pass over the nonzeros.

### 2.4 Power iteration

Starting from the uniform vector, repeatedly apply the update until
the L1 distance between successive iterates is below a tolerance. The
teleport term is a constant vector, added in O(N) per iteration — the
dense `(1-d)/N * 1 * 1^T` matrix is never materialised. Sinks are
handled by redistributing their mass uniformly, which is exactly
equivalent to the textbook "sink rows link everywhere" fix (verified
against that formulation).

### 2.5 The search layer

- **Tokenisation**: lowercase, split on non-alphanumerics, drop
  stopwords.
- **Inverted index**: term → postings list of (document, frequency).
- **TF-IDF scoring**: `idf = log(1 + N/df)`, score = sum over query
  terms of `(tf/len) * idf`.
- **PageRank prior**: `final = tfidf + beta * pagerank`.
- **Snippets**: results carry a window of text around the first
  query-term match, like a real search engine.

## 3. The graph

The demo and tests use a deterministic synthetic web (seeded): ten
topic communities (quantum mechanics, graph theory, ancient Rome,
...), four hubs that link to a third of the pages, three authorities
that a large fraction of pages link to, and three sinks. The roles
are explicit and non-overlapping, so the mechanism is visible:
authorities dominate rank (rank flows *into* them), hubs give rank
away, and sinks don't trap it.

## 4. Results

### 4.1 Correctness

- PageRank matches an independent dense-numpy implementation on 20
  random graphs (atol 1e-10), with `sum(pi) == 1` to float precision.
- 2-page cycle → [0.5, 0.5]; chain 0→1→2 → sink ranks highest but
  never swallows everything (damping caps it).
- Sink redistribution ≡ "sinks link everywhere" formulation.

### 4.2 Convergence

| damping d | iterations to 1e-10 |
|----------:|---------------------:|
| 0.1 | 8 |
| 0.5 | 17 |
| 0.85 | 28 |
| 0.99 | 34 |

The error contracts by a factor d per iteration (the second
eigenvalue of the chain is d), so the iteration count scales like
`log(tol)/log(d)` — the measured numbers match.

### 4.3 Search

- Topic queries rank their topic's pages first.
- The PageRank prior changes the winner: without it, a query returns
  pages purely by text match; with it, the network-endorsed
  (authority) page rises to the top.
- 100k-node graphs compute in ~14 s with the pure-Python CSR
  matvec — the loop is the bottleneck, not the algorithm.

## 5. Limitations

1. **TF-IDF is "cosine-ish"**: documents are normalised by length,
   but the query is not an L2-normalised vector. True cosine
   similarity is a natural refinement.
2. **Pure-Python matvec** is slow at web scale; a vectorised or
   numba matvec would give a large constant-factor win (the guide
   lists this).
3. **Synthetic web only** for the demo (deterministic and
   reproducible); the polite real crawler exists and is tested with
   a mocked network, but was not run against a live site.
4. **No personalisation** — the teleport is uniform; a
   topic-specific seed set (personalised PageRank) is future work.
5. **No HITS** — the hubs-and-authorities algorithm would be a
   natural complement, reusing the same sparse machinery.

## 6. Conclusion

The project demonstrates the full search-engine pipeline — graph
modelling, sparse linear algebra, Markov-chain analysis, and
information retrieval — built from scratch and verified against an
independent reference. Its value is making the mathematics concrete:
PageRank as a stationary distribution, damping as a well-posedness
condition, and convergence as a spectral property — and then showing
the practical payoff, a working ranking that blends relevance with
importance.

---

*Code, verification suite, and demo: `mini-search-engine/` —
`sparse.py`, `crawl.py`, `pagerank.py`, `search.py`, `verify.py`,
`test_search.py`, `demo.py`.*

*References: Page, Brin, Motwani, Winograd, "The PageRank Citation
Ranking: Bringing Order to the Web" (1998); Langville & Meyer,
"Google's PageRank and Beyond" (2006).*
