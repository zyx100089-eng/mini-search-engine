# Mini Search Engine

A search engine implemented from scratch: a web crawler, a sparse (CSR) representation of the web graph, PageRank via power iteration, and a TF-IDF × PageRank ranking blend.

## Background

PageRank models the web as a Markov chain: a random surfer follows outbound links uniformly at random, and occasionally teleports to an arbitrary page. The rank of each page is the stationary distribution of this chain. This project implements that pipeline on both a deterministic synthetic web and real crawled networks, using a sparse matrix representation throughout.

## Implementation

### Sparse linear algebra

The web graph is stored as a CSR matrix (`sparse.py`). All computation is over the nonzeros: matvec and the transposed matvec that PageRank's power iteration needs are single passes, with no dense matrices anywhere. The implementation is verified against a dense NumPy reference.

### PageRank

Power iteration with damping factor d and teleportation, with L1 convergence criterion and sink handling: mass on out-degree-0 pages is redistributed uniformly. The damping factor is what makes the chain irreducible and aperiodic (teleportation keeps every page reachable), guaranteeing a unique stationary distribution.

### Crawling

A polite rate-limited crawler (`crawl.py`) based on `html.parser`, with deferred link resolution (links to queued pages are kept). A deterministic `synthetic_web()` generates a web-like graph with communities, hubs, authorities, and sinks.

### Search

An inverted index with tokenisation and stopwords, TF-IDF scoring, and a PageRank prior blend controlled by a `beta` parameter. Results include query-term snippets.

## Files

| File | Purpose |
|---|---|
| `sparse.py` | CSR matrix: from_coo, matvec, matvec_T, row sums, density |
| `crawl.py` | `synthetic_web()` (deterministic web-like graph) and `PoliteCrawler` for real crawling |
| `pagerank.py` | Power iteration with damping, sink redistribution, convergence diagnostics |
| `search.py` | Inverted index, TF-IDF + PageRank blend (`beta`), query-term snippets |
| `verify.py` | CSR vs dense NumPy, PageRank vs independent dense reference, distribution checks, known-graph checks, damping convergence-rate check, search-ranking checks |
| `test_search.py` | Unit test suite (29 tests, including the crawler with a mocked network) |
| `demo.py` | Demonstration: random surfer, CSR, power iteration, the graph, search blending |

## Results

- PageRank matches an independent dense NumPy implementation on 20 random graphs (atol 1e-10); sum(pi) == 1 exactly to float precision.
- Convergence rate follows the spectral radius: d=0.1 converges in 9 iterations, d=0.99 in 61 iterations (1e-10 tolerance).
- On the synthetic web (40 pages, 120 links, mean out-degree 3.0): authorities (in-degree 18-23) dominate rank; hubs (out-degree 13-14) give rank away; sinks do not trap it.
- Search: topic queries rank their topic's pages first; the PageRank prior visibly changes the winner (e.g. 'caesar legion' promotes the Ancient-Rome authority page over its siblings).

## Running

```
python3 -m pytest test_search.py -q   # unit tests
python3 verify.py   # full verification
python3 demo.py     # demonstration
```

## References

- Page, Brin, Motwani, Winograd, *The PageRank Citation Ranking: Bringing Order to the Web* (1998)
- Langville & Meyer, *Google's PageRank and Beyond: The Science of Search Engine Rankings* (2006)
