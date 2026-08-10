# Mini Search Engine: PageRank from Scratch

Graph + sparse linear algebra project: crawl a real network (or use a
deterministic synthetic web), build the web graph as a CSR sparse
matrix, compute PageRank with my own power iteration, and rank search
results by a TF-IDF x PageRank blend.

## The maths

- The web as a Markov chain: the random-surfer model
- PageRank = stationary distribution of the chain
- Existence and uniqueness: irreducibility, aperiodicity, and how the
  damping factor fixes both (teleportation keeps every page reachable)
- Why power iteration converges: spectral radius = d, so the error
  shrinks by a factor d per iteration
- Sink handling: mass on out-degree-0 pages is redistributed uniformly

## The CS

- Sparse matrix/vector formats (CSR) — no dense matrices; matvec is a
  single pass over the nonzeros
- Power iteration with teleportation and L1 convergence criterion
- Polite rate-limited crawler (html.parser-based) for real networks,
  with deferred link resolution (links to queued pages are kept)
- Inverted index, tokenisation, stopwords, TF-IDF scoring, PageRank
  prior blend, query-term snippets in results

## Files

- `sparse.py` - CSR matrix: from_coo, matvec, matvec_T (the direction
  PageRank needs), row sums, density
- `crawl.py` - `synthetic_web()` (deterministic web-like graph:
  communities, hubs, authorities, sinks) and `PoliteCrawler` for real
  crawling
- `pagerank.py` - power iteration with damping, sink redistribution,
  convergence diagnostics
- `search.py` - inverted index, TF-IDF + PageRank blend (`beta`),
  query-term snippets
- `verify.py` - CSR vs dense numpy, PageRank vs an independent dense
  reference, distribution checks, known-graph checks, damping
  convergence-rate check, search-ranking checks
- `test_search.py` - pytest unit suite (29 tests, including the
  crawler with a mocked network)
- `demo.py` - the story: random surfer, CSR, power iteration, the
  graph, search blending

## Running

```
python3 -m pytest test_search.py -q   # fast unit tests
python3 verify.py   # full verification
python3 demo.py     # personal-statement walkthrough
```

## Results

- PageRank matches an independent dense numpy implementation on 20
  random graphs (atol 1e-10); sum(pi) == 1 exactly to float precision
- Convergence rate follows the spectral radius: d=0.1 -> 9 iterations,
  d=0.99 -> 61 iterations (1e-10 tolerance)
- The synthetic web (40 pages, 86 links): authorities (in-degree ~20)
  dominate rank; hubs (out-degree ~13) give rank away; sinks don't
  trap it
- Search: topic queries rank their topic's pages first; the PageRank
  prior visibly changes the winner (e.g. 'caesar legion' promotes the
  Ancient-Rome authority page over its siblings)

## References

- Page, Brin, Motwani, Winograd, *The PageRank Citation Ranking:
  Bringing Order to the Web* (1998)
- Langville & Meyer, *Google's PageRank and Beyond: The Science of
  Search Engine Rankings* (2006)
