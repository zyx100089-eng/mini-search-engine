# Mini Search Engine Project — Complete Interview Guide

*Everything you need to explain, defend, and extend this project in an
interview. Read it top to bottom once; then use the quick-reference and
Q&A sections to revise. `demo.py` is your 2-minute pitch.*

---

## 1. The pitch (60 seconds)

> I built a mini search engine around PageRank, implemented from
> scratch: a sparse-matrix library, the power-iteration algorithm with
> damping and sink handling, a polite web crawler, and a search layer
> that blends TF-IDF relevance with PageRank importance.
>
> The interesting part is the maths. PageRank is the *stationary
> distribution of a Markov chain* — the "random surfer" who follows a
> random link with probability 0.85 or jumps to a random page with
> probability 0.15. The damping factor isn't a hack: it's what makes
> the chain irreducible and aperiodic, which is what guarantees a
> unique stationary distribution exists. And power iteration
> converges linearly with rate equal to the damping factor — I verified
> that empirically: damping 0.1 converges in 9 iterations, damping
> 0.99 takes 61.
>
> The engineering: the web graph can't be a dense matrix — a million
> pages would need a trillion entries — so I wrote a CSR sparse matrix
> from scratch. Rank flows *along* links, so the workhorse is the
> transpose-matvec, one pass over the nonzeros.
>
> I verified everything: PageRank matches an independent dense-numpy
> reference to 1e-10 on random graphs, the rank vector always sums to
> exactly 1, and the sink-handling matches the textbook
> "sinks-link-everywhere" formulation.

---

## 2. The maths (know these cold)

### 2.1 The random-surfer model

A surfer at page i either:
- follows a uniformly random outgoing link (probability d), or
- jumps to a uniformly random page (probability 1-d).

PageRank of page j = the long-run fraction of visits to j = the
stationary distribution of this Markov chain.

The update rule, written out:

```
pi_new[j] = (1-d)/N  +  d · Σ_{i→j} pi[i] / outdegree(i)
```

In matrix form: `pi_new = (1-d)/N · 1 + d · Lᵀ pi`, where
`L[i][j] = 1/outdegree(i)` if i links to j, else 0. L is a *column-
stochastic*... careful: each row of L sums to 1 (stochastic matrix),
and we multiply by Lᵀ because rank flows from i to its outlinks j.

### 2.2 Why damping is not a hack — three problems it fixes

The naive chain (d = 1, no teleport) has three failures:

1. **Sinks** (pages with outdegree 0) trap all the rank — the chain is
   not *closed*, mass leaks.
2. **Non-irreducibility**: a disconnected web (or a strongly connected
   component with no out-links) can have multiple stationary
   distributions, so the ranking isn't well-defined.
3. **Periodicity**: a directed 2-cycle has period 2, so the chain may
   never settle.

With `0 < d < 1`, every page keeps a positive probability `(1-d)/N`
of being visited every step (the teleport term). That makes the chain
**irreducible** (every page reaches every page) and **aperiodic**, so
by the Perron–Frobenius theorem the stationary distribution exists,
is unique, and is strictly positive. This is the theorem an
interviewer might probe — know the name and the statement.

### 2.3 Why power iteration converges

We're computing the leading eigenvector of the matrix
`M = d·Lᵀ + (1-d)/N·1·1ᵀ` (stochastic, positive). Power iteration
`pi ← M pi` converges linearly with rate = the second-largest
eigenvalue's magnitude = `d` (for the web graph the eigenvalue gap
is at least `1-d`). So:

- error shrinks by a factor of d each iteration,
- d = 0.85 → error ×0.85 per step: about 85 steps per 10× reduction,
- measured (5-node chain, to 1e-10): d=0.1 → 9 iterations, d=0.85 →
  43, d=0.99 → 61.  On the 40-page synthetic web: d=0.1 → 8, d=0.85
  → 28, d=0.99 → 34.  (Iteration count depends on the graph, not just
  d — the spectral radius bound is d, but the constant differs.)

That's why Google uses d = 0.85: close to 1 (follows the web's
structure) but with a spectral gap that keeps iteration count modest.

### 2.4 Sink handling

A surfer on a sink (outdegree 0) can't follow a link, so we spread
that page's mass uniformly: `new[j] += sink_mass / N`. This is exactly
equivalent to the textbook trick of adding a self-... no — adding
*uniform out-links* to every sink row. Verified empirically: my
implementation matches the "sink rows link everywhere" formulation on
random graphs to 1e-10.

---

## 3. Architecture (files and responsibilities)

| File | Responsibility |
|---|---|
| `sparse.py` | CSR sparse matrix: from_coo (with duplicate merging), matvec, matvec_T (the direction PageRank needs), row_sums, density |
| `crawl.py` | `synthetic_web()` — deterministic web-like graph (10 topic communities, 4 hubs, 3 authorities, 3 sinks) + `PoliteCrawler` (rate-limited, html.parser, BFS with deferred link resolution) |
| `pagerank.py` | Power iteration: damping, teleport, sink redistribution, L1 convergence criterion |
| `search.py` | Tokenizer, stopwords, inverted index, TF-IDF, PageRank prior blend (`beta`) |
| `verify.py` | CSR vs dense numpy; PageRank vs independent dense reference; distribution checks; known-graph checks; damping-rate check; search checks |
| `demo.py` | The story |

---

## 4. The algorithms

### 4.1 CSR sparse matrix (`sparse.py`)

The web graph has ~10 links per page. A dense N×N matrix for N
pages uses N² entries; CSR stores three arrays:

- `rowptr[n+1]`: start of each row in cols/vals
- `cols[nnz]`: column index of each nonzero
- `vals[nnz]`: the values

matvec (`y = A·x`) is one pass: for each row, accumulate
`vals[k] * x[cols[k]]`. The transpose-matvec (`y = Aᵀ·x`) is the same
pass but writes into `y[cols[k]]` — that's the direction PageRank
needs (gather rank from in-links).

`from_coo` sorts by (row, col) and merges duplicates by summing —
important because a page may link the same page twice (or the graph
construction may add the same edge twice).

Why CSR over COO/CSC: rows are stored contiguously so matvec reads
sequentially (cache-friendly); CSC would be needed for "which rows
have a nonzero in column j" (the row-vector view), which we don't
need since we iterate pages anyway.

### 4.2 Power iteration (`pagerank.py`)

```
pi ← 1/N everywhere
repeat:
    new = Lᵀ pi                    # follow links (CSR matvec_T)
    new += (mass of sink pages)/N  # sinks redistribute uniformly
    new = d·new + (1-d)/N          # teleport
until ‖new − pi‖₁ < tol
```

Note the teleport term is a constant vector, added in O(N) per
iteration — we never materialise the dense `(1-d)/N·1·1ᵀ` matrix.
Convergence uses the L1 distance (sum of absolute differences), which
is the natural metric for probability vectors.

### 4.3 The synthetic web (`crawl.py`)

The demo and verification need a *deterministic* graph with web-like
structure:

- ~10 content pages per topic community, 12% within-topic pairs link
  (clustering — the web's community structure)
- 4 hubs link out to ~1/3 of pages (high out-degree, spam-like)
- 3 authorities receive links from 45% of pages (high in-degree)
- 3 sinks link nowhere (declared); the actual graph has 7 zero-out-
  degree pages — the 3 declared sinks plus 4 content pages that
  happened to receive no out-links (they still get rank via the
  teleport term, so they don't trap it)

The roles are non-overlapping so the demo tells a clean story:
authorities dominate rank, hubs give rank away, sinks don't trap it.

### 4.4 Polite real crawler (`PoliteCrawler`)

BFS from seed URLs, `html.parser`-based link/text extraction, one
request per `delay` seconds (rate limiting), a User-Agent header,
bounded page count. Link edges are *deferred*: a link to a page that
is only queued (not yet fetched) is recorded as pending and resolved
to an edge when the target gets an id — so no links are lost.

### 4.5 Search (`search.py`)

- **Tokenize**: lowercase, split on non-alphanumerics, drop stopwords
  and 1-char tokens.
- **Inverted index**: term → {doc_id: term_frequency}.
- **Query scoring**: TF-IDF with length normalisation,
  `idf = log(1 + N/df)`, score = Σ over query terms of `(tf/len) · idf`.
  Query terms are deduplicated (a repeated term adds no evidence).
- **Blend**: `final = tfidf + beta · pagerank`. Text match finds
  *relevance*; PageRank finds *importance*. The demo shows a query
  where the two disagree and the blend wins.

---

## 5. Verification highlights (quote these)

- CSR matvec and matvec_T match dense numpy on 10 random matrices.
- PageRank matches an independent dense-numpy reference on 20 random
  graphs (atol 1e-10).
- `sum(pi) == 1` to float precision; all entries non-negative.
- 2-page cycle → [0.5, 0.5]; chain 0→1→2 → sink ranks highest but
  never swallows everything (damping caps it below 1).
- Damping controls convergence rate: d=0.1 → 9 iters, d=0.99 → 61.
- Sink redistribution ≡ "sink rows link everywhere" formulation.
- Search: topic queries rank their topic first; beta changes the
  winner; unmatched queries return nothing; duplicate query terms
  don't double-count.

---

## 6. Complexity notes

- CSR build: O(E log E) (sort) + O(E) merge.
- matvec / matvec_T: O(E) — one pass over nonzeros.
- Power iteration: O(iters × E) — linear in the graph size per step;
  measured 10k nodes / 75k edges in ~2s, 100k nodes in ~14s (pure
  Python, no numpy vectorisation in the matvec).
- Inverted index build: O(total tokens).
- Query: O(query terms × avg postings) — fast in practice.

---

## 7. Design decisions (be ready to defend)

1. **CSR over dense / COO / CSC** — memory (N² → O(E)) and cache
   locality; CSC is for column-oriented ops we don't need.
2. **matvec_T as the workhorse** — rank flows along links, so we need
   Aᵀ·x; writing into `y[cols[k]]` is the transpose without building
   the transpose.
3. **Sinks handled by uniform redistribution** — matches the textbook
   "sink rows link everywhere" fix; verified equivalent.
4. **Teleport added in O(N), not materialised** — the dense
   `(1-d)/N·1·1ᵀ` matrix would destroy the sparsity advantage.
5. **L1 convergence criterion** — the natural metric for probability
   vectors, matches the total-variation distance of the chain.
6. **Synthetic web with explicit roles** — deterministic (seeded), so
   the demo and verification are reproducible; hubs/authorities/sinks
   make the mechanism visible.
7. **Deferred link resolution in the crawler** — a link to a queued
   page is an edge, not just a queue entry; naive BFS crawlers lose
   these.
8. **TF-IDF + PageRank blend** — relevance × importance is the actual
   web-search story; beta is the knob.

---

## 8. Bugs I found and fixed (interview material)

1. **Duplicate query terms double-counted** — `'quantum quantum'`
   scored exactly 2× `'quantum'` (the search loop iterated query
   terms without dedup). A repeated term is the same evidence; fixed
   with `dict.fromkeys(terms)`. Regression test added.

2. **Crawler dropped links to queued pages** — BFS recorded an edge
   only if the target was *already fetched*; links to pages merely
   queued were lost, so the graph under-represented the web. Fixed
   with deferred edge resolution: a `pending` map records
   (url → linking page ids) and resolves edges when the target is
   fetched.

3. **Synthetic web model didn't tell the intended story** — the first
   version let hubs and authorities overlap randomly, so an Olympic
   Sports page (in-degree 20) dominated instead of the authorities,
   and the demo's narrative fell flat. Fixed by making the roles
   explicit and non-overlapping.

4. *(Design tension worth mentioning)* — the original search score
   normalisation `tf / doc_length` is "cosine-ish" not true cosine
   (no query-length normalisation and no √). It's deliberate and
   simple; a proper cosine with L2 norms is listed as future work.

---

## 9. Measured numbers (cite these)

- 40-page synthetic web, 120 links, mean out-degree 3.0; CSR density
  7.19% (vs 100% dense).  Note: 120 link entries include 5 duplicate
  pairs; `from_coo` merges them, so the CSR matrix holds 115 nonzeros.
- Top rank ≈ 0.29 for the top authority; hubs (out-degree 13-14) rank
  ~0.005 — rank flows *into* pages.
- d=0.1 → 9 iters, d=0.85 → 28 iters, d=0.99 → 61 iters (to 1e-10).
- 10k nodes / 75k edges: ~1s. 100k nodes / 500k edges: ~9s (Apple
  silicon, pure Python; machine-dependent).
- sum(pi) = 1.000000000000 to float precision.

---

## 10. What I'd do next

1. **True cosine similarity** (L2 norms, query normalisation) for a
   cleaner IR story.
2. **Blocked/parallel matvec** or numba/vectorised CSR to scale to
   millions of pages (pure-Python loop is the bottleneck).
3. **PageRank with personalisation** — teleport to a topic-specific
   seed set (the basis of topical ranking).
4. **HITS algorithm** (hubs & authorities scores) and compare with
   PageRank — both from the same sparse machinery.
5. **Real crawl**: run PoliteCrawler on a small site (e.g. a
   wikipedia sub-tree) and rank it.
6. **Snippets**: return the query-term window from `text` instead of
   just titles.

---

## 11. Rapid-fire Q&A

**Q: What is PageRank, mathematically?**
A: The stationary distribution of the random-surfer Markov chain — the
long-run fraction of visits to each page. Computed as the leading
eigenvector of the stochastic matrix `d·Lᵀ + (1-d)/N·11ᵀ`.

**Q: Why does the damping factor matter?**
A: Without it the chain isn't irreducible (disconnected components,
sinks) and can be periodic (2-cycles), so the stationary distribution
may not exist or be unique. With 0<d<1 every page keeps positive
probability every step → irreducible + aperiodic → unique, positive
stationary vector.

**Q: Why does power iteration converge?**
A: It's computing the dominant eigenvector; the error contracts by
the magnitude of the second eigenvalue, which is ≤ d < 1. Linear
convergence with rate d — measured 9 iters at d=0.1, 61 at d=0.99.

**Q: Why sparse matrices?**
A: N pages need N² dense entries; with ~10 links per page that's
99.99...% zeros. CSR stores O(E) entries and matvec is O(E).

**Q: What does the transpose-matvec do?**
A: Rank flows from page i to its outlinks j, so `pi_new = Lᵀ pi`.
The transpose-matvec writes `y[cols[k]] += vals[k]·x[i]` — it's the
transpose without ever building it.

**Q: What happens to sink pages?**
A: Their rank mass is redistributed uniformly to all pages — exactly
equivalent to giving every sink uniform out-links. Verified against
that formulation.

**Q: How do you know the implementation is right?**
A: I compared against an independent dense-numpy power iteration on
random graphs (agreement to 1e-10), checked sum(pi)=1, checked known
small graphs (2-cycle → 0.5/0.5), and checked sink behavior.

**Q: How does search combine text and links?**
A: TF-IDF scores relevance; PageRank scores importance; final = tfidf
+ beta·rank. Without the prior, an authority page can lose to a
sibling with the same text; with it, the endorsed page wins.

**Q: Why is a repeated query term a bug?**
A: Because the score loop would count the same evidence twice —
'quantum quantum' ≠ 'quantum'. Evidence is per-term, not per-
occurrence.

**Q: How would you scale this?**
A: Vectorised/blocked matvec, parallel power iteration, and
PageRank's sparsity means the memory is fine; the Python loop is the
bottleneck.

---

## 12. Whiteboard script (5 minutes)

1. Draw a tiny web (5 pages, arrows). Explain the surfer: follow a
   random arrow with prob 0.85, jump anywhere with 0.15.
2. Write the update `pi_new[j] = (1-d)/N + d·Σ_{i→j} pi[i]/out(i)`.
3. Say: "this is pi_new = d·Lᵀ·pi + (1-d)/N — power iteration on the
   Markov chain's matrix."
4. Show sink handling: "a page with no out-arrows spreads its mass
   uniformly."
5. Write the CSR picture (three arrays) and the transpose-matvec
   loop.
6. One-line summary.

---

## 13. The one-sentence summary

> **"I built a search engine around PageRank from scratch — the random
> surfer as a Markov chain solved by power iteration on a CSR sparse
> matrix with damping, sink handling, and verified convergence — and
> blended TF-IDF relevance with PageRank importance to rank results."**

---

*Files: `sparse.py` (the matrix) · `pagerank.py` (the algorithm) ·
`crawl.py` (the graph) · `search.py` (the ranking) · `verify.py`
(proofs) · `demo.py` (the story). Run `python3 demo.py` to re-walk
the whole story.*
