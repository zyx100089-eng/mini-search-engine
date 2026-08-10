"""pytest suite for the mini search engine.

Run with:  python3 -m pytest test_search.py -q

Fast unit-level tests.  The slow end-to-end checks (dense-reference
PageRank, batch search behaviour) live in verify.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from crawl import PoliteCrawler, synthetic_web
from pagerank import check_convergence, pagerank, rank_scores
from search import Index, tokenize
from sparse import CSR


# ----------------------------------------------------------------------
# CSR
# ----------------------------------------------------------------------

class TestCSR:
    def test_matvec_matches_dense(self):
        rng = np.random.default_rng(0)
        n = 8
        entries = []
        A = np.zeros((n, n))
        for _ in range(20):
            i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
            v = float(rng.normal())
            entries.append((i, j, v))
            A[i, j] += v
        csr = CSR.from_coo(n, entries)
        x = rng.normal(size=n)
        assert np.allclose(csr.matvec(x), A @ x, atol=1e-10)
        assert np.allclose(csr.matvec_T(x), A.T @ x, atol=1e-10)

    def test_duplicate_entries_merged(self):
        csr = CSR.from_coo(2, [(0, 0, 1.0), (0, 0, 2.0)])
        assert csr.to_dense()[0, 0] == 3.0
        assert csr.nnz() == 1

    def test_empty_matrix(self):
        csr = CSR(5)
        x = np.ones(5)
        assert np.allclose(csr.matvec(x), 0)
        assert np.allclose(csr.matvec_T(x), 0)

    def test_row_sums(self):
        csr = CSR.from_coo(3, [(0, 1, 2.0), (0, 2, 1.0), (2, 0, 3.0)])
        assert np.allclose(csr.row_sums(), [3, 0, 3])

    def test_identity(self):
        csr = CSR.identity(4)
        x = np.arange(4, dtype=float)
        assert np.allclose(csr.matvec(x), x)

    def test_density(self):
        csr = CSR.from_coo(4, [(0, 1, 1.0), (2, 3, 1.0)])
        assert csr.density() == 2 / 16


# ----------------------------------------------------------------------
# PageRank
# ----------------------------------------------------------------------

class TestPageRank:
    def test_two_cycle_symmetric(self):
        pi = pagerank([(0, 1), (1, 0)], 2)
        assert np.allclose(pi, [0.5, 0.5], atol=1e-9)

    def test_chain_ordered(self):
        pi = pagerank([(0, 1), (1, 2)], 3)
        assert pi[2] > pi[1] > pi[0]

    def test_sum_one(self):
        rng = np.random.default_rng(1)
        for _ in range(5):
            n = int(rng.integers(2, 12))
            adj = [(i, j) for i in range(n) for j in range(n)
                   if i != j and rng.random() < 0.2]
            pi = pagerank(adj, n)
            assert check_convergence(pi)

    def test_sink_not_trapping(self):
        pi = pagerank([(0, 1), (1, 2)], 3)
        assert pi[2] < 0.95

    def test_single_page(self):
        assert np.allclose(pagerank([(0, 0)], 1), [1.0])
        assert np.allclose(pagerank([], 1), [1.0])

    def test_damping_bounds(self):
        with pytest.raises(ValueError):
            pagerank([(0, 1)], 2, damping=0.0)
        with pytest.raises(ValueError):
            pagerank([(0, 1)], 2, damping=1.0)

    def test_invalid_edges_rejected(self):
        for bad in ([(0, 5)], [(-1, 0)], [(5, 0)]):
            with pytest.raises(ValueError):
                pagerank(bad, 3)

    def test_rank_scores_ordering(self):
        pi = np.array([0.1, 0.6, 0.3])
        names = rank_scores(pi, ["a", "b", "c"])
        assert [n for n, _ in names] == ["b", "c", "a"]


# ----------------------------------------------------------------------
# tokenize / search
# ----------------------------------------------------------------------

class TestSearch:
    @pytest.fixture(autouse=True)
    def _index(self):
        pages, links = synthetic_web()
        self.pages, self.links = pages, links
        self.idx = Index(pages, links)

    def test_tokenize(self):
        assert tokenize("Hello, World!") == ["hello", "world"]
        assert tokenize("the THE of") == []
        assert tokenize("") == []

    def test_topic_ranked_first(self):
        for query, topic in [("quantum wavefunction", "Quantum Mechanics"),
                             ("caesar legion", "Ancient Rome")]:
            top = self.idx.search(query, top_k=5)
            assert top
            assert all(topic in r["title"] for r in top[:3])

    def test_no_match(self):
        assert self.idx.search("zzzznotaword") == []

    def test_duplicate_terms_no_double_count(self):
        one = self.idx.search("quantum", top_k=3)
        two = self.idx.search("quantum quantum", top_k=3)
        assert [r["id"] for r in one] == [r["id"] for r in two]
        assert all(abs(x["score"] - y["score"]) < 1e-12
                   for x, y in zip(one, two))

    def test_beta_changes_ordering(self):
        plain = Index(self.pages, self.links, beta=0.0)
        blend = Index(self.pages, self.links, beta=1.0)
        assert plain.search("quantum", 5) != blend.search("quantum", 5)

    def test_negative_top_k(self):
        with pytest.raises(ValueError):
            self.idx.search("quantum", top_k=-1)

    def test_snippet_contains_query_term(self):
        top = self.idx.search("quantum", top_k=1)
        assert top and "**quantum**" in top[0]["snippet"]

    def test_snippet_absent_when_disabled(self):
        top = self.idx.search("quantum", top_k=1, snippet=False)
        assert "snippet" not in top[0]

    def test_score_is_tfidf_plus_beta_rank(self):
        top = self.idx.search("quantum", top_k=1)
        r = top[0]
        assert abs(r["score"] - (r["tfidf"] + 1.0 * r["rank"])) < 1e-9


# ----------------------------------------------------------------------
# crawler (mocked network)
# ----------------------------------------------------------------------

FETCH_MAP = {
    "http://a.com": ("text a", ["http://b.com", "http://c.com"]),
    "http://b.com": ("text b", ["http://c.com"]),
    "http://c.com": ("text c", []),
    "http://d.com": None,  # dead page: fetch fails
}


class MockCrawler(PoliteCrawler):
    def fetch(self, url):
        r = FETCH_MAP.get(url)
        return None if r is None else (r[0], r[1])


class TestCrawler:
    def test_bfs_crawl_order_and_edges(self):
        c = MockCrawler(delay=0)
        pages, links = c.crawl(["http://a.com"])
        assert [p["url"] for p in pages] == ["http://a.com", "http://b.com", "http://c.com"]
        # a->b, a->c, b->c: the B->C edge must be kept even though C
        # was only queued when B was fetched (deferred link resolution)
        assert sorted(links) == [(0, 1), (0, 2), (1, 2)]

    def test_dead_page_skipped(self):
        c = MockCrawler(delay=0)
        pages, links = c.crawl(["http://d.com", "http://a.com"])
        assert "http://d.com" not in [p["url"] for p in pages]

    def test_fragment_duplicates_deduped(self):
        c = MockCrawler(delay=0)
        pages, links = c.crawl(["http://a.com", "http://a.com/#frag"])
        assert len(pages) == 3  # a, b, c fetched once each

    def test_max_pages_bound(self):
        c = MockCrawler(delay=0, max_pages=2)
        pages, _ = c.crawl(["http://a.com"])
        assert len(pages) == 2

    def test_scheme_filtering(self):
        FETCH = {
            "http://a.com": ("t", ["mailto:x@y.z", "javascript:void(0)", "http://b.com"]),
            "http://b.com": ("t2", []),
        }
        class C(PoliteCrawler):
            def fetch(self, url):
                r = FETCH.get(url)
                return None if r is None else (r[0], r[1])
        pages, _ = C(delay=0).crawl(["http://a.com"])
        urls = [p["url"] for p in pages]
        assert "mailto:x@y.z" not in urls
        assert "javascript:void(0)" not in urls
        assert "http://b.com" in urls

    def test_synthetic_web_structure(self):
        pages, links = synthetic_web()
        assert len(pages) >= 30
        outdeg = [0] * len(pages)
        for (i, j) in links:
            outdeg[i] += 1
        assert any(d == 0 for d in outdeg), "need sinks"
        assert any(d > 5 for d in outdeg), "need hubs"
