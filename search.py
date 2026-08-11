"""Inverted index and query scoring, combined with PageRank.

Retrieval model (a small classic pipeline):

- Tokenise: lowercase, split on non-alphanumerics, drop stopwords and
  single-character tokens.
- Inverted index: term -> posting list of (doc_id, term frequency).
- Query scoring: TF-IDF with length normalisation (cosine-ish: divide
  by doc length, not an L2 norm — see INTERVIEW_GUIDE for the honest
  caveat and future work).
      tf = term frequency in the doc
      idf = log(1 + N / df)   (doc-frequency of the term)
      score(d, q) = sum over query terms of (tf / doc_len) * idf
  plus a PageRank prior: final = tfidf + beta * pagerank.

Why combine both?  Pure text match finds *relevance* (does the page
talk about the topic); PageRank finds *importance* (do other pages
endorse it).  Web search is a blend of both - the demo shows a query
where the two disagree, and the blend wins.
"""

from __future__ import annotations

import math
import re

from pagerank import pagerank

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at",
    "to", "for", "with", "about", "is", "are", "was", "were", "be",
    "this", "that", "it", "its", "as", "by", "from", "not", "we",
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


class Index:
    def __init__(self, pages: list[dict], links: list[tuple[int, int]],
                 damping: float = 0.85, beta: float = 1.0):
        """Build the inverted index and compute PageRank once."""
        self.pages = pages
        self.n = len(pages)
        self.beta = beta
        self.rank = pagerank(links, self.n, damping=damping)

        # term -> {doc_id: term_frequency}
        self.postings: dict[str, dict[int, int]] = {}
        self.doc_lengths: list[int] = []      # sum of tf (for normalising)
        for doc in pages:
            terms = tokenize(doc["text"])
            self.doc_lengths.append(len(terms))
            local: dict[str, int] = {}
            for t in terms:
                local[t] = local.get(t, 0) + 1
            for t, f in local.items():
                self.postings.setdefault(t, {})[doc["id"]] = f

        self.df = {t: len(p) for t, p in self.postings.items()}

    def search(self, query: str, top_k: int = 10,
               snippet: bool = True) -> list[dict]:
        """Rank pages for a query.  Returns list of dicts with 'id',
        'title', 'url', 'score', 'rank', 'tfidf', and (if snippet)
        'snippet' - a window of text around the first query-term
        match, like a real search engine."""
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        terms = tokenize(query)
        if not terms:
            return []
        # dedupe: a repeated query term adds no evidence
        terms = list(dict.fromkeys(terms))

        # tf-idf accumulation (cosine-ish: divide by doc length)
        scores = [0.0] * self.n
        for t in terms:
            postings = self.postings.get(t)
            if not postings:
                continue
            idf = math.log(1.0 + self.n / self.df[t])
            for doc_id, tf in postings.items():
                norm = max(self.doc_lengths[doc_id], 1)
                scores[doc_id] += (tf / norm) * idf

        ranked = []
        for doc_id in range(self.n):
            if scores[doc_id] == 0:
                continue
            s = scores[doc_id] + self.beta * self.rank[doc_id]
            result = {
                "id": doc_id,
                "title": self.pages[doc_id]["title"],
                "url": self.pages[doc_id]["url"],
                "text": self.pages[doc_id]["text"],
                "score": s,
                "tfidf": scores[doc_id],
                "rank": float(self.rank[doc_id]),
            }
            if snippet:
                result["snippet"] = self._snippet(doc_id, terms)
            ranked.append(result)
        ranked.sort(key=lambda r: r["score"], reverse=True)
        return ranked[:top_k]

    def _snippet(self, doc_id: int, terms: list[str],
                 window: int = 10) -> str:
        """A ~window-word window of the document text around the first
        occurrence of any query term, with the term marked by ** **.
        Falls back to the first words if no term is found."""
        words = self.pages[doc_id]["text"].split()
        if not words:
            return ""
        # find the earliest occurrence of any query term
        best = None
        for i, w in enumerate(words):
            if w.lower().strip(".,;:!?()") in terms:
                best = i
                break
        if best is None:
            return " ".join(words[:window]) + ("..." if len(words) > window else "")
        lo = max(0, best - window // 2)
        hi = min(len(words), best + window // 2 + 1)
        part = words[lo:hi]
        for i, w in enumerate(part):
            if w.lower().strip(".,;:!?()") in terms:
                part[i] = f"**{w}**"
        prefix = "..." if lo > 0 else ""
        suffix = "..." if hi < len(words) else ""
        return prefix + " " + " ".join(part) + " " + suffix
