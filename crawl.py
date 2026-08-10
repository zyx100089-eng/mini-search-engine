"""Crawler + synthetic web dataset.

Two sources of web graphs:

1. `synthetic_web()` - a deterministic, web-like graph generated from
   a seeded random model: content pages grouped into topical
   communities, hub pages that link to many pages, authority pages
   that many pages link to, and some sink pages.  Reproducible and
   fast, which is what the verification suite and demo need.

2. `PoliteCrawler` - a real, rate-limited crawler that fetches a list
   of URLs (html.parser based), extracts links and visible text, and
   keeps them in RAM.  Use it if you want to crawl a small real
   network; the rest of the project only needs (pages, links).

A page record is a dict: {"id", "url", "title", "text", "links"}.
"""

from __future__ import annotations

import random
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser


# ----------------------------------------------------------------------
# Synthetic web
# ----------------------------------------------------------------------

_TOPICS = [
    ("quantum mechanics", "wavefunction superposition entanglement measurement observable"),
    ("graph theory", "vertices edges paths connectivity cycles trees bipartite"),
    ("deep learning", "neural network gradient backpropagation layers weights training"),
    ("ancient rome", "empire senate caesar legion provinces aqueducts gladiators"),
    ("python programming", "interpreter loops functions modules imports exceptions syntax"),
    ("olympic sports", "athletes medal events sprint marathon podium records"),
    ("linear algebra", "vectors matrices eigenvalues eigenvectors determinant basis"),
    ("renaissance art", "painting fresco sculpture patron perspective pigment canvas"),
    ("climate science", "temperature emission warming carbon atmosphere ocean ice"),
    ("solar system", "planets orbit gravity asteroid comet sun telescope nebula"),
]


def synthetic_web(n_pages: int = 60, seed: int = 0) -> tuple[list[dict], list[tuple[int, int]]]:
    """A deterministic web-like graph.

    Returns (pages, links): pages is a list of dicts with 'title',
    'text', 'url'; links is a list of (i, j) directed edges.

    Model (roles are explicit and non-overlapping, so the demo and
    verification tell a clean story):
      - content pages grouped into 10 topical communities; within a
        community ~12% of pairs link (clustering)
      - 4 hub pages link out to ~40% of content pages (low content,
        high out-degree)
      - 3 authority pages receive links from ~45% of content pages
        (high in-degree, little out-linking)
      - 3 sink pages link nowhere (out-degree 0)
    """
    rng = random.Random(seed)
    pages: list[dict] = []
    links: list[tuple[int, int]] = []

    # community content pages
    topic_assign = []
    for t, (topic, words) in enumerate(_TOPICS):
        for _ in range(max(3, n_pages // 14)):
            i = len(pages)
            pages.append({
                "id": i,
                "title": f"{topic.title()} #{i}",
                "text": f"About {topic}. {words}",
                "url": f"https://example.org/{topic.replace(' ', '-')}/{i}",
            })
            topic_assign.append(t)

    n = len(pages)

    def link(i, j):
        if i != j:
            links.append((i, j))

    # within-topic links (communities)
    for i in range(n):
        for j in range(i + 1, n):
            if topic_assign[i] == topic_assign[j] and rng.random() < 0.12:
                link(i, j)
                link(j, i)

    # hubs: link out to many content pages
    hub_ids = rng.sample(range(n), min(4, n))
    hub_set = set(hub_ids)
    for h in hub_ids:
        targets = rng.sample([i for i in range(n) if i != h], n // 3)
        for t in targets:
            link(h, t)

    # authorities: many pages link to them; they link out little
    auth_ids = rng.sample([i for i in range(n) if i not in hub_set], min(3, n))
    for a in auth_ids:
        for i in range(n):
            if i != a and i not in hub_set and rng.random() < 0.45:
                link(i, a)
        # authorities link to a couple of other authorities (web norm)
        for a2 in auth_ids:
            if a2 != a:
                link(a, a2)

    # sinks: link nowhere
    sink_ids = rng.sample([i for i in range(n)
                           if i not in hub_set and i not in set(auth_ids)],
                          min(3, n))
    sink_set = set(sink_ids)
    links = [(i, j) for (i, j) in links if i not in sink_set]

    return pages, links


# ----------------------------------------------------------------------
# Polite real crawler
# ----------------------------------------------------------------------

class _LinkTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []
        self.text: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        if tag == "a":
            for k, v in attrs:
                if k == "href":
                    self.links.append(v)

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0 and data.strip():
            self.text.append(data.strip())


class PoliteCrawler:
    """Minimal polite crawler: one request per `delay` seconds."""

    def __init__(self, delay: float = 1.0, timeout: float = 10.0,
                 max_pages: int = 50):
        self.delay = delay
        self.timeout = timeout
        self.max_pages = max_pages
        self.user_agent = "MiniSearchEngine/0.1 (educational)"

    def fetch(self, url: str) -> tuple[str, list[str]] | None:
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None
        time.sleep(self.delay)  # rate limit
        parser = _LinkTextParser()
        parser.feed(html)
        text = " ".join(parser.text)
        return text, parser.links

    def crawl(self, seeds: list[str]) -> tuple[list[dict], list[tuple[int, int]]]:
        """Breadth-first crawl from the seeds (bounded by max_pages).

        Link extraction is deferred: a link to a page that is only
        queued (not yet fetched) is recorded as (from_id, target_url)
        and resolved to an edge once the target gets an id.  This keeps
        every link, not just links to already-visited pages.
        """
        pages: list[dict] = []
        index: dict[str, int] = {}
        links: list[tuple[int, int]] = []
        pending: dict[str, list[int]] = {}  # url -> ids of pages linking it
        queued: set[str] = set(seeds)       # urls already in queue
        queue = list(seeds)
        seen: set[str] = set()
        while queue and len(pages) < self.max_pages:
            url = queue.pop(0)
            url = urllib.parse.urlsplit(url)._replace(fragment="").geturl()
            if url in seen:
                continue
            seen.add(url)
            result = self.fetch(url)
            if result is None:
                pending.pop(url, None)
                continue
            text, raw_links = result
            i = len(pages)
            index[url] = i
            pages.append({
                "id": i,
                "url": url,
                "title": url.split("/")[-1] or url,
                "text": text,
                "links": [],
            })
            # resolve edges from pages that linked here earlier
            for from_id in pending.pop(url, []):
                links.append((from_id, i))
            for l in raw_links:
                absolute = urllib.parse.urljoin(url, l)
                absolute = urllib.parse.urlsplit(absolute)._replace(fragment="").geturl()
                # only http(s) pages are fetchable; skip mailto:, javascript:, etc.
                scheme = urllib.parse.urlsplit(absolute).scheme
                if scheme not in ("http", "https"):
                    continue
                if absolute == url:
                    continue
                if absolute in index:
                    links.append((i, index[absolute]))
                elif absolute not in seen:
                    pending.setdefault(absolute, []).append(i)
                    if absolute not in queued:
                        queued.add(absolute)
                        queue.append(absolute)
        return pages, links
