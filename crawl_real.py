"""Crawl a small real site and run the full pipeline on it.

The synthetic web is the reproducible core of the project; this script
is the honest complement: a bounded, rate-limited crawl of a real
site, then PageRank + search over the real link graph.

Usage:
    python3 crawl_real.py --seeds https://www.imperial.ac.uk/mathematics/ --max-pages 30

Results are saved to data/real_crawl.json and printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from crawl import PoliteCrawler
from pagerank import pagerank, rank_scores
from search import Index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--max-pages", type=int, default=30)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join("data", "real_crawl.json"))
    args = ap.parse_args()

    crawler = PoliteCrawler(delay=args.delay, max_pages=args.max_pages)
    t0 = time.time()
    pages, links = crawler.crawl(args.seeds)
    print(f"crawled {len(pages)} pages, {len(links)} links "
          f"in {time.time() - t0:.1f}s")

    if len(pages) < 2:
        print("too few pages; nothing to rank")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"pages": pages, "links": links}, f, indent=1)
    print(f"saved {args.out}")

    pi = pagerank(links, len(pages))
    top = rank_scores(pi, [p["url"] for p in pages])
    print("\ntop-10 by PageRank:")
    for url, r in top[:10]:
        print(f"  {r:8.4f}  {url}")

    idx = Index(pages, links)
    for q in ("mathematics", "research", "admissions"):
        print(f"\nquery: '{q}'")
        for r in idx.search(q, top_k=3):
            print(f"  {r['title'][:60]:60s} score={r['score']:.3f}")


if __name__ == "__main__":
    main()
