"""Author records and real collaborators from the Semantic Scholar Graph API.

    python tools/fetch_s2.py

Writes data/external/:
    s2_papers.jsonl.gz         per paper: its authors as Semantic Scholar resolves them
                               (authorId, name, hIndex, citationCount, paperCount)
    s2_collaborators.jsonl.gz  per paper: co-authors of its first author on up to 20 of the
                               first author's other papers closest in year, most frequent first

The sample's paper ids are Semantic Scholar paper ids, so authors resolve exactly; the
release's collaborators.py searched authors by name instead. No API key is needed for the
batch endpoints. Responses are cached under data/external/s2_cache/, so a rerun resumes.
"""

import gzip
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gts.data import load_papers  # noqa: E402
from gts.paths import ROOT  # noqa: E402

API = "https://api.semanticscholar.org/graph/v1"
OUT = ROOT / "data" / "external"
CACHE = OUT / "s2_cache"
PAPERS_PER_AUTHOR = 20
AUTHOR_FIELDS = "authors.authorId,authors.name,authors.hIndex,authors.citationCount,authors.paperCount"


def post(endpoint: str, fields: str, ids: list[str]):
    """One batch request, cached on disk; retries on rate limits and server errors."""
    key = hashlib.sha256(f"{endpoint}|{fields}|{','.join(ids)}".encode()).hexdigest()[:32]
    path = CACHE / f"{key}.json.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    body = json.dumps({"ids": ids}).encode()
    for attempt in range(10):
        req = urllib.request.Request(f"{API}/{endpoint}?fields={fields}", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(min(2 ** attempt, 60))
    else:
        raise RuntimeError(f"{endpoint}: gave up after 10 attempts")
    CACHE.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    time.sleep(1.0)  # unauthenticated requests share one rate limit
    return data


def chunks(rows: list, size: int):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    t0 = time.time()
    papers = load_papers().sort_values("paper_id")
    ids = papers["paper_id"].tolist()
    year = dict(zip(papers["paper_id"], papers["year"].astype(int)))

    # 1. the sample's own author lists, resolved
    resolved = {}
    for batch in chunks(ids, 500):
        for pid, rec in zip(batch, post("paper/batch", AUTHOR_FIELDS, batch)):
            resolved[pid] = rec["authors"] if rec else None
    write_jsonl([{"paper_id": p, "authors": resolved[p]} for p in ids], OUT / "s2_papers.jsonl.gz")
    found = sum(a is not None for a in resolved.values())
    print(f"papers resolved: {found} of {len(ids)} ({time.time() - t0:.0f}s)", flush=True)

    # 2. each first author's other papers, the PAPERS_PER_AUTHOR closest in year
    first = {p: a[0]["authorId"] for p, a in resolved.items() if a and a[0].get("authorId")}
    sources: dict[str, list[str]] = {}
    for batch in chunks(sorted(first), 100):
        authors = sorted({first[p] for p in batch})
        recs = {r["authorId"]: r for r in post("author/batch", "papers.paperId,papers.year", authors) if r}
        for pid in batch:
            rec = recs.get(first[pid])
            other = [(abs((q.get("year") or 0) - year[pid]) if q.get("year") else 99, q["paperId"])
                     for q in (rec or {}).get("papers", []) if q["paperId"] != pid]
            sources[pid] = [q for _, q in sorted(other)[:PAPERS_PER_AUTHOR]]
    print(f"first authors with other papers: {sum(bool(v) for v in sources.values())} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # 3. the authors of those papers
    wanted = sorted({q for qs in sources.values() for q in qs})
    team_of: dict[str, list] = {}
    for n, batch in enumerate(chunks(wanted, 500)):
        for qid, rec in zip(batch, post("paper/batch", "authors.authorId,authors.name", batch)):
            team_of[qid] = [(a.get("authorId"), a.get("name")) for a in rec["authors"]] if rec else []
        if n % 50 == 0:
            print(f"  source papers {min((n + 1) * 500, len(wanted))} of {len(wanted)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for pid in ids:
        if pid not in first:
            rows.append({"paper_id": pid, "first_author_id": None, "n_source_papers": 0, "collaborators": []})
            continue
        own = first[pid]
        counts, last_seen = Counter(), {}
        for rank, qid in enumerate(sources.get(pid, [])):
            for aid, name in team_of.get(qid, []):
                if aid != own and name:
                    counts[name] += 1
                    last_seen.setdefault(name, rank)
        ranked = sorted(counts, key=lambda nm: (-counts[nm], last_seen[nm]))
        rows.append({"paper_id": pid, "first_author_id": own, "n_source_papers": len(sources.get(pid, [])),
                     "collaborators": [[nm, counts[nm]] for nm in ranked]})
    write_jsonl(rows, OUT / "s2_collaborators.jsonl.gz")
    print(f"collaborators: {sum(len(r['collaborators']) >= 3 for r in rows)} of {len(rows)} papers "
          f"have at least 3 ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
