"""The retrieval loop: search -> fetch -> rank passages -> read with the model -> verify -> maybe follow a link."""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from . import extract, web
from .llm import parse_json

SYSTEM = """You are a precise web research agent. You read numbered passages taken from real web pages and
extract the one specific piece of information the user asked for.

Rules:
- Answer with the value only: a number with its unit, a name, a date, a version, a price, an address.
  No sentence, no restating the question.
- Give the complete value, including qualifiers that identify it (currency, unit, year, model suffix).
- Use only the passages. If they do not contain the answer, return an empty answer. Never guess from memory.
- "quote" must be copied exactly from the passage that contains the answer.
- "passages" lists only the passage numbers you actually needed.
- If the answer is not in the passages but one of the listed links probably leads to it, put that link
  number in "follow".

Reply with one JSON object and nothing else:
{"answer": "...", "passages": [1], "quote": "...", "follow": null}"""

URL_RE = re.compile(r"https?://[^\s\"'<>)]+")


@dataclass
class Result:
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    evidence: str = ""
    confidence: float = 0.0
    verified: bool = False
    steps: list[str] = field(default_factory=list)
    pages_read: int = 0
    seconds: float = 0.0

    def to_json(self) -> dict:
        d = asdict(self)
        d["citations"] = list(self.sources)  # same list under the name used by earlier League matches
        return d


def _prompt(question: str, passages: list, links: list) -> str:
    parts = [f"Question: {question}", "", "Passages:"]
    for p in passages:
        parts.append(f"[P{p.pid}] {p.title} | {p.url}\n{p.text}\n")
    if links:
        parts.append("Links you could open next:")
        for i, (a, h) in enumerate(links, 1):
            parts.append(f"[L{i}] {a} -> {h}")
    parts.append("\nJSON:")
    return "\n".join(parts)


def _as_int_list(v) -> list[int]:
    out = []
    for x in v if isinstance(v, list) else [v]:
        m = re.search(r"\d+", str(x))
        if m:
            out.append(int(m.group(0)))
    return out


class Agent:
    def __init__(self, llm, budget_s: float = 24.0, fetch_timeout: float = 6.0):
        self.llm = llm
        self.budget = budget_s
        self.fetch_timeout = fetch_timeout

    def run(self, question: str, start_url: str | None = None) -> Result:
        t0 = time.time()
        res = Result()
        left = lambda: self.budget - (time.time() - t0)
        seeds = [u.rstrip(".,;") for u in URL_RE.findall(question)]
        if start_url:
            seeds.insert(0, start_url)
        q_clean = URL_RE.sub("", question).strip() or question

        pages: list = []
        visited: set[str] = set()

        def read(urls: list[str], why: str, timeout: float | None = None):
            urls = [u for u in urls if u not in visited and not u.lower().split("?")[0].endswith(".pdf")]
            if not urls:
                return
            visited.update(urls)
            got = web.fetch_many(urls, timeout=min(timeout or self.fetch_timeout, max(2.0, left() - 8)))
            ok = [p for p in got if p.text]
            pages.extend(ok)
            errs = "; ".join(f"{p.url[:60]} {p.error[:50]}" for p in got if p.error)
            res.steps.append(f"{why}: fetched {len(ok)}/{len(urls)}" + (f" ({errs})" if errs else ""))

        if seeds:
            read(seeds[:2], "start page", timeout=10.0)
            # one hop from the start page toward links that match the request
            hop = [h for _, h in extract.rank_links(q_clean, [l for p in pages for l in p.links], k=3,
                                                    exclude=visited)]
            read(hop, "links from start page")
        if not seeds or not pages:
            hits, log = web.search(q_clean, timeout=5.0)
            res.steps.append("search " + ", ".join(log))
            read([u for _, u, _ in hits[:4]], "search results")
            snippets = [web.Page(url=u, title=t, text=s) for t, u, s in hits if s]
        else:
            hits, snippets = [], []

        for round_no in range(3):
            if left() < 4:
                res.steps.append("budget exhausted")
                break
            passages = extract.rank_passages(q_clean, pages or snippets, k=8, max_chars=9000)
            if not passages:
                res.steps.append("no readable passages")
                break
            links = extract.rank_links(q_clean, [l for p in pages for l in p.links], k=6, exclude=visited)
            raw = self.llm.chat(SYSTEM, _prompt(question, passages, links), max_new_tokens=200)
            j = parse_json(raw)
            ans = str(j.get("answer") or "").strip()
            quote = str(j.get("quote") or "").strip()
            cited = [p for p in passages if p.pid in _as_int_list(j.get("passages") or [])]
            res.steps.append(f"read round {round_no + 1}: {len(passages)} passages, answer={ans[:60]!r}")
            if ans:
                page_text = {p.url: p.text for p in pages}
                good = [p for p in cited if extract.supported(ans, quote, p.text, page_text.get(p.url, ""))]
                if not good:  # model cited the wrong passage number; accept a passage that does hold the value
                    good = [p for p in passages if extract.supported(ans, quote, p.text, page_text.get(p.url, ""))][:1]
                if good:
                    res.answer = ans
                    res.evidence = quote
                    res.sources = list(dict.fromkeys(p.url for p in good))
                    res.verified = True
                    agree = len({p.url for p in passages if extract.supported(ans, "", p.text)})
                    if not quote or extract.norm(quote) not in extract.norm(good[0].text):
                        # keep evidence honest: show the line of the source that contains the value
                        line = next((l for l in good[0].text.split("\n") if extract.norm(ans) in extract.norm(l)), "")
                        quote = line[:300] or quote
                    res.evidence = quote
                    res.confidence = round(min(0.97, 0.6 + 0.12 * agree), 2)
                    break
                res.steps.append("answer not found in the cited page text; rejected")
            follow = _as_int_list(j.get("follow") if j.get("follow") is not None else [])
            nxt = [links[i - 1][1] for i in follow if 0 < i <= len(links)]
            if not nxt:
                nxt = [h for _, h in links[:2]]
            more = [u for _, u, _ in hits[4:7] if u not in visited]
            if not nxt and not more:
                break
            read(nxt[:2] + more[:2], "follow")
        res.pages_read = len(pages)
        res.seconds = round(time.time() - t0, 2)
        return res
