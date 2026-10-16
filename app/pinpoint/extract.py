"""Passage ranking, link scoring and evidence checks. Pure Python, no model."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

STOP = set("""a an the of in on at to for from by with and or is are was were be been being what which who whom
whose when where why how does do did can could should would will shall may might much many is it its this that
these those as about into than then there their they them i me my we our you your he she his her not no yes
please tell find give show me current latest official page site website list number name""".split())

WORD = re.compile(r"[a-z0-9][a-z0-9.\-+/%$€£]*", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("’", "'").replace("–", "-").replace("—", "-").replace(" ", " ")
    return " ".join(s.split())


def squash(s: str) -> str:
    """Grader-style normalisation: upper, no whitespace, no - . · _"""
    return re.sub(r"[\s\-.·_]", "", unicodedata.normalize("NFKC", s)).upper()


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and not w[-2].isdigit():
        return w[:-1]
    return w


def terms(s: str) -> list[str]:
    out = []
    for w in WORD.findall(norm(s)):
        w = w.strip(".-/")
        if w and w not in STOP:
            out.append(_stem(w))
    return out


@dataclass
class Passage:
    pid: int
    url: str
    title: str
    text: str
    score: float = 0.0


def split_passages(url: str, title: str, text: str, size: int = 700, overlap: int = 120) -> list[tuple[str, str, str]]:
    """Split on line boundaries into ~size-char windows, keeping table rows intact."""
    lines = text.split("\n")
    out, buf = [], ""
    for line in lines:
        if len(buf) + len(line) + 1 > size and buf:
            out.append(buf)
            buf = buf[-overlap:] + "\n" + line
        else:
            buf = (buf + "\n" + line) if buf else line
    if buf:
        out.append(buf)
    return [(url, title, p) for p in out]


def rank_passages(query: str, pages: list, k: int = 8, max_chars: int = 9000) -> list[Passage]:
    q = terms(query)
    if not q:
        q = norm(query).split()
    qset = Counter(q)
    chunks = []
    for pg in pages:
        if not pg.text:
            continue
        chunks.extend(split_passages(pg.url, pg.title, pg.text))
    if not chunks:
        return []
    df = Counter()
    toks = []
    for _, _, t in chunks:
        tt = set(terms(t))
        toks.append(tt)
        df.update(tt & set(qset))
    n = len(chunks)
    scored = []
    for i, (u, title, t) in enumerate(chunks):
        tt = toks[i]
        s = 0.0
        for w in qset:
            if w in tt:
                s += math.log(1 + n / (1 + df[w]))
        # bigram bonus: query phrases that appear verbatim
        nt = norm(t)
        for a, b in zip(q, q[1:]):
            if f"{a} {b}" in nt:
                s += 1.0
        # title match bonus
        s += 0.3 * sum(1 for w in qset if w in norm(title))
        # digits are often the value being sought
        if re.search(r"\d", t):
            s += 0.2
        scored.append((s, i))
    scored.sort(reverse=True)
    picked, used, per_url = [], 0, Counter()
    cap = k if len({u for u, _, _ in chunks}) == 1 else 4
    for s, i in scored:
        if s <= 0 or len(picked) >= k:
            break
        u, title, t = chunks[i]
        if per_url[u] >= cap or used + len(t) > max_chars:
            continue
        per_url[u] += 1
        used += len(t)
        picked.append(Passage(pid=len(picked) + 1, url=u, title=title, text=t, score=s))
    return picked


def rank_links(query: str, links: list[tuple[str, str]], k: int = 6, exclude: set | None = None) -> list[tuple[str, str]]:
    exclude = exclude or set()
    q = set(terms(query))
    scored = []
    for anchor, href in links:
        if href in exclude or re.search(r"\.(jpg|jpeg|png|gif|svg|zip|mp4|mp3|exe|dmg)(\?|$)", href, re.I):
            continue
        a = set(terms(anchor)) | set(terms(href.rsplit("/", 1)[-1].replace("-", " ").replace("_", " ")))
        s = len(q & a)
        if s:
            scored.append((s, len(anchor), anchor, href))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out, seen = [], set()
    for _, _, a, h in scored:
        if h not in seen:
            seen.add(h)
            out.append((a, h))
        if len(out) >= k:
            break
    return out


def supported(answer: str, quote: str, passage_text: str, page_text: str = "") -> bool:
    """Accept only if the answer string is really present: in the cited passage, or in the verified quote."""
    if not answer:
        return False
    na, sa = norm(answer), squash(answer)
    if na in norm(passage_text) or sa in squash(passage_text):
        return True
    page = page_text or passage_text
    nq = norm(quote) if quote else ""
    if nq and len(nq) >= 8 and nq in norm(page) and (na in nq or sa in squash(quote)):
        return True
    return False
