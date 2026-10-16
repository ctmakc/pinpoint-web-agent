"""Search and fetch. Standard library only, so the container needs no extra HTTP stack."""
from __future__ import annotations

import gzip
import html
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/128.0 Safari/537.36 PinpointAgent/0.1")
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/128.0.0.0 Safari/537.36")
MAX_BYTES = 3_000_000

SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "iframe", "head", "form", "button", "select"}
BLOCK_TAGS = {"p", "div", "section", "article", "main", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3",
              "h4", "h5", "h6", "br", "hr", "dd", "dt", "dl", "pre", "blockquote", "figcaption", "caption",
              "header", "footer", "nav", "aside", "tbody", "thead"}


CITATION_LINE = re.compile(r"(Retrieved (on )?\d{1,2} \w+ \d{4}|Retrieved \w+ \d{1,2}, \d{4}|^\s*[\u2191^]\s|Archived from the original)")


@dataclass
class Page:
    url: str
    title: str = ""
    text: str = ""
    links: list[tuple[str, str]] = field(default_factory=list)  # (anchor text, absolute url)
    error: str = ""


class _Extractor(HTMLParser):
    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.title = ""
        self._skip = 0
        self._in_title = False
        self._href: str | None = None
        self._anchor: list[str] = []
        self._cell = False

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip += 1
            if tag == "head":
                self._skip -= 1  # keep <title> reachable; head has little body text
            return
        if tag == "title":
            self._in_title = True
        if tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag in ("td", "th"):
            self.parts.append(" | ")
        if tag == "a":
            href = dict(attrs).get("href")
            if href and not href.startswith(("javascript:", "mailto:", "#")):
                self._href = urllib.parse.urljoin(self.base, href)
                self._anchor = []

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and tag != "head":
            self._skip = max(0, self._skip - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._href:
            text = " ".join("".join(self._anchor).split())
            if text:
                self.links.append((text[:120], self._href.split("#")[0]))
            self._href = None
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._skip:
            return
        self.parts.append(data)
        if self._href is not None:
            self._anchor.append(data)


def html_to_page(url: str, raw: str) -> Page:
    ex = _Extractor(url)
    try:
        ex.feed(raw)
        ex.close()
    except Exception:  # malformed markup: keep whatever was parsed
        pass
    text = "".join(ex.parts)
    text = re.sub(r"\[(\d+|[a-z]|edit|citation needed)\]", "", text)
    lines = [" ".join(l.split()) for l in text.split("\n")]
    text = "\n".join(l for l in lines if l and not CITATION_LINE.search(l))
    for marker in ("\nReferences\n", "\nNotes and references\n", "\nExternal links\n"):
        cut = text.find(marker)
        if cut > len(text) * 0.4:  # bibliography tail, mostly on wiki-style pages
            text = text[:cut]
    seen, links = set(), []
    for a, h in ex.links:
        if h.startswith("http") and h not in seen:
            seen.add(h)
            links.append((a, h))
    return Page(url=url, title=" ".join(ex.title.split()), text=text, links=links)


def http_get(url: str, timeout: float = 6.0, data: bytes | None = None) -> tuple[str, str]:
    """Return (final_url, decoded body). Raises on network errors."""
    req = urllib.request.Request(url, data=data, headers={
        "User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8", "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read(MAX_BYTES)
        enc = (r.headers.get("Content-Encoding") or "").lower()
        if enc == "gzip":
            body = gzip.decompress(body)
        elif enc == "deflate":
            body = zlib.decompress(body)
        ctype = r.headers.get("Content-Type", "")
        m = re.search(r"charset=([\w-]+)", ctype)
        charset = m.group(1) if m else "utf-8"
        if "pdf" in ctype:
            raise ValueError("pdf not supported")
        return r.geturl(), body.decode(charset, errors="replace")


def fetch(url: str, timeout: float = 6.0) -> Page:
    try:
        final, body = http_get(url, timeout)
    except (urllib.error.URLError, socket.timeout, TimeoutError, ValueError, OSError, Exception) as e:
        return Page(url=url, error=f"{type(e).__name__}: {e}"[:200])
    stripped = body.lstrip()
    if stripped[:1] in "{[":
        flat = _flatten_json(stripped)
        if flat:
            return Page(url=final, title=final, text=flat)
    page = html_to_page(final, body)
    page.url = final
    return page


def _flatten_json(body: str) -> str:
    """API responses: one 'path: value' line per leaf, so a field and its value stay in one passage."""
    import json
    try:
        data = json.loads(body)
    except ValueError:
        return ""
    lines: list[str] = []

    def walk(x, path):
        if len(lines) > 4000:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x[:200]):
                walk(v, f"{path}[{i}]")
        elif x is not None and x != "":
            if isinstance(x, str) and x.startswith("http") and "api.github.com" in x:
                return  # hypermedia links add noise, not facts
            lines.append(f"{path.replace('_', ' ')}: {x}")

    walk(data, "")
    return "\n".join(lines)


def fetch_many(urls: list[str], timeout: float = 6.0, workers: int = 6) -> list[Page]:
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(urls))) as ex:
        return list(ex.map(lambda u: fetch(u, timeout), urls))


# ---------------------------------------------------------------- search
# Search results are (title, url, snippet). Providers are keyless; the first that answers wins.

def _post_form(url: str, fields: dict, timeout: float) -> str:
    req = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode(), headers={
        "User-Agent": BROWSER_UA, "Content-Type": "application/x-www-form-urlencoded",
        "Referer": url, "Accept": "text/html", "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(MAX_BYTES).decode("utf-8", "replace")


def _clean(fragment: str) -> str:
    return " ".join(html.unescape(re.sub("<[^>]+>", "", fragment)).split())


def _ddg_lite(query: str, timeout: float) -> list[tuple[str, str, str]]:
    body = _post_form("https://lite.duckduckgo.com/lite/", {"q": query, "kl": "us-en"}, timeout)
    if "anomaly" in body.lower():
        raise RuntimeError("challenge page")
    links = re.findall(r'<a rel="nofollow" href="([^"]+)" class=\'result-link\'>(.*?)</a>', body, re.S)
    snippets = re.findall(r"<td class='result-snippet'>(.*?)</td>", body, re.S)
    out = []
    for i, (href, title) in enumerate(links):
        href = html.unescape(href)
        if "duckduckgo.com/y.js" in href or not href.startswith("http"):
            continue
        out.append((_clean(title), href, _clean(snippets[i]) if i < len(snippets) else ""))
    return out


def _ddg_html(query: str, timeout: float) -> list[tuple[str, str, str]]:
    body = _post_form("https://html.duckduckgo.com/html/", {"q": query, "kl": "us-en"}, timeout)
    if "anomaly" in body.lower():
        raise RuntimeError("challenge page")
    out = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', body, re.S):
        href = html.unescape(m.group(1))
        if "uddg=" in href:
            href = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(href).query)["uddg"][0])
        if "duckduckgo.com/y.js" in href or not href.startswith("http"):
            continue
        out.append((_clean(m.group(2)), href, ""))
    return out


def _keywords(query: str) -> str:
    from .extract import terms
    return " ".join(terms(query)) or query


def _wikipedia(query: str, timeout: float) -> list[tuple[str, str, str]]:
    import json
    _, body = http_get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": _keywords(query), "format": "json", "srlimit": 5}), timeout)
    hits = json.loads(body).get("query", {}).get("search", [])
    return [(h["title"], "https://en.wikipedia.org/wiki/" + urllib.parse.quote(h["title"].replace(" ", "_")),
             _clean(h.get("snippet", ""))) for h in hits]


def _unwrap_bing(href: str) -> str:
    """Bing wraps results as /ck/a?...&u=a1<base64url(target)>."""
    if "bing.com/ck/" not in href:
        return href
    import base64
    u = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("u", [""])[0]
    if u.startswith("a1"):
        try:
            return base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode("utf-8", "replace")
        except Exception:
            return href
    return href


def _bing(query: str, timeout: float) -> list[tuple[str, str, str]]:
    req = urllib.request.Request("https://www.bing.com/search?" + urllib.parse.urlencode(
        {"q": _keywords(query), "setlang": "en-US", "cc": "US"}), headers={"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read(MAX_BYTES).decode("utf-8", "replace")
    out = []
    for m in re.finditer(r'<li class="b_algo".*?<h2[^>]*>\s*<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>(.*?)</li>', body, re.S):
        url = _unwrap_bing(html.unescape(m.group(1)))
        snip = re.search(r"<p[^>]*>(.*?)</p>", m.group(3), re.S)
        if url.startswith("http"):
            out.append((_clean(m.group(2)), url, _clean(snip.group(1)) if snip else ""))
    # Bing serves unrelated filler to some clients; keep only results that share a word with the query
    from .extract import terms
    q = set(terms(query))
    need = max(1, (len(q) + 1) // 2)
    return [r for r in out if len(q & set(terms(r[0] + " " + r[2] + " " + r[1].replace("/", " ")))) >= need]


SEARCHERS = [("ddg-lite", _ddg_lite), ("ddg-html", _ddg_html), ("bing", _bing), ("wikipedia", _wikipedia)]


def search(query: str, timeout: float = 5.0, want: int = 8) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Try providers in order until one returns enough results. Returns (results, provider log)."""
    log, results, seen = [], [], set()
    for name, fn in SEARCHERS:
        try:
            hits = fn(query, timeout)
        except Exception as e:  # blocked, timeout, layout change
            log.append(f"{name}: {type(e).__name__}")
            continue
        log.append(f"{name}: {len(hits)}")
        for t, u, sn in hits:
            key = u.split("#")[0].rstrip("/")
            if key not in seen:
                seen.add(key)
                results.append((t, u, sn))
        if len(results) >= want // 2:
            break
    return results[:want], log
