"""
Web tools: search, read webpage, fetch RSS.
"""

import re
import sqlite3
import hashlib
from json import dumps as json_to_string
from os import makedirs

from langchain.tools import tool
from curl_cffi.requests import get as http_get
from html2text import HTML2Text
from xml.etree.ElementTree import fromstring as xml_parse
from bs4 import BeautifulSoup
from os import environ as env

# ── Search result cache ────────────────────────────────────────────────────────

_SEARCH_CACHE_DB = ".cache/search.db"


def _search_cache_conn() -> sqlite3.Connection:
    makedirs(".cache", exist_ok=True)
    conn = sqlite3.connect(_SEARCH_CACHE_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS search_cache "
        "(key TEXT PRIMARY KEY, result TEXT)"
    )
    conn.commit()
    return conn


def _search_cache_key(search_term: str, max_results: int, page: int) -> str:
    raw = f"{search_term}|{max_results}|{page}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> str | None:
    with _search_cache_conn() as conn:
        row = conn.execute("SELECT result FROM search_cache WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _cache_put(key: str, result: str) -> None:
    with _search_cache_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO search_cache (key, result) VALUES (?, ?)", (key, result))
        conn.commit()

# Config
READ_WEBPAGE_MAX_CHARS = 100_000
SUMMY_SUMMARIZE_WHEN_OVER_CHARS = 25_000
SUMMY_MAX_SENTENCES = 80

_STRIP_HTML_TAGS = (
    "script", "style", "noscript", "iframe", "svg", "object", "embed",
    "nav", "header", "footer", "aside", "form", "button", "input", "select", "textarea",
)
_STRIP_HTML_ROLES = ("navigation", "banner", "contentinfo", "complementary", "search")


def _strip_html_noise(html: str) -> str:
    """Remove script, style, nav, and other typical noise before converting to text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_STRIP_HTML_TAGS):
        tag.decompose()
    for el in soup.find_all(attrs={"role": lambda v: v and v in _STRIP_HTML_ROLES}):
        el.decompose()
    main = soup.find("main") or soup.find("article")
    if main and len(main.get_text(strip=True)) > 200:
        body = main
    else:
        body = soup.find("body") or soup
    return str(body)


def _clean_extracted_text(text: str) -> str:
    """Collapse excessive whitespace and drop common UI-only lines."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = []
    skip_phrases = re.compile(
        r"^(share|tweet|subscribe|newsletter|cookie|accept all|manage preferences|"
        r"follow us|related articles|read more|advertisement|sponsored|\\|/|·)$",
        re.I,
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or skip_phrases.match(line) or len(line) <= 2:
            continue
        lines.append(line)
    return "\n\n".join(lines)


def _extractive_summary(text: str, max_sentences: int = SUMMY_MAX_SENTENCES) -> str:
    """Use sumy TextRank to reduce long text to top-ranked sentences (no LLM)."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        n = min(max_sentences, len(parser.document.sentences))
        if n <= 0:
            return text
        summary_sentences = summarizer(parser.document, n)
        return " ".join(str(s) for s in summary_sentences)
    except Exception:
        return text


@tool(description="Search the web for information. max_results controls how many results to return (default 5, max 20). page selects the result page (default 1).")
def website_search(search_term: str, max_results: int = 5, page: int = 1) -> str:
    max_results = max(1, min(max_results, 20))
    cache_key = _search_cache_key(search_term, max_results, page)
    if not env.get("FRESH_RUN"):
        cached = _cache_get(cache_key)
        if cached:
            return cached
    offset = (page - 1) * max_results
    response = http_get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={
            "X-Subscription-Token": env["BRAVE_SEARCH_API_KEY"],
            "Accept": "application/json",
        },
        params={"q": search_term, "count": max_results, "offset": offset},
    )
    if response.status_code == 200:
        results = response.json().get("web", {}).get("results", [])
        result = json_to_string([{"title": r["title"], "url": r["url"]} for r in results[:max_results]])
        _cache_put(cache_key, result)
        return result
    return f"Search failed: {response.status_code}"


@tool(description="Read and retrieve the content of a webpage by URL")
def read_webpage(url: str) -> str:
    try:
        response = http_get(url, timeout=10, impersonate="chrome124")
        response.raise_for_status()
        html_clean = _strip_html_noise(response.text)
        h = HTML2Text()
        h.ignore_links = False
        h.body_width = 0
        text = h.handle(html_clean)
        text = _clean_extracted_text(text)
        if len(text) > SUMMY_SUMMARIZE_WHEN_OVER_CHARS:
            text = _extractive_summary(text, SUMMY_MAX_SENTENCES)
        if len(text) > READ_WEBPAGE_MAX_CHARS:
            text = text[:READ_WEBPAGE_MAX_CHARS] + "\n\n[... content truncated to fit context ...]"
        return text
    except Exception as e:
        return f"Failed to fetch page: {e}"


@tool(description="Fetch an RSS feed and return article titles, descriptions, and URLs. Prefer this over read_webpage when the URL ends in .xml or is a known RSS/Atom feed.")
def fetch_rss_articles(feed_url: str, max_article_count: int = 5) -> str:
    try:
        response = http_get(feed_url, timeout=10, impersonate="chrome124")
        response.raise_for_status()
        root = xml_parse(response.content)
        items = root.findall(".//item")
        articles = []
        for item in items[:max_article_count]:
            title = (item.findtext("title") or "").strip()
            desc = (item.findtext("description") or "").strip()
            link = (item.findtext("link") or "").strip()
            articles.append({"title": title, "description": desc[:300], "url": link})
        return json_to_string(articles)
    except Exception as e:
        return f"Failed to fetch RSS feed: {e}"


website_ops_tooling = [
    website_search,
    read_webpage,
    fetch_rss_articles,
]

__all__ = ["website_search", "read_webpage", "fetch_rss_articles", "website_ops_tooling"]
