import re

from ddgs import DDGS


def normalize_for_comparison(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def deduplicate_lines(text: str) -> str:
    seen = set()
    result = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line in seen:
            continue

        seen.add(line)
        result.append(line)

    return "\n".join(result)


def clean_content(content):
    content = content.replace("\r\n", "\n")
    content = content.replace("\r", "\n")
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    return "\n".join(lines)


class SearchEngineLoader:
    def __init__(self):
        self.min_content_chars = 120

    def _query_terms(self, query):
        normalized = normalize_for_comparison(query)
        return [term for term in normalized.split() if len(term) > 2]

    def search(self, query, max_results=10):
        if not query or not query.strip():
            return []

        ddgs = DDGS()
        query_terms = set(self._query_terms(query))
        results = ddgs.text(query, max_results=max_results)
        documents = []
        seen_hrefs = set()

        for result in results:
            href = result.get("href")
            if not href or href in seen_hrefs:
                continue
            seen_hrefs.add(href)

            try:
                page = ddgs.extract(href, fmt="text_markdown")
            except Exception:
                continue

            if not isinstance(page, dict):
                continue

            raw_text = page.get("content", "")
            if not raw_text:
                continue

            text = clean_content(raw_text)
            text = deduplicate_lines(text)

            if len(text) < self.min_content_chars:
                continue

            title = result.get("title", "")
            text_norm = normalize_for_comparison(text)
            title_norm = normalize_for_comparison(title)
            match_count = sum(
                1 for term in query_terms if term in text_norm or term in title_norm
            )

            if match_count == 0:
                continue

            documents.append(
                {
                    "text": text,
                    "href": href,
                    "title": title,
                }
            )

            if len(documents) >= max_results:
                break

        return documents
