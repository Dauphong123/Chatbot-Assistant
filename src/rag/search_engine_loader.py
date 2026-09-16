from ddgs import DDGS
import re


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

    # Remove excessive spaces/tabs
    content = re.sub(r"[ \t]+", " ", content)

    # Remove excessive blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Remove whitespace around lines
    lines = [line.strip() for line in content.split("\n") if line.strip()]

    return "\n".join(lines)


class SearchEngineLoader:
    def __init__(self):
        pass

    def search(self, query, max_results=10):
        ddgs = DDGS()

        results = ddgs.text(query, max_results=max_results)

        documents = []

        for result in results:
            try:
                page = ddgs.extract(result["href"], fmt="text_markdown")
                text = clean_content(page["content"])
                text = deduplicate_lines(text)

                documents.append(
                    {
                        "text": text,
                        "href": result["href"],
                        "title": result["title"],
                    }
                )
            except Exception as e:
                print(f"Failed to extract {result['href']}: {e}")
                continue
        return documents
