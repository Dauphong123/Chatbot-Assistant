from ddgs import DDGS


class SearchEngineLoader:
    def __init__(self):
        pass

    def search(self, query, max_results=10):
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
