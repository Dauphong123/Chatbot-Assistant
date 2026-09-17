class ContextBuilder:
    def __init__(self) -> None:
        pass

    def build(self, results, max_chars=1200):
        context = ""

        for result in results:
            text = result["document"].text

            if len(context) + len(text) > max_chars:
                break

            context += text + "\n\n"

        return context
