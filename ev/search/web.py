from duckduckgo_search import DDGS

SEARCH_TRIGGERS = (
    "search for", "search", "look up", "look for",
    "what is", "what are", "what was", "what were",
    "who is", "who are", "who was",
    "how do", "how does", "how did", "how to",
    "when did", "when was", "when is",
    "where is", "where are", "where was",
    "why is", "why does", "why did",
    "tell me about", "find out about", "find",
    "explain", "define", "definition of",
)


def is_search_query(text: str) -> bool:
    normalized = text.strip().lower().rstrip("?.!")
    return any(normalized.startswith(t) for t in SEARCH_TRIGGERS)


def search(query: str, max_results: int = 3) -> str:
    """Run a DDG text search and return a snippet string for the LLM."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        parts = []
        for r in results:
            parts.append(f"{r['title']}: {r['body']}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Search failed: {e}"
