from duckduckgo_search import DDGS

# Explicit search phrases — must appear at the start
SEARCH_PREFIXES = (
    "search for",
    "look up",
    "look for",
    "tell me about",
    "find out about",
    "find information on",
    "find information about",
    "what is",
    "what are",
    "what was",
    "what were",
    "who is",
    "who are",
    "who was",
    "how does",
    "how do",
    "how did",
    "when did",
    "when was",
    "where is",
    "where was",
    "why does",
    "why did",
    "why is",
    "explain",
    "define",
    "definition of",
)


def is_search_query(text: str) -> bool:
    normalized = text.strip().lower().rstrip("?.!")
    return any(normalized.startswith(p) for p in SEARCH_PREFIXES)


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
