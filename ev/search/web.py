from duckduckgo_search import DDGS

# Explicit search phrases — must appear at the start
SEARCH_PREFIXES = (
    "search for",
    "search up",
    "look up",
    "look for",
    "tell me about",
    "find out about",
    "find information on",
    "find information about",
    "can you find",
    "can you look up",
    "can you search",
    "i need to know about",
    "i want to know about",
    "i want to know more about",
    "find me",
    "get me info on",
    "get me information on",
    "what is",
    "what are",
    "what was",
    "what were",
    "what's",
    "who is",
    "who are",
    "who was",
    "who's",
    "how does",
    "how do",
    "how did",
    "how to",
    "when did",
    "when was",
    "when is",
    "where is",
    "where was",
    "where are",
    "why does",
    "why did",
    "why is",
    "explain",
    "define",
    "definition of",
    "tell me more about",
    "do you know about",
    "do you know anything about",
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
