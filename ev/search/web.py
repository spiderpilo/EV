from duckduckgo_search import DDGS

# Must appear at the start — strong explicit intent
SEARCH_PREFIXES = (
    "search for", "search", "look up", "look for",
    "tell me about", "find out about",
    "explain", "define", "definition of",
    "find information on", "find information about",
)

# Question words — checked anywhere in the first 4 words so Whisper dropouts don't break it
QUESTION_WORDS = (
    "what", "what's", "whats",
    "who", "who's", "whos",
    "how", "how's",
    "when", "where", "why",
)


def is_search_query(text: str) -> bool:
    normalized = text.strip().lower().rstrip("?.!")
    words = normalized.split()

    if any(normalized.startswith(p) for p in SEARCH_PREFIXES):
        return True

    # Question word anywhere in first 4 words covers Whisper dropouts like
    # "um who is Elon Musk" or "so what is quantum computing"
    return any(w in QUESTION_WORDS for w in words[:4])


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
