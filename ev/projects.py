import difflib
from pathlib import Path

REPOS_DIR = Path.home() / "Documents" / "Repos"

PROJECT_TRIGGERS = (
    "open my project", "open a project", "open project",
    "continue my project", "continue a project", "continue project",
    "work on my project", "work on a project", "work on project",
    "let's work on", "lets work on",
    "open up a project", "open up my project",
    "switch to a project", "switch project",
)


def list_projects() -> list[str]:
    if not REPOS_DIR.exists():
        return []
    return sorted(
        p.name for p in REPOS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def match_project(spoken: str, projects: list[str]) -> str | None:
    spoken_lower = spoken.strip().lower()

    # Exact match first
    for p in projects:
        if p.lower() == spoken_lower:
            return p

    # Substring match (e.g. "EV" matches "EV", "drone" matches "Drone_AI_UAV")
    for p in projects:
        if spoken_lower in p.lower() or p.lower() in spoken_lower:
            return p

    # Fuzzy match as fallback
    normalized = [p.lower().replace("_", " ").replace("-", " ") for p in projects]
    close = difflib.get_close_matches(spoken_lower, normalized, n=1, cutoff=0.4)
    if close:
        idx = normalized.index(close[0])
        return projects[idx]

    return None


def open_project(name: str) -> str:
    path = REPOS_DIR / name
    return f"code {path}"
