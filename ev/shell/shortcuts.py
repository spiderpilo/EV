"""
Hard-coded voice shortcuts for terminal mode.

Checked before the LLM so common commands are instant and reliable.
Add new entries to SHORTCUTS or extend match_shortcut() for dynamic patterns.
"""

# (phrase tuple) -> (command, spoken reply)
SHORTCUTS = {
    # Chrome — exact sites
    ("open youtube", "go to youtube", "open youtube.com"): ("google-chrome https://youtube.com", "Opening YouTube."),
    ("open gmail", "go to gmail", "open gmail.com", "check my email", "open my email"): ("google-chrome https://gmail.com", "Opening Gmail."),
    ("open google", "go to google", "open google.com"): ("google-chrome https://google.com", "Opening Google."),
    ("open github", "go to github", "open github.com"): ("google-chrome https://github.com", "Opening GitHub."),
    ("open reddit", "go to reddit", "open reddit.com"): ("google-chrome https://reddit.com", "Opening Reddit."),
    ("open twitter", "go to twitter", "open x.com", "open x"): ("google-chrome https://x.com", "Opening X."),
    ("open netflix", "go to netflix", "open netflix.com"): ("google-chrome https://netflix.com", "Opening Netflix."),
    ("open spotify", "go to spotify", "open spotify.com"): ("google-chrome https://open.spotify.com", "Opening Spotify."),
    ("open discord", "go to discord", "open discord.com"): ("google-chrome https://discord.com/app", "Opening Discord."),
    ("open chatgpt", "go to chatgpt", "open chat gpt"): ("google-chrome https://chat.openai.com", "Opening ChatGPT."),
    ("open maps", "go to maps", "open google maps"): ("google-chrome https://maps.google.com", "Opening Maps."),

    # Chrome — generic
    ("open chrome", "launch chrome", "open google chrome", "launch google chrome"): ("google-chrome", "Got it."),

    # Apps
    ("open discord app", "launch discord"): ("discord", "Launching Discord."),
    ("open spotify app", "launch spotify"): ("spotify", "Launching Spotify."),
    ("open vs code", "open vscode", "launch vs code", "launch vscode", "open code"): ("code", "Opening VS Code."),
    ("open files", "open file manager", "open nautilus"): ("nautilus", "Opening file manager."),

    # System
    ("what time is it", "what's the time", "tell me the time"): ("date '+%I:%M %p'", None),
    ("what's the date", "what is today", "what day is it"): ("date '+%A, %B %d %Y'", None),
    ("how much space do i have", "check disk space", "check storage"): ("df -h --output=avail,pcent / | tail -1", None),
    ("what's my ip", "what is my ip", "show my ip"): ("hostname -I | awk '{print $1}'", None),
    ("show running apps", "what's running", "list processes"): ("ps aux --sort=-%mem | head -10 | awk '{print $11}' | tail -9", None),
}


# Flatten to simple dicts: phrase -> (command, reply)
_FLAT: dict[str, tuple[str, str | None]] = {}
for phrases, entry in SHORTCUTS.items():
    for phrase in phrases:
        _FLAT[phrase.lower().strip()] = entry


def _normalize(text: str) -> str:
    return text.lower().strip().rstrip("?.!")


def match_shortcut(text: str) -> tuple[str, str | None] | None:
    """Return (command, spoken_reply) for a known phrase, or None if not matched.
    spoken_reply is None for commands whose output should be read back directly.
    """
    normalized = _normalize(text)

    if normalized in _FLAT:
        return _FLAT[normalized]

    # Dynamic: "open <site>" / "go to <site>" → chrome to that URL
    for prefix in ("open ", "go to ", "navigate to ", "search for "):
        if normalized.startswith(prefix):
            site = normalized[len(prefix):].strip()
            if site and " " not in site:
                url = site if "." in site else f"{site}.com"
                label = site.split(".")[0].capitalize()
                return f"google-chrome https://{url}", f"Opening {label}."
            if site:
                query = site.replace(" ", "+")
                label = site.title()
                return f"google-chrome 'https://www.google.com/search?q={query}'", f"Searching for {site}."

    return None
