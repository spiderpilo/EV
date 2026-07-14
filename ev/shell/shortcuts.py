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


OPEN_INTENTS = (
    "open", "launch", "start", "pull up", "load", "go to",
    "navigate to", "show me", "bring up", "put on", "watch",
    "can you open", "can you launch", "can you pull up",
    "could you open", "please open", "hey open",
)

# Map of keywords to their shortcut key for intent matching
_KEYWORDS: dict[str, str] = {
    "youtube": "open youtube",
    "gmail": "open gmail",
    "email": "open gmail",
    "github": "open github",
    "reddit": "open reddit",
    "twitter": "open twitter",
    "netflix": "open netflix",
    "spotify": "open spotify",
    "discord": "open discord",
    "chatgpt": "open chatgpt",
    "chat gpt": "open chatgpt",
    "maps": "open maps",
    "google maps": "open maps",
    "chrome": "open chrome",
    "vscode": "open vs code",
    "vs code": "open vs code",
    "code": "open vs code",
    "files": "open files",
    "file manager": "open files",
}


def _normalize(text: str) -> str:
    return text.lower().strip().rstrip("?.!")


def match_shortcut(text: str) -> tuple[str, str | None] | None:
    """Return (command, spoken_reply) for a known phrase, or None if not matched."""
    normalized = _normalize(text)

    # 1. Exact match
    if normalized in _FLAT:
        return _FLAT[normalized]

    # 2. Substring match — "hey can you open netflix for me" contains "open netflix"
    for phrase, entry in _FLAT.items():
        if phrase in normalized:
            return entry

    # 3. Intent + keyword — "I've had a long day, put on netflix" → find "netflix" + open intent
    for keyword, shortcut_key in _KEYWORDS.items():
        if keyword in normalized:
            has_intent = any(intent in normalized for intent in OPEN_INTENTS)
            # also match if it's just the keyword alone (e.g. "netflix")
            is_standalone = normalized.strip() == keyword
            if has_intent or is_standalone:
                if shortcut_key in _FLAT:
                    return _FLAT[shortcut_key]

    # 4. Dynamic: any open intent + single unknown word → chrome URL
    for intent in OPEN_INTENTS:
        if normalized.startswith(intent + " "):
            site = normalized[len(intent):].strip().split()[0] if normalized[len(intent):].strip() else ""
            if site and site.isalpha():
                url = f"{site}.com"
                label = site.capitalize()
                return f"google-chrome https://{url}", f"Opening {label}."

    return None
