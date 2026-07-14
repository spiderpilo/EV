"""
Hard-coded voice shortcuts for terminal mode.

Checked before the LLM so common commands are instant and reliable.
Add new entries to SHORTCUTS or extend match_shortcut() for dynamic patterns.
"""
import random


def _r(*lines: str) -> str:
    return random.choice(lines)


# (phrase tuple) -> (command, reply)
# reply can be a lambda for random variety or None to read output directly
SHORTCUTS: dict[tuple, tuple[str, object]] = {
    # Chrome — exact sites
    ("open youtube", "go to youtube", "open youtube.com"): (
        "google-chrome https://youtube.com",
        lambda: _r("Sure thing Piolo, enjoy!", "You got it!", "Opening YouTube for you."),
    ),
    ("open gmail", "go to gmail", "open gmail.com", "check my email", "open my email"): (
        "google-chrome https://gmail.com",
        lambda: _r("Opening your inbox, Piolo.", "Here's your email.", "You got it!"),
    ),
    ("open google", "go to google", "open google.com"): (
        "google-chrome https://google.com",
        lambda: _r("Opening Google.", "Sure thing!"),
    ),
    ("open github", "go to github", "open github.com"): (
        "google-chrome https://github.com",
        lambda: _r("Opening GitHub. Time to build something.", "You got it, Piolo.", "Here's your GitHub."),
    ),
    ("open reddit", "go to reddit", "open reddit.com"): (
        "google-chrome https://reddit.com",
        lambda: _r("Opening Reddit. Don't get too lost in there.", "Sure thing!", "Here you go."),
    ),
    ("open twitter", "go to twitter", "open x.com", "open x"): (
        "google-chrome https://x.com",
        lambda: _r("Opening X.", "Sure thing, Piolo.", "You got it!"),
    ),
    ("open netflix", "go to netflix", "open netflix.com"): (
        "google-chrome https://netflix.com",
        lambda: _r("Sure thing Piolo, enjoy your movie!", "You got it! Enjoy!", "Netflix it is. Relax, you deserve it."),
    ),
    ("open spotify", "go to spotify", "open spotify.com"): (
        "google-chrome https://open.spotify.com",
        lambda: _r("Opening Spotify. Good vibes incoming.", "You got it! Enjoy the music.", "Sure thing, Piolo!"),
    ),
    ("open discord", "go to discord", "open discord.com"): (
        "google-chrome https://discord.com/app",
        lambda: _r("Opening Discord. Go say hi.", "Sure thing!", "You got it, Piolo."),
    ),
    ("open chatgpt", "go to chatgpt", "open chat gpt"): (
        "google-chrome https://chat.openai.com",
        lambda: _r("Opening ChatGPT. You know you have me though, right?", "Sure thing!", "Here you go."),
    ),
    ("open maps", "go to maps", "open google maps"): (
        "google-chrome https://maps.google.com",
        lambda: _r("Opening Maps. Where are we headed?", "Sure thing, Piolo!", "You got it!"),
    ),

    # Chrome — generic
    ("open chrome", "launch chrome", "open google chrome", "launch google chrome"): (
        "google-chrome",
        lambda: _r("Sure thing!", "You got it!", "Opening Chrome."),
    ),

    # Apps
    ("open discord app", "launch discord"): (
        "discord",
        lambda: _r("Launching Discord. Go say hi.", "Sure thing!", "You got it!"),
    ),
    ("open spotify app", "launch spotify"): (
        "spotify",
        lambda: _r("Launching Spotify. Enjoy the music, Piolo.", "Sure thing!", "You got it!"),
    ),
    ("open vs code", "open vscode", "launch vs code", "launch vscode", "open code"): (
        "code",
        lambda: _r("Opening VS Code. Let's build something.", "You got it, Piolo. Time to code.", "Sure thing!"),
    ),
    ("open files", "open file manager", "open nautilus"): (
        "nautilus",
        lambda: _r("Opening file manager.", "Sure thing!", "You got it!"),
    ),

    # System
    ("what time is it", "what's the time", "tell me the time"): ("date '+%I:%M %p'", None),
    ("what's the date", "what is today", "what day is it"): ("date '+%A, %B %d %Y'", None),
    ("how much space do i have", "check disk space", "check storage"): ("df -h --output=avail,pcent / | tail -1", None),
    ("what's my ip", "what is my ip", "show my ip"): ("hostname -I | awk '{print $1}'", None),
    ("show running apps", "what's running", "list processes"): ("ps aux --sort=-%mem | head -10 | awk '{print $11}' | tail -9", None),
}


# Flatten to simple dicts: phrase -> (command, reply)
_FLAT: dict[str, tuple[str, object]] = {}
for phrases, entry in SHORTCUTS.items():
    for phrase in phrases:
        _FLAT[phrase.lower().strip()] = entry


OPEN_INTENTS = (
    "open", "launch", "start", "pull up", "load", "go to",
    "navigate to", "show me", "bring up", "put on", "watch",
    "can you open", "can you launch", "can you pull up",
    "could you open", "please open", "hey open",
)

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
    "files": "open files",
    "file manager": "open files",
}


def _normalize(text: str) -> str:
    return text.lower().strip().rstrip("?.!")


def _resolve(entry: tuple[str, object]) -> tuple[str, str | None]:
    """Resolve a (command, reply) entry — call lambda if needed."""
    command, reply = entry
    if callable(reply):
        return command, reply()
    return command, reply


def match_shortcut(text: str) -> tuple[str, str | None] | None:
    """Return (command, spoken_reply) for a known phrase, or None if not matched."""
    normalized = _normalize(text)

    # 1. Exact match
    if normalized in _FLAT:
        return _resolve(_FLAT[normalized])

    # 2. Substring match — "hey can you open netflix for me" contains "open netflix"
    for phrase, entry in _FLAT.items():
        if phrase in normalized:
            return _resolve(entry)

    # 3. Intent + keyword — "put on netflix" → find "netflix" + open intent
    for keyword, shortcut_key in _KEYWORDS.items():
        if keyword in normalized:
            has_intent = any(intent in normalized for intent in OPEN_INTENTS)
            is_standalone = normalized.strip() == keyword
            if has_intent or is_standalone:
                if shortcut_key in _FLAT:
                    return _resolve(_FLAT[shortcut_key])

    # 4. Dynamic: open intent + unknown site word → chrome URL
    for intent in OPEN_INTENTS:
        if normalized.startswith(intent + " "):
            site = normalized[len(intent):].strip().split()[0] if normalized[len(intent):].strip() else ""
            if site and site.isalpha():
                url = f"{site}.com"
                label = site.capitalize()
                return f"google-chrome https://{url}", f"Opening {label}."

    return None
