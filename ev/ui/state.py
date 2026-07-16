import threading

_lock = threading.Lock()
_current = "idle"
_message = ""


def set_state(s: str, message: str = "") -> None:
    global _current, _message
    with _lock:
        _current = s
        _message = message


def get_state() -> tuple[str, str]:
    with _lock:
        return _current, _message
