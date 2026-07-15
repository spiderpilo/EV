import json
from datetime import datetime
from pathlib import Path

from ev.config import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "data" / "conversations"


class ConversationLogger:
    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        self._path = LOG_DIR / f"{date_str}.jsonl"

    def log(self, user: str, assistant: str):
        if not user.strip() or not assistant.strip():
            return
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user.strip(),
            "assistant": assistant.strip(),
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")
