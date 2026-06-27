import subprocess
import shutil
import time

DANGEROUS_PREFIXES = (
    "rm ", "rm\t", "rmdir ",
    "sudo ", "su ",
    "kill ", "killall ", "pkill ",
    "mkfs", "dd ",
    "chmod 777", "chown ",
    "> /dev/", "mv /",
    "shutdown", "reboot", "poweroff",
    ":(){ :|:& };:",
)

HANGING_FLAGS = (" -f", " --follow", " -w", " --watch", " tail -f")

GUI_APPS = (
    "google-chrome", "chromium", "firefox",
    "discord", "spotify", "slack", "telegram",
    "code", "nautilus", "thunar", "nemo",
    "gimp", "vlc", "mpv", "totem",
    "libreoffice", "steam",
    "xdg-open", "gnome-open", "gio open",
    "evince", "eog",
)


def is_dangerous(command: str) -> bool:
    cmd = command.strip().lstrip("&|;")
    for prefix in DANGEROUS_PREFIXES:
        if cmd.startswith(prefix):
            return True
    for part in command.split("|"):
        part = part.strip()
        for prefix in DANGEROUS_PREFIXES:
            if part.startswith(prefix):
                return True
    for flag in HANGING_FLAGS:
        if flag in command:
            return True
    return False


def _is_gui_command(command: str) -> bool:
    cmd = command.strip().split()[0] if command.strip() else ""
    for app in GUI_APPS:
        if cmd.endswith(app) or cmd == app:
            return True
    if command.strip().endswith("&"):
        return True
    return False


def run_command(command: str, timeout: int = 15) -> tuple[bool, str]:
    if _is_gui_command(command):
        return _launch_gui(command)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr.strip():
            output = result.stderr.strip() if not output else output + "\n" + result.stderr.strip()
        return True, output[:2000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return False, "Command timed out after 15 seconds."
    except Exception as e:
        return False, str(e)


def _launch_gui(command: str) -> tuple[bool, str]:
    command = command.rstrip("& ")
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(1.5)
        if proc.poll() is not None and proc.returncode != 0:
            err = proc.stderr.read().decode().strip() if proc.stderr else ""
            return False, err or "App exited immediately."
        app_name = command.strip().split()[0].split("/")[-1]
        return True, f"Launched {app_name} successfully."
    except Exception as e:
        return False, str(e)
