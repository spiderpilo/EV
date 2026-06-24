import subprocess
import shlex

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


def run_command(command: str, timeout: int = 15) -> tuple[bool, str]:
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
