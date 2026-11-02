"""Small, dependency-free guards for a publicly hosted demo.

* input validation   - whitelist names/tags before they touch the filesystem or SQL
* sanitize_text      - strip anything but safe characters from free text (run names)
* rate limiting      - per-session cooldowns + a global cap on MLflow runs
* admin check        - constant-time password comparison for destructive actions
"""
from __future__ import annotations

import hmac
import os
import re
import time

TAG_RE = re.compile(r"^v\d+\.\d+$")
DB_NAME_RE = re.compile(r"^mlflow(-\d{10})?\.db$")
SAFE_TEXT_RE = re.compile(r"[^A-Za-z0-9 _.,:|()\-]")

MAX_RUNS = int(os.environ.get("MAX_MLFLOW_RUNS", "80"))      # global cap on stored runs
COOLDOWN_S = float(os.environ.get("ACTION_COOLDOWN_S", "8"))  # per-session, per-action


def valid_tag(tag: str) -> str:
    if not isinstance(tag, str) or not TAG_RE.match(tag):
        raise ValueError(f"Invalid data version tag: {tag!r}")
    return tag


def sanitize_text(text: str, max_len: int = 60, default: str = "run") -> str:
    cleaned = SAFE_TEXT_RE.sub("", (text or "")).strip()[:max_len]
    return cleaned or default


def cooldown_ok(state, action: str, seconds: float = COOLDOWN_S) -> tuple[bool, float]:
    """Return (allowed, seconds_left) and record the attempt when allowed."""
    key = f"_last_{action}"
    now = time.monotonic()
    left = seconds - (now - state.get(key, -1e9))
    if left > 0:
        return False, left
    state[key] = now
    return True, 0.0


def check_admin(entered: str | None, expected: str | None) -> bool:
    if not expected or not entered:
        return False
    return hmac.compare_digest(entered.encode(), expected.encode())
