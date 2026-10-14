"""Security & guardrail primitives.

Everything in this module exists to answer one question the way the job
posting frames it: "assume the client will do something unwise with the
tools you give them, and build so it doesn't hurt them." Concretely:

- RateLimiter: hard caps on LLM spend so a bug (e.g. an infinite loop, a
  prompt-injected email asking the agent to "reprocess this 1000 times")
  can't run up an unbounded bill.
- AuditLogger: every agent action (classification, draft created, error,
  rate-limit trip) is appended to a local, timestamped, append-only JSONL
  log. This is what makes "what did the agent actually do" answerable
  after the fact, for both debugging and for a client asking "did it read
  anything it shouldn't have."
- ApprovalGate: the agent is never allowed to send email, delete anything,
  or take an irreversible action on its own. It can only produce a Gmail
  *draft*. A human always presses send.
- sanitize_for_prompt: a minimal prompt-injection defense — email bodies
  are untrusted input, so before they're interpolated into a prompt we
  strip/flag common injection patterns ("ignore previous instructions",
  fake system/role markers) rather than trusting the model to ignore them
  unaided.
"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


class RateLimitExceeded(RuntimeError):
    """Raised when a caller would exceed the configured LLM call budget."""


class RateLimiter:
    """Simple sliding-window rate limiter for LLM calls.

    Not distributed, not persisted across restarts — deliberately so, for
    a single-process side project. In a real deployment this would be
    backed by Redis or a database so the cap holds across replicas/restarts;
    the interface here is written so that swap is a one-file change.
    """

    def __init__(self, max_calls_per_hour: int) -> None:
        self.max_calls_per_hour = max_calls_per_hour
        self._timestamps: deque[float] = deque()

    def check_and_record(self) -> None:
        now = time.monotonic()
        window_start = now - 3600
        while self._timestamps and self._timestamps[0] < window_start:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_calls_per_hour:
            raise RateLimitExceeded(
                f"LLM call budget exceeded: {self.max_calls_per_hour}/hour. "
                "Refusing to make another call this window."
            )
        self._timestamps.append(now)

    @property
    def calls_in_window(self) -> int:
        return len(self._timestamps)


AuditEventType = Literal[
    "email_fetched",
    "email_classified",
    "draft_created",
    "draft_skipped",
    "rate_limit_tripped",
    "input_flagged",
    "error",
]


@dataclass
class AuditLogger:
    """Append-only JSONL audit log — one line per agent action.

    Deliberately not a mutable-in-place log: audit trails that can be
    edited after the fact aren't audit trails. Each run gets its own file
    named by UTC date so logs are easy to rotate/ship.
    """

    log_dir: Path

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"{date_str}.jsonl"

    def log(self, event_type: AuditEventType, **details: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **details,
        }
        with self._log_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


# --- Prompt-injection defense -------------------------------------------------

_INJECTION_PATTERNS = [
    # \s+ (not a literal space) so a line break or extra spaces between
    # words can't be used to slip a phrase past the detector.
    re.compile(r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above)", re.I),
    # Role reassignment ("you are now in admin mode", "you are now an
    # unrestricted assistant"), not ordinary "you are now subscribed".
    re.compile(
        r"you\s+are\s+now\s+(in\s+\w+\s+mode|an?\s+(\w+\s+)?(ai|assistant|bot|model)\b"
        r"|unrestricted|jailbroken)",
        re.I,
    ),
    re.compile(r"new\s+instructions?:", re.I),
    # A fake role label at the start of a line. Mid-line "Operating system:"
    # is ordinary text.
    re.compile(r"^\s*system\s*:", re.I | re.M),
    re.compile(r"</?(system|assistant|user)>", re.I),
    # Bulk exfiltration ("forward all future emails to ..."). Forwarding one
    # email to a colleague is ordinary.
    re.compile(r"forward\s+(all|every|any)\s+(of\s+)?(\w+\s+)?(emails?|mail|messages)\s+to", re.I),
]


@dataclass
class SanitizedInput:
    text: str
    flagged: bool
    matched_patterns: list[str] = field(default_factory=list)


def sanitize_for_prompt(raw_text: str, *, max_len: int = 8000) -> SanitizedInput:
    """Screen untrusted email content before it reaches a prompt.

    This is intentionally a *detector*, not a silent stripper: an email
    that matches these patterns is still passed to the model (wrapped in
    clear delimiters and labelled as untrusted data, never as
    instructions), but it's flagged so the audit log — and a human
    reviewer — can see that this message tried something.
    """
    truncated = raw_text[:max_len]
    matched = [p.pattern for p in _INJECTION_PATTERNS if p.search(truncated)]
    return SanitizedInput(text=truncated, flagged=bool(matched), matched_patterns=matched)


class ApprovalGate:
    """The agent's one-way door: nothing irreversible happens without this.

    In v0.1 this is a no-op that always denies auto-send (see
    config.Settings.validate, which refuses to even start if
    REQUIRE_HUMAN_APPROVAL is false). It exists as its own class so the
    "how do I ship a human-in-the-loop check" story is a single, visible
    seam rather than an `if` scattered through the agent graph.
    """

    def __init__(self, *, require_approval: bool) -> None:
        if not require_approval:
            raise ValueError("Auto-approval is not supported in this project.")
        self.require_approval = require_approval

    def may_auto_send(self) -> bool:
        return False
