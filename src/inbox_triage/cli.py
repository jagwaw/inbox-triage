"""Command-line entrypoint.

Usage:
    python -m inbox_triage.cli

Runs one triage pass over unread inbox mail, prints a summary, and exits.
Nothing here loops or schedules itself — running this on a cadence (cron,
a scheduled task, etc.) is a deliberate choice left to the operator, not
something the agent does to itself.
"""
from __future__ import annotations

import sys

from .agent import build_agent
from .config import ConfigError, load_settings


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    agent = build_agent(settings)

    print(f"Fetching up to {settings.max_emails_per_run} unread emails...")
    results = agent.run()

    print(f"\nProcessed {len(results)} email(s):\n")
    for r in results:
        email = r["email"]
        if r["error"]:
            print(f"  [ERROR] {email.subject!r} from {email.sender}: {r['error']}")
            continue

        c = r["classification"]
        flag = " [INJECTION FLAGGED]" if r["injection_flagged"] else ""
        draft_note = f" -> draft {r['draft_id']} created" if r["draft_id"] else ""
        print(f"  [{c.priority.upper():6}] {c.category:20} {email.subject!r}{flag}{draft_note}")
        print(f"           {c.summary}")

    print(f"\nAudit log written to: {settings.audit_log_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
