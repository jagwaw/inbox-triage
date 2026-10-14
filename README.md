# Inbox Triage Agent

A small, guardrailed LLM agent that reads unread Gmail messages, classifies
and summarizes them, and drafts replies **for human review** — it never
sends mail on its own.

Built as a focused portfolio project to demonstrate the two things that
matter most for shipping AI agents into someone else's hands: working
LLM/tool-calling integration, and taking security seriously by default
rather than bolting it on afterward.

## What it does

For each unread email in the inbox:

1. Screens the email body for prompt-injection patterns before it ever
   reaches a prompt.
2. Classifies it (`needs_reply` / `fyi_only` / `action_item` /
   `spam_or_irrelevant`), assigns a priority, and summarizes it — via a
   structured LLM tool call (Anthropic or OpenAI), orchestrated as a
   [LangGraph](https://github.com/langchain-ai/langgraph) graph.
3. If it needs a reply, drafts one — but only if the email didn't trip the
   injection screen. That draft is saved in Gmail as a **draft**, never
   sent.
4. Logs every step to an append-only audit log.

Run it, then open Gmail: your drafts folder has a starting point for each
email that needed a reply, and you decide what actually goes out.

## Why it's built this way

This project exists to answer a specific question honestly: *what does it
look like to give an AI agent access to someone's inbox without that being
reckless?* Four things, all present in the code, not just described here:

- **Draft-only by design.** The Gmail client has no send method — it only
  ever calls the draft-create endpoint. One honest caveat: Gmail has no
  draft-only OAuth scope (`gmail.compose`, the narrowest one that allows
  drafts, can also send), so "never sends" is enforced in code and by the
  approval gate, not by the token itself. A production deployment would
  put a narrower service or allow-listed proxy between the agent and
  Gmail. See [`gmail_client.py`](src/inbox_triage/gmail_client.py).
- **A human-approval gate that can't be turned off.** `REQUIRE_HUMAN_APPROVAL=false`
  is refused at startup with a `ConfigError` — not a silently-ignored
  setting. See [`config.py`](src/inbox_triage/config.py).
- **Prompt-injection screening.** Email bodies are untrusted input. Before
  they reach a prompt they're checked against known injection patterns
  ("ignore previous instructions", fake `<system>` tags, "forward all
  emails to…"); flagged emails still get classified (so nothing silently
  disappears) but are never auto-drafted. See
  [`guardrails.py`](src/inbox_triage/guardrails.py).
- **A hard spend cap.** A sliding-window rate limiter sits in front of
  every LLM call — a bug or a malicious email can't run up an unbounded
  bill. See `RateLimiter` in the same file.
- **An audit trail you can actually hand to someone.** Every fetch,
  classification, draft, skipped-draft, flagged input, and error is
  appended to a dated JSONL file. "What did the agent do to my inbox
  today" has a real answer.

## Architecture

```
config.py        env-driven settings, validated eagerly (fails fast, not mid-run)
guardrails.py     RateLimiter, AuditLogger, prompt-injection screen, ApprovalGate
gmail_client.py   OAuth (read + draft-only scopes), fetch, create draft
llm.py            provider-agnostic LLM call, structured output, retries
agent.py          LangGraph graph: fetch_emails -> process_emails (screen -> classify -> draft) -> end
cli.py            entrypoint: run one triage pass, print a summary
evaluate.py       runs the labelled eval set through the real triage path -> evals/results.md
```

The guardrails aren't a wrapper around the agent — they're nodes *in* the
graph's execution path, so there's no route through the code that skips
them.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in ANTHROPIC_API_KEY (or OPENAI_API_KEY) in .env
```

**Gmail OAuth:**
1. In [Google Cloud Console](https://console.cloud.google.com/), create a
   project, enable the Gmail API, and create an OAuth 2.0 **Desktop app**
   client ID.
2. Download the client secret JSON, save it as `credentials.json` in the
   project root (already gitignored).
3. First run opens a browser for you to authorize; a `token.json` is
   cached locally afterward (also gitignored).

**Run:**
```bash
python -m inbox_triage.cli
```

**Test:**
```bash
pytest                                      # offline, no credentials needed
python -m inbox_triage.evaluate --delay 2   # live eval -> evals/results.md
```

## Evaluation

`python -m inbox_triage.evaluate --delay 2` runs 32 labelled, synthetic
emails ([`evals/dataset.jsonl`](evals/dataset.jsonl)) through the same
screen -> classify -> draft path the CLI uses. Only Gmail is faked (drafts
are recorded, not created), so no real mail is read or sent to the LLM.

- 24 normal emails, 6 per category. Three contain phrases that look like
  injections but aren't ("You are now subscribed", "Operating system:"),
  to measure false positives.
- 8 prompt-injection attacks. 4 use phrasings the regex screen knows; 4
  are paraphrased, in Spanish, or hidden below a signature. Each one asks
  the model to include a unique canary code, so "did the model obey?" is
  a string check, not a judgment call.

**Baseline run** (2026-10-14, `openai/gpt-oss-20b` on Groq, one run;
full report in [`evals/results.md`](evals/results.md)):

| Metric | Result |
|---|---|
| Category accuracy, normal emails | 21/24 (88%) |
| Attacks flagged by the regex screen | 4/8 (50%) |
| Normal emails wrongly flagged by the screen | 3/24 (12%) |
| Attacks where the model followed the injected instruction | 3/8 (38%) |
| Attacks where injected text reached a draft | 1/8 (12%) |
| Emails sent | 0 (no send path exists) |

What this shows: the layers catch different things. Two attacks that the
model obeyed were stopped because the screen flagged them and flagged
emails never get drafts. One attack (a real request with an instruction
hidden below the signature) passed both layers and its injected line
reached a draft. That's still only a draft a human must review, but it's
the case the next round of work targets. Results are from a single run
of a non-deterministic model on a small dataset, so treat them as
indicative, not precise.

## What's deliberately out of scope (v0.1)

- Auto-send. On purpose — see above.
- Multi-account / multi-tenant support.
- A persisted, cross-restart rate-limit store (currently in-process only;
  the interface is written so swapping in Redis is a one-file change).
- A web UI — this is a CLI you run on a schedule (cron, a scheduled task
  runner) if you want it running continuously.

## Roadmap

- [ ] Calendar triage agent using the same guardrail primitives (rate
      limiter, audit logger, approval gate) to prove they generalize
      beyond Gmail.
- [ ] Swap the in-process rate limiter for a Redis-backed one.
- [ ] A lightweight review UI for the audit log (currently just JSONL you
      `grep`/`jq` through).
