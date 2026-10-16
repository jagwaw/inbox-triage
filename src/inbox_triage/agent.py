"""The triage agent graph.

Flow (LangGraph state machine):

    fetch_emails -> [for each email] -> screen_input -> classify -> draft_if_needed -> record

Each node's only job is the thing its name says, and every state
transition is a line in the audit log. This is what "security built in,
not bolted on" looks like in code: the guardrails aren't a wrapper around
the agent, they're nodes IN the graph, so there's no path through the
graph that skips them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from .config import Settings
from .gmail_client import EmailMessage, GmailClient
from .guardrails import AuditLogger, RateLimiter, sanitize_for_prompt
from .llm import LLMClient, TriageClassification


class TriageResult(TypedDict):
    email: EmailMessage
    classification: TriageClassification | None
    draft_id: str | None
    injection_flagged: bool
    # True when the model's draft itself tripped the injection screen.
    output_flagged: bool
    # True when the second-opinion draft check found injected text (or
    # could not run, which also blocks the draft).
    draft_check_flagged: bool
    error: str | None


class AgentState(TypedDict):
    emails: list[EmailMessage]
    results: list[TriageResult]


@dataclass
class TriageAgent:
    settings: Settings
    gmail: GmailClient
    llm: LLMClient
    audit: AuditLogger

    def _fetch_node(self, state: AgentState) -> AgentState:
        emails = self.gmail.fetch_unread(
            max_results=self.settings.max_emails_per_run,
            query=self.settings.gmail_query,
        )
        return {"emails": emails, "results": []}

    def _process_node(self, state: AgentState) -> AgentState:
        results: list[TriageResult] = []
        for email in state["emails"]:
            result = self._process_one(email)
            results.append(result)
        return {"emails": state["emails"], "results": results}

    def _process_one(self, email: EmailMessage) -> TriageResult:
        sanitized = sanitize_for_prompt(email.body)
        if sanitized.flagged:
            self.audit.log(
                "input_flagged",
                message_id=email.id,
                patterns=sanitized.matched_patterns,
            )

        try:
            classification = self.llm.classify_email(
                sender=email.sender,
                subject=email.subject,
                body=sanitized.text,
                injection_flagged=sanitized.flagged,
            )
            self.audit.log(
                "email_classified",
                message_id=email.id,
                category=classification.category,
                priority=classification.priority,
            )
        except Exception as exc:  # noqa: BLE001 - a triage failure shouldn't crash the run
            self.audit.log("error", message_id=email.id, error=str(exc))
            # A flagged email is done: retrying it would keep sending the
            # attack to the model. An ordinary failure stays unlabelled.
            if sanitized.flagged:
                self._mark_triaged(email)
            return TriageResult(
                email=email,
                classification=None,
                draft_id=None,
                injection_flagged=sanitized.flagged,
                output_flagged=False,
                draft_check_flagged=False,
                error=str(exc),
            )

        draft_id = None
        output_flagged = False
        draft_check_flagged = False
        draft_check_error = False
        if classification.category == "needs_reply" and classification.suggested_reply:
            # The model's draft is screened too: an injection the input
            # screen missed can still show up in what the model wrote.
            output_flagged = sanitize_for_prompt(classification.suggested_reply).flagged
            # Guardrail: never draft a reply for an email whose body tripped
            # the injection screen, even if the model still tried to comply.
            # A human should look at these by hand.
            if sanitized.flagged:
                self.audit.log(
                    "draft_skipped", message_id=email.id, reason="injection_flagged"
                )
            elif output_flagged:
                self.audit.log(
                    "draft_skipped", message_id=email.id, reason="output_flagged"
                )
            else:
                outcome = self._draft_check_flags(
                    email, sanitized.text, classification.suggested_reply
                )
                if outcome == "clear":
                    draft_id = self.gmail.create_draft_reply(
                        thread_id=email.thread_id,
                        to=email.sender,
                        subject=f"Re: {email.subject}",
                        body_text=classification.suggested_reply,
                    )
                else:
                    draft_check_flagged = True
                    draft_check_error = outcome == "error"

        # Label when handling finished: a draft was saved, the email was
        # flagged, or there was nothing to draft. A draft-check outage stays
        # unlabelled so the next run can try again.
        if not draft_check_error:
            self._mark_triaged(email)

        return TriageResult(
            email=email,
            classification=classification,
            draft_id=draft_id,
            injection_flagged=sanitized.flagged,
            output_flagged=output_flagged,
            draft_check_flagged=draft_check_flagged,
            error=None,
        )

    def _mark_triaged(self, email: EmailMessage) -> None:
        try:
            self.gmail.apply_triaged_label(email.id)
        except Exception as exc:  # noqa: BLE001 - a label failure must not abort the run
            self.audit.log(
                "error",
                message_id=email.id,
                error=f"could not apply triaged label: {exc}",
            )

    def _draft_check_flags(
        self, email: EmailMessage, body: str, draft: str
    ) -> Literal["clear", "injected", "error"]:
        """Ask a second, narrowly-scoped LLM call whether the draft carries
        text the email smuggled in. Fails closed: if the check can't run,
        no draft is saved and the email is left for a human."""
        try:
            check = self.llm.check_draft(
                sender=email.sender, subject=email.subject, body=body, draft=draft
            )
        except Exception as exc:  # noqa: BLE001 - fail closed, never crash the run
            self.audit.log(
                "draft_skipped",
                message_id=email.id,
                reason="draft_check_error",
                error=str(exc),
            )
            return "error"
        if check.injected:
            self.audit.log(
                "draft_skipped",
                message_id=email.id,
                reason="draft_check_flagged",
                evidence=check.evidence,
            )
            return "injected"
        return "clear"

    def build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("fetch_emails", self._fetch_node)
        graph.add_node("process_emails", self._process_node)
        graph.set_entry_point("fetch_emails")
        graph.add_edge("fetch_emails", "process_emails")
        graph.add_edge("process_emails", END)
        return graph.compile()

    def run(self) -> list[TriageResult]:
        app = self.build_graph()
        final_state: AgentState = app.invoke({"emails": [], "results": []})
        return final_state["results"]


def build_agent(settings: Settings) -> TriageAgent:
    audit = AuditLogger(log_dir=settings.audit_log_dir)
    gmail = GmailClient(
        credentials_path=settings.gmail_credentials_path,
        token_path=settings.gmail_token_path,
        audit_logger=audit,
    )
    rate_limiter = RateLimiter(max_calls_per_hour=settings.max_llm_calls_per_hour)
    llm = LLMClient(settings=settings, rate_limiter=rate_limiter)
    return TriageAgent(settings=settings, gmail=gmail, llm=llm, audit=audit)
