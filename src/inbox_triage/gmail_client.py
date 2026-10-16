"""Gmail API wrapper — deliberately narrow.

Security posture (see README "Guardrails" section for the full writeup):

- OAuth scopes are `gmail.readonly` + `gmail.compose` + `gmail.modify`.
  Gmail has no draft-only scope, and `gmail.compose` can also send.
  `gmail.modify` is here only so a processed message can be labelled
  `triaged`; that scope can also trash mail. "Never sends" and "never
  deletes" are enforced by this class exposing no send, trash, or delete
  method (plus the approval gate), not by the token itself.
- Token is cached to a local, gitignored file and refreshed silently;
  it's never logged or printed.
- Every fetch and every draft creation goes through AuditLogger.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .guardrails import AuditLogger

# gmail.compose can also send. gmail.modify can also trash. This class
# never exposes send, trash, or delete; a human presses send in Gmail.
# modify is required to create the "triaged" label and apply it.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
]

TRIAGED_LABEL = "triaged"


@dataclass
class EmailMessage:
    id: str
    thread_id: str
    sender: str
    subject: str
    snippet: str
    body: str


class GmailClient:
    def __init__(
        self,
        credentials_path: Path,
        token_path: Path,
        audit_logger: AuditLogger,
    ) -> None:
        self._credentials_path = credentials_path
        self._token_path = token_path
        self._audit = audit_logger
        self._service = None
        self._triaged_label_id: str | None = None

    def _get_credentials(self) -> Credentials:
        creds: Credentials | None = None
        if self._token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self._token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self._credentials_path.exists():
                    raise FileNotFoundError(
                        f"Gmail OAuth client file not found at {self._credentials_path}. "
                        "Download it from Google Cloud Console > Credentials and place it there."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self._credentials_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
            self._token_path.write_text(creds.to_json())
            # Belt-and-suspenders: token file should never be world/group
            # readable even if the OS default umask is loose.
            self._token_path.chmod(0o600)
        return creds

    def _get_service(self):
        if self._service is None:
            creds = self._get_credentials()
            self._service = build("gmail", "v1", credentials=creds)
        return self._service

    def fetch_unread(
        self, max_results: int, query: str | None = None
    ) -> list[EmailMessage]:
        service = self._get_service()
        list_args: dict[str, Any] = {
            "userId": "me",
            "labelIds": ["INBOX", "UNREAD"],
            "maxResults": max_results,
            # Gmail drops already-labelled mail. The id check below is a
            # second guard for a list call that still returns one.
            "q": _exclude_triaged(query),
        }
        resp = service.users().messages().list(**list_args).execute()
        message_refs = resp.get("messages", []) if isinstance(resp, dict) else []

        triaged_id = self._lookup_triaged_label_id()
        messages: list[EmailMessage] = []
        skipped: list[str] = []
        for ref in message_refs:
            full = (
                service.users()
                .messages()
                .get(userId="me", id=ref["id"], format="full")
                .execute()
            )
            if triaged_id and triaged_id in full.get("labelIds", []):
                skipped.append(full["id"])
                continue
            messages.append(self._parse_message(full))

        details: dict[str, Any] = {
            "count": len(messages),
            "message_ids": [m.id for m in messages],
        }
        if skipped:
            details["skipped_triaged"] = skipped
        self._audit.log("email_fetched", **details)
        return messages

    def apply_triaged_label(self, message_id: str) -> None:
        """Mark a message handled. Creates the label the first time."""
        service = self._get_service()
        label_id = self._ensure_triaged_label_id()
        service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"addLabelIds": [label_id]},
        ).execute()
        self._audit.log("email_labelled", message_id=message_id, label=TRIAGED_LABEL)

    def _lookup_triaged_label_id(self) -> str | None:
        if self._triaged_label_id:
            return self._triaged_label_id
        service = self._get_service()
        resp = service.users().labels().list(userId="me").execute()
        if not isinstance(resp, dict):
            return None
        for label in resp.get("labels", []):
            if isinstance(label, dict) and label.get("name") == TRIAGED_LABEL:
                self._triaged_label_id = str(label["id"])
                return self._triaged_label_id
        return None

    def _ensure_triaged_label_id(self) -> str:
        found = self._lookup_triaged_label_id()
        if found:
            return found
        service = self._get_service()
        created = (
            service.users()
            .labels()
            .create(
                userId="me",
                body={
                    "name": TRIAGED_LABEL,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                },
            )
            .execute()
        )
        self._triaged_label_id = str(created["id"])
        return self._triaged_label_id

    @staticmethod
    def _parse_message(raw: dict[str, Any]) -> EmailMessage:
        headers = {h["name"]: h["value"] for h in raw["payload"].get("headers", [])}
        body = GmailClient._extract_body(raw["payload"])
        return EmailMessage(
            id=raw["id"],
            thread_id=raw["threadId"],
            sender=headers.get("From", "unknown"),
            subject=headers.get("Subject", "(no subject)"),
            snippet=raw.get("snippet", ""),
            body=body,
        )

    @staticmethod
    def _extract_body(payload: dict[str, Any]) -> str:
        if payload.get("mimeType") == "text/plain" and "data" in payload.get(
            "body", {}
        ):
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode(
                "utf-8", errors="replace"
            )
        for part in payload.get("parts", []) or []:
            body = GmailClient._extract_body(part)
            if body:
                return body
        return ""

    def create_draft_reply(
        self, thread_id: str, to: str, subject: str, body_text: str
    ) -> str:
        """Create a Gmail draft. Does NOT send. Returns the draft id."""
        service = self._get_service()
        message = MIMEText(body_text)
        message["to"] = to
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        draft = (
            service.users()
            .drafts()
            .create(
                userId="me",
                body={"message": {"raw": raw, "threadId": thread_id}},
            )
            .execute()
        )
        draft_id = draft["id"]
        self._audit.log(
            "draft_created",
            draft_id=draft_id,
            thread_id=thread_id,
            to=to,
            subject=subject,
        )
        return draft_id


def _exclude_triaged(query: str | None) -> str:
    exclusion = f"-label:{TRIAGED_LABEL}"
    if not query:
        return exclusion
    if exclusion in query:
        return query
    return f"({query}) {exclusion}"
