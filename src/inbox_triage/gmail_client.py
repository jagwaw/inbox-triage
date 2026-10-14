"""Gmail API wrapper — deliberately narrow.

Security posture (see README "Guardrails" section for the full writeup):

- OAuth scopes are `gmail.readonly` + `gmail.compose`. Gmail has no
  draft-only scope, and `gmail.compose` can also send, so "never sends"
  is enforced by this class exposing no send method (plus the approval
  gate), not by the token itself.
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

# Narrowest scopes that allow reading mail and creating drafts.
# gmail.compose technically also permits sending; this class simply never
# exposes a send method, and a human presses send in Gmail.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]


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

    def fetch_unread(self, max_results: int) -> list[EmailMessage]:
        service = self._get_service()
        resp = (
            service.users()
            .messages()
            .list(userId="me", labelIds=["INBOX", "UNREAD"], maxResults=max_results)
            .execute()
        )
        message_refs = resp.get("messages", [])

        messages: list[EmailMessage] = []
        for ref in message_refs:
            full = (
                service.users()
                .messages()
                .get(userId="me", id=ref["id"], format="full")
                .execute()
            )
            messages.append(self._parse_message(full))

        self._audit.log(
            "email_fetched",
            count=len(messages),
            message_ids=[m.id for m in messages],
        )
        return messages

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
        if payload.get("mimeType") == "text/plain" and "data" in payload.get("body", {}):
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode(
                "utf-8", errors="replace"
            )
        for part in payload.get("parts", []) or []:
            body = GmailClient._extract_body(part)
            if body:
                return body
        return ""

    def create_draft_reply(self, thread_id: str, to: str, subject: str, body_text: str) -> str:
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
