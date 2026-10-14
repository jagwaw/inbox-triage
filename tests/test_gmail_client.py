from unittest.mock import MagicMock

from inbox_triage.gmail_client import GmailClient
from inbox_triage.guardrails import AuditLogger


def make_client(tmp_path):
    client = GmailClient(
        credentials_path=tmp_path / "credentials.json",
        token_path=tmp_path / "token.json",
        audit_logger=AuditLogger(log_dir=tmp_path),
    )
    service = MagicMock()
    service.users().messages().list().execute.return_value = {"messages": []}
    client._service = service
    return client, service


def test_fetch_unread_passes_gmail_query(tmp_path):
    client, service = make_client(tmp_path)

    client.fetch_unread(max_results=5, query='subject:"Invoice request"')

    kwargs = service.users().messages().list.call_args.kwargs
    assert kwargs["q"] == 'subject:"Invoice request"'
    assert kwargs["labelIds"] == ["INBOX", "UNREAD"]


def test_fetch_unread_without_query_reads_all_unread(tmp_path):
    client, service = make_client(tmp_path)

    client.fetch_unread(max_results=5)

    assert "q" not in service.users().messages().list.call_args.kwargs
