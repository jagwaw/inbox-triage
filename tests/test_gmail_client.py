import base64
from unittest.mock import MagicMock

from inbox_triage.agent import TriageAgent
from inbox_triage.gmail_client import GmailClient
from inbox_triage.guardrails import AuditLogger
from inbox_triage.llm import DraftCheck, TriageClassification


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
    assert kwargs["q"] == '(subject:"Invoice request") -label:triaged'
    assert kwargs["labelIds"] == ["INBOX", "UNREAD"]


def test_fetch_unread_without_query_still_excludes_triaged(tmp_path):
    client, service = make_client(tmp_path)

    client.fetch_unread(max_results=5)

    kwargs = service.users().messages().list.call_args.kwargs
    assert kwargs["q"] == "-label:triaged"
    assert kwargs["labelIds"] == ["INBOX", "UNREAD"]


def test_fetch_does_not_double_the_triaged_exclusion(tmp_path):
    client, service = make_client(tmp_path)

    client.fetch_unread(max_results=5, query="subject:hi -label:triaged")

    assert (
        service.users().messages().list.call_args.kwargs["q"]
        == "subject:hi -label:triaged"
    )


def _plain_message(
    message_id: str, body: str, label_ids: list[str] | None = None
) -> dict:
    encoded = base64.urlsafe_b64encode(body.encode()).decode()
    return {
        "id": message_id,
        "threadId": f"thread-{message_id}",
        "labelIds": label_ids or ["INBOX", "UNREAD"],
        "snippet": body[:40],
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "a@example.com"},
                {"name": "Subject", "value": "Hello"},
            ],
            "body": {"data": encoded},
        },
    }


def test_fetch_skips_message_that_already_has_triaged_label(tmp_path):
    client, service = make_client(tmp_path)
    service.users().labels().list().execute.return_value = {
        "labels": [{"id": "Label_9", "name": "triaged"}]
    }
    service.users().messages().list().execute.return_value = {
        "messages": [{"id": "msg1"}]
    }
    service.users().messages().get().execute.return_value = _plain_message(
        "msg1", "Hi", ["INBOX", "UNREAD", "Label_9"]
    )

    assert client.fetch_unread(max_results=5) == []


def test_apply_triaged_label_creates_label_when_missing(tmp_path):
    client, service = make_client(tmp_path)
    service.users().labels().list().execute.return_value = {"labels": []}
    service.users().labels().create().execute.return_value = {
        "id": "Label_1",
        "name": "triaged",
    }
    service.users().messages().modify().execute.return_value = {}

    client.apply_triaged_label("msg1")

    created = service.users().labels().create.call_args.kwargs["body"]
    assert created["name"] == "triaged"
    modify = service.users().messages().modify.call_args.kwargs
    assert modify["id"] == "msg1"
    assert modify["body"] == {"addLabelIds": ["Label_1"]}


def test_apply_triaged_label_reuses_existing_label(tmp_path):
    client, service = make_client(tmp_path)
    service.users().labels().list().execute.return_value = {
        "labels": [{"id": "Label_9", "name": "triaged"}]
    }
    service.users().messages().modify().execute.return_value = {}

    client.apply_triaged_label("msg1")

    service.users().labels().create.assert_not_called()
    assert service.users().messages().modify.call_args.kwargs["body"] == {
        "addLabelIds": ["Label_9"]
    }


class _Exec:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class FakeGmailService:
    """In-memory Gmail API. list() ignores the search query on purpose,
    so a test only passes if GmailClient itself skips labelled mail."""

    def __init__(self, messages: dict[str, dict]):
        # Not named messages/labels: those names are the Gmail API methods.
        self.mailbox = messages
        self.label_ids: dict[str, str] = {}
        self.draft_count = 0

    def users(self):
        return self

    def labels(self):
        return _LabelsApi(self)

    def messages(self):
        return _MessagesApi(self)

    def drafts(self):
        return _DraftsApi(self)


class _LabelsApi:
    def __init__(self, service: FakeGmailService):
        self.service = service

    def list(self, **kwargs):
        labels = [
            {"id": label_id, "name": name}
            for name, label_id in self.service.label_ids.items()
        ]
        return _Exec({"labels": labels})

    def create(self, **kwargs):
        name = kwargs["body"]["name"]
        label_id = f"Label_{len(self.service.label_ids) + 1}"
        self.service.label_ids[name] = label_id
        return _Exec({"id": label_id, "name": name})


class _MessagesApi:
    def __init__(self, service: FakeGmailService):
        self.service = service

    def list(self, **kwargs):
        refs = [
            {"id": message_id}
            for message_id, message in self.service.mailbox.items()
            if "INBOX" in message.get("labelIds", [])
            and "UNREAD" in message.get("labelIds", [])
        ]
        return _Exec({"messages": refs})

    def get(self, **kwargs):
        return _Exec(self.service.mailbox[kwargs["id"]])

    def modify(self, **kwargs):
        message = self.service.mailbox[kwargs["id"]]
        for label_id in kwargs["body"].get("addLabelIds", []):
            if label_id not in message["labelIds"]:
                message["labelIds"].append(label_id)
        return _Exec({"id": kwargs["id"]})


class _DraftsApi:
    def __init__(self, service: FakeGmailService):
        self.service = service

    def create(self, **kwargs):
        self.service.draft_count += 1
        return _Exec({"id": f"draft-{self.service.draft_count}"})


def test_rerun_does_not_reprocess_the_same_email(tmp_path):
    service = FakeGmailService(
        {"msg1": _plain_message("msg1", "Could you confirm the delivery address?")}
    )
    client = GmailClient(
        credentials_path=tmp_path / "credentials.json",
        token_path=tmp_path / "token.json",
        audit_logger=AuditLogger(log_dir=tmp_path),
    )
    client._service = service
    llm = MagicMock()
    llm.check_draft.return_value = DraftCheck(injected=False)
    llm.classify_email.return_value = TriageClassification(
        category="needs_reply",
        priority="medium",
        summary="Asks to confirm an address.",
        suggested_reply="Yes, that address is correct.",
    )
    agent = TriageAgent(
        settings=MagicMock(max_emails_per_run=25, gmail_query=None),
        gmail=client,
        llm=llm,
        audit=AuditLogger(log_dir=tmp_path),
    )

    first = agent.run()
    second = agent.run()

    assert len(first) == 1
    assert first[0]["draft_id"] == "draft-1"
    assert second == []
    assert llm.classify_email.call_count == 1
    assert service.draft_count == 1
    assert "triaged" in service.label_ids
    assert service.label_ids["triaged"] in service.mailbox["msg1"]["labelIds"]
