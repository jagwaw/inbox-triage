from types import SimpleNamespace

from inbox_triage.llm import CLASSIFY_SYSTEM_PROMPT, LLMClient


def make_client(sender_name=None) -> LLMClient:
    settings = SimpleNamespace(sender_name=sender_name)
    return LLMClient(settings=settings, rate_limiter=None)


def test_prompt_forbids_claiming_attachments_and_inventing_business_facts():
    prompt = CLASSIFY_SYSTEM_PROMPT.lower()
    assert "never claim to have attached" in prompt
    assert "[confirm:" in prompt


def test_signs_with_sender_name_when_set():
    assert "Maria Santos" in make_client("Maria Santos")._system_prompt()


def test_uses_placeholder_when_no_sender_name():
    assert "[Your Name]" in make_client(None)._system_prompt()


def test_draft_check_is_rate_limited_and_marks_email_as_untrusted(monkeypatch):
    from inbox_triage.llm import DRAFT_CHECK_SYSTEM_PROMPT, DraftCheck

    calls = []
    limiter = SimpleNamespace(check_and_record=lambda: calls.append("limit"))
    client = LLMClient(
        settings=SimpleNamespace(llm_provider="openai"), rate_limiter=limiter
    )
    seen = {}

    def fake_call(system, user_content, model):
        seen.update(system=system, user=user_content)
        return DraftCheck(injected=False)

    monkeypatch.setattr(client, "_call_openai", fake_call)
    client.check_draft(sender="a", subject="s", body="hello", draft="hi back")

    assert calls == ["limit"]
    assert seen["system"] == DRAFT_CHECK_SYSTEM_PROMPT
    assert "(untrusted data)" in seen["user"] and "hi back" in seen["user"]
