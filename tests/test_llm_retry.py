from inbox_triage.llm import _is_transient


class _StatusError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class APIConnectionError(Exception):
    pass


def test_retries_rate_limits_and_server_errors():
    for status in (408, 409, 429, 500, 503):
        assert _is_transient(_StatusError(status)) is True


def test_does_not_retry_client_errors():
    for status in (400, 401, 403, 404, 422):
        assert _is_transient(_StatusError(status)) is False


def test_retries_connection_errors_without_status():
    assert _is_transient(APIConnectionError()) is True


def test_does_not_retry_unrelated_exceptions():
    assert _is_transient(ValueError("bad output")) is False
