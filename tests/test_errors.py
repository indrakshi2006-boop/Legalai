"""Provider failure classification. Errors are built the way the OpenAI SDK builds them."""

import httpx
import openai
import pytest

import analyze
import app
import errors
import report
import risk


def _sdk_error(status, code=None, error_type=None, headers=None, message="sk-secret-value leaked"):
    """Use the SDK's own factory, so body/code have the real shape (body is the inner error)."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request, headers=headers or {})
    body = {"error": {"message": message, "type": error_type, "code": code}}
    client = openai.OpenAI(api_key="unit-test-key-not-real", max_retries=0)
    return client._make_status_error(message, body=body, response=response)


@pytest.mark.parametrize(
    "code,error_type",
    [
        ("credit_balance_exhausted", "insufficient_quota"),
        ("credit_balance_exhausted", None),
        ("insufficient_quota", "insufficient_quota"),
        (None, "insufficient_quota"),
        ("organization_spend_limit_exceeded", None),
        ("project_spend_limit_exceeded", None),
        ("organization_usage_limit_exceeded", None),
    ],
)
def test_no_credit_is_not_a_rate_limit(code, error_type):
    exc = _sdk_error(429, code=code, error_type=error_type)
    assert isinstance(exc, openai.RateLimitError)
    failure = errors.classify(exc)
    assert failure.kind == errors.NO_CREDIT
    assert failure.retryable is False
    assert failure.fatal is True
    for text in (app.openai_failure_message(exc), analyze._failure_warning(exc)):
        assert "Retrying will not help." in text
        assert "rate limit" not in text.lower()
        assert "sk-secret" not in text


def test_old_test_shape_with_nested_error_still_classifies():
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    exc = openai.RateLimitError(
        "x", response=httpx.Response(429, request=request), body={"error": {"code": "credit_balance_exhausted"}}
    )
    assert errors.classify(exc).kind == errors.NO_CREDIT


def test_genuine_rate_limit_is_retryable_and_says_wait():
    exc = _sdk_error(429, code="rate_limit_exceeded", error_type="requests", headers={"retry-after": "7"})
    failure = errors.classify(exc)
    assert failure.kind == errors.RATE_LIMIT
    assert failure.retryable is True
    assert failure.fatal is False
    assert failure.retry_after == 7.0
    assert "Wait a minute" in app.openai_failure_message(exc)


def test_slow_down_is_a_rate_limit():
    assert errors.classify(_sdk_error(429, code="slow_down", error_type="rate_limit_error")).kind == errors.RATE_LIMIT


def test_retry_after_ms_header_wins():
    exc = _sdk_error(429, code="rate_limit_exceeded", headers={"retry-after-ms": "1500", "retry-after": "9"})
    assert errors.classify(exc).retry_after == 1.5


@pytest.mark.parametrize("status", [500, 502, 503])
def test_server_errors_are_retryable(status):
    failure = errors.classify(_sdk_error(status))
    assert failure.kind == errors.SERVER
    assert failure.retryable is True
    assert str(status) in errors.user_message(failure)


def test_auth_and_permission_are_fatal_and_not_retryable():
    auth = errors.classify(_sdk_error(401, code="invalid_api_key"))
    assert auth.kind == errors.AUTH and auth.fatal and not auth.retryable
    denied = errors.classify(_sdk_error(403, code="model_not_found"))
    assert denied.kind == errors.PERMISSION and denied.fatal and not denied.retryable


def test_timeout_and_connection_errors():
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    assert errors.classify(openai.APITimeoutError(request=request)).kind == errors.TIMEOUT
    assert errors.classify(openai.APIConnectionError(request=request)).kind == errors.NETWORK


def test_wrapped_errors_are_found_through_the_cause_chain():
    inner = _sdk_error(429, code="credit_balance_exhausted")
    try:
        try:
            raise inner
        except Exception as exc:
            raise RuntimeError("wrapper") from exc
    except RuntimeError as outer:
        assert errors.classify(outer).kind == errors.NO_CREDIT


def test_unrelated_errors_are_not_classified():
    assert errors.classify(ValueError("x")) is None
    assert app.openai_failure_message(ValueError("x")) is None


def test_codes_never_carry_keys_or_messages():
    assert errors.safe_code("sk-abc") == ""
    assert errors.safe_code("AIzaSyFake") == ""
    assert errors.safe_code("two words") == ""
    assert errors.safe_code("x" * 81) == ""
    assert errors.safe_code("credit_balance_exhausted") == "credit_balance_exhausted"


def test_comparison_warnings_always_lead_to_review_required():
    for exc in (
        _sdk_error(429, code="credit_balance_exhausted"),
        _sdk_error(429, code="rate_limit_exceeded"),
        _sdk_error(500),
        _sdk_error(401),
    ):
        warning = analyze._failure_warning(exc)
        assert "no comparison was made" in warning.lower()
        result = {
            "clause_id": "clause-001",
            "contract_text": "The customer pays within thirty days.",
            "evidence_id": "payment-01",
            "reference_clause": "ref",
            "severity": "low",
            "deviations": [],
            "warnings": [warning],
            "retrieved": [{"evidence_id": "payment-01"}],
        }
        assert risk.score_analysis(result)["label"] == risk.REVIEW_REQUIRED


def _hit():
    return {
        "evidence_id": "payment-01",
        "text": "The customer pays undisputed invoices within thirty days.",
        "category": "Payment",
        "title": "Invoice timing",
        "distance": 0.2,
        "similarity": 0.8,
    }


def test_no_credit_stops_the_remaining_clauses_and_keeps_rows():
    calls = {"retrieve": 0, "compare": 0}

    def retriever(text, category=None, k=3):
        calls["retrieve"] += 1
        return [_hit()]

    def comparer(contract_text, hits):
        calls["compare"] += 1
        raise _sdk_error(429, code="credit_balance_exhausted")

    text = (
        "1. The customer pays each invoice within ten days of receipt.\n"
        "2. The customer pays a late fee on overdue invoices each month.\n"
        "3. The customer pays the supplier's travel costs at cost.\n"
    )
    rows = report.run_contract_review(text, retriever=retriever, comparer=comparer)
    assert len(rows) == 3
    assert calls["compare"] == 1
    assert all(row["label"] == "Review required" for row in rows)
    assert "Retrying will not help." in " ".join(rows[0]["warnings"])
    assert "review stopped after clause-001" in " ".join(rows[1]["warnings"])


def test_a_retrieval_failure_on_the_category_query_keeps_earlier_rows():
    state = {"n": 0}

    def retriever(text, category=None, k=3):
        state["n"] += 1
        if category is not None:
            raise _sdk_error(503)
        return []

    text = "1. The customer pays each invoice within ten days of receipt.\n"
    clause = {"id": "clause-001", "text": text}
    result = analyze.analyze_clause(clause, 0, retriever, lambda *a: None)
    assert result["evidence_id"] == ""
    assert result["error_kind"] == errors.SERVER
    assert "no comparison was made" in " ".join(result["warnings"]).lower()
