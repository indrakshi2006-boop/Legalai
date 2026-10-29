"""Bounded retries: temporary errors only, at most three, 1/2/4 s, Retry-After honoured."""

import httpx
import openai
import pytest

import analyze
import errors
import report
import retry


def _sdk_error(status, code=None, headers=None):
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(status, request=request, headers=headers or {})
    client = openai.OpenAI(api_key="unit-test-key-not-real", max_retries=0)
    return client._make_status_error("m", body={"error": {"code": code}}, response=response)


class Flaky:
    def __init__(self, failures, result="ok"):
        self.failures = list(failures)
        self.calls = 0
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.result


def test_rate_limit_is_retried_with_1_2_4_second_waits(isolated_environment):
    waits = isolated_environment
    function = Flaky([_sdk_error(429, "rate_limit_exceeded")] * 3)
    assert retry.CallPolicy().call("embed", function) == "ok"
    assert function.calls == 4
    assert waits == [1.0, 2.0, 4.0]


def test_gives_up_after_three_retries(isolated_environment):
    function = Flaky([_sdk_error(503)] * 5)
    with pytest.raises(openai.InternalServerError):
        retry.CallPolicy().call("chat", function)
    assert function.calls == 4
    assert isolated_environment == [1.0, 2.0, 4.0]


def test_retry_after_is_honoured_when_longer(isolated_environment):
    function = Flaky([_sdk_error(429, "rate_limit_exceeded", {"retry-after": "5"})])
    retry.CallPolicy().call("embed", function)
    assert isolated_environment == [5.0]


def test_retry_after_over_the_cap_is_not_waited_for(isolated_environment):
    function = Flaky([_sdk_error(429, "rate_limit_exceeded", {"retry-after": "3600"})])
    with pytest.raises(openai.RateLimitError):
        retry.CallPolicy().call("embed", function)
    assert function.calls == 1
    assert isolated_environment == []


@pytest.mark.parametrize(
    "exc",
    [
        _sdk_error(429, "credit_balance_exhausted"),
        _sdk_error(429, "insufficient_quota"),
        _sdk_error(401, "invalid_api_key"),
        _sdk_error(403),
        _sdk_error(400, "invalid_request_error"),
        ValueError("not a provider error"),
    ],
)
def test_never_retries_auth_no_credit_or_bad_requests(exc, isolated_environment):
    function = Flaky([exc])
    with pytest.raises(type(exc)):
        retry.CallPolicy().call("chat", function)
    assert function.calls == 1
    assert isolated_environment == []


def test_max_retries_is_configurable_and_capped(monkeypatch):
    monkeypatch.setenv("AEGIS_MAX_RETRIES", "0")
    assert retry.configured_max_retries() == 0
    monkeypatch.setenv("AEGIS_MAX_RETRIES", "10")
    assert retry.configured_max_retries() == 3
    monkeypatch.setenv("AEGIS_MAX_RETRIES", "x")
    assert retry.configured_max_retries() == 3


def test_retries_count_against_the_budget(isolated_environment):
    budget = retry.CallBudget(embed=2)
    function = Flaky([_sdk_error(429, "rate_limit_exceeded")] * 3)
    with pytest.raises(openai.RateLimitError):
        retry.CallPolicy(budget=budget).call("embed", function)
    assert function.calls == 2
    assert budget.used["embed"] == 2


def test_budget_exhaustion_reports_the_real_error_not_the_budget(isolated_environment):
    request = __import__("httpx").Request("POST", "https://api.openai.com/v1/embeddings")
    function = Flaky([openai.APIConnectionError(request=request)])
    with pytest.raises(openai.APIConnectionError):
        retry.CallPolicy(budget=retry.CallBudget(embed=1)).call("embed", function)
    assert function.calls == 1
    assert isolated_environment == [], "no wait for a retry that cannot be made"


def test_first_attempt_over_budget_raises_budget_exceeded():
    budget = retry.CallBudget(chat=0)
    with pytest.raises(errors.BudgetExceeded):
        retry.CallPolicy(budget=budget).call("chat", lambda: "never")


def test_budget_caps_a_whole_review():
    calls = {"embed": 0, "chat": 0}

    def retriever(text, category=None, k=3):
        calls["embed"] += 1
        return [
            {
                "evidence_id": "payment-01",
                "text": "The customer pays undisputed invoices within thirty days.",
                "category": "Payment",
                "title": "t",
                "distance": 0.2,
                "similarity": 0.8,
            }
        ]

    def comparer(contract_text, hits):
        calls["chat"] += 1
        return {"evidence_id": "payment-01", "deviations": [], "explanation": "x", "severity": "low"}

    text = "".join(f"{n}. The customer pays invoice number {n} within ten days.\n" for n in range(1, 6))
    policy = retry.CallPolicy(budget=retry.CallBudget(embed=4, chat=2))
    rows = report.run_contract_review(text, retriever=retriever, comparer=comparer, policy=policy)
    assert len(rows) == 5
    assert calls["chat"] <= 2
    assert calls["embed"] <= 4
    assert any("call limit" in " ".join(row["warnings"]) for row in rows)


def test_sdk_clients_have_their_own_retries_disabled(monkeypatch):
    captured = []

    class Fake:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def similarity_search_with_score(self, *args, **kwargs):
            return []

        def invoke(self, *args, **kwargs):
            return type("M", (), {"content": "{}"})()

        def __call__(self, value):
            return type("M", (), {"content": "{}"})()

    monkeypatch.setattr(analyze, "OpenAIEmbeddings", Fake)
    monkeypatch.setattr(analyze, "Chroma", Fake)
    monkeypatch.setattr(analyze, "ChatOpenAI", Fake)
    analyze.playbook_retriever("invoice")
    analyze.compare_with_model("text", [{"evidence_id": "a", "category": "Payment", "text": "t"}])
    import agent

    monkeypatch.setattr(agent, "ChatOpenAI", Fake)
    agent.chat_model()
    import embed

    assert embed.openai_embedding_function().client.max_retries == 0
    retries = [kwargs["max_retries"] for kwargs in captured if "max_retries" in kwargs]
    assert retries == [0, 0, 0]
