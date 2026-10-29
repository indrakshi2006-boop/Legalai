"""Classify provider API failures into the action a user can take.

The classification decides two things:
- whether a retry can help (rate limit, temporary server error, network);
- whether the whole review must stop (bad key, no credit, call budget used).

Only short provider codes are ever shown. Exception messages are never
copied into user text, because a provider message can echo request data.

OpenAI codes follow the error-codes guide
(https://developers.openai.com/api/docs/guides/error-codes, checked
2026-10-29): credit_balance_exhausted and the spend/usage limit codes are
HTTP 429 but are not rate limits, and the broader error.type can still be
insufficient_quota.
"""

from dataclasses import dataclass
import email.utils
import json as JS
import time

AUTH = "auth"
PERMISSION = "permission"
NO_CREDIT = "no_credit"
RATE_LIMIT = "rate_limit"
SERVER = "server"
TIMEOUT = "timeout"
NETWORK = "network"
BAD_REQUEST = "bad_request"
BUDGET = "budget"
NOT_READY = "not_ready"
DAILY_QUOTA = "daily_quota"
UNKNOWN = "unknown"

# A retry may help only for these.
RETRYABLE = frozenset({RATE_LIMIT, SERVER, TIMEOUT, NETWORK})
# These stop the remaining clauses of a review: every later call would fail the same way.
FATAL = frozenset({AUTH, PERMISSION, NO_CREDIT, BUDGET, NOT_READY, DAILY_QUOTA})

NO_CREDIT_CODES = frozenset(
    {
        "credit_balance_exhausted",
        "insufficient_quota",
        "billing_not_active",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
    }
)

MAX_RETRY_AFTER_SECONDS = 60.0


@dataclass(frozen=True)
class Failure:
    kind: str
    provider: str = "OpenAI"
    code: str = ""
    status: int | None = None
    retry_after: float | None = None

    @property
    def retryable(self):
        return self.kind in RETRYABLE

    @property
    def fatal(self):
        return self.kind in FATAL


class BudgetExceeded(RuntimeError):
    """Raised before a call that would go over the approved call limit."""

    def __init__(self, kind, limit):
        super().__init__(f"{kind} call limit {limit} reached")
        self.call_kind = kind
        self.limit = limit


def classify(exc, provider="OpenAI"):
    """Return a Failure for a provider error, or None for an unrelated exception."""
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        failure = _classify_one(current, provider)
        if failure is not None:
            return failure
        current = current.__cause__ or current.__context__
    return None


def _classify_one(exc, provider):
    if isinstance(exc, BudgetExceeded):
        return Failure(BUDGET, provider)
    if type(exc).__name__ == "ProviderNotReady":
        return Failure(NOT_READY, provider)
    failure = _classify_openai(exc, provider)
    if failure is not None:
        return failure
    failure = _classify_gemini(exc, provider)
    if failure is not None:
        return failure
    if isinstance(exc, TimeoutError):
        return Failure(TIMEOUT, provider)
    if isinstance(exc, ConnectionError):
        return Failure(NETWORK, provider)
    return None


def _classify_openai(exc, provider):
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a hard requirement
        return None
    if not isinstance(exc, openai.OpenAIError):
        return None
    if isinstance(exc, openai.APITimeoutError):
        return Failure(TIMEOUT, provider)
    if isinstance(exc, openai.APIConnectionError):
        return Failure(NETWORK, provider)
    if not isinstance(exc, openai.APIStatusError):
        return None
    status = int(getattr(exc, "status_code", 0) or 0)
    code = safe_code(_openai_field(exc, "code"))
    error_type = safe_code(_openai_field(exc, "type"))
    retry_after = retry_after_seconds(getattr(exc, "response", None))
    if status == 401 or isinstance(exc, openai.AuthenticationError):
        return Failure(AUTH, provider, code, status)
    if status == 403 or isinstance(exc, openai.PermissionDeniedError):
        return Failure(PERMISSION, provider, code, status)
    if code in NO_CREDIT_CODES or error_type in NO_CREDIT_CODES:
        return Failure(NO_CREDIT, provider, code or error_type, status)
    if status == 429:
        return Failure(RATE_LIMIT, provider, code, status, retry_after)
    if status >= 500:
        return Failure(SERVER, provider, code, status, retry_after)
    if status in (408,):
        return Failure(TIMEOUT, provider, code, status, retry_after)
    return Failure(BAD_REQUEST, provider, code, status)


def _openai_field(exc, name):
    value = getattr(exc, name, None)
    if value:
        return value
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        if body.get(name):
            return body.get(name)
        inner = body.get("error")
        if isinstance(inner, dict):
            return inner.get(name)
    return None


def _classify_gemini(exc, provider):
    """google-genai raises errors.APIError with an HTTP code, a status and the error JSON.

    Gemini API errors page (https://ai.google.dev/gemini-api/docs/api-errors,
    last updated 2026-10-23): 400 failed_precondition (e.g. billing disabled),
    402 payment_required (Prepay credit depleted: do not retry), 403
    permission_denied, 429 rate_limit_exceeded / too_many_requests (wait and
    retry) and quota_exceeded (daily quota: wait for the reset). The
    troubleshooting page (last updated 2026-10-27) treats 408, 429 and 5xx as
    transient and 400/402/403 as do-not-retry.
    """
    try:
        from google.genai import errors as genai_errors
    except ImportError:
        return None
    if not isinstance(exc, genai_errors.APIError):
        return None
    if provider == "OpenAI":
        provider = "Gemini"
    status = int(getattr(exc, "code", 0) or 0)
    state = str(getattr(exc, "status", "") or "")
    try:
        detail = JS.dumps(getattr(exc, "details", None), default=str).lower()
    except (TypeError, ValueError):
        detail = ""
    marker = state.lower() + " " + detail
    code = safe_code(state)
    retry_after = retry_after_seconds(getattr(exc, "response", None))
    if retry_after is None:
        retry_after = _google_retry_delay(getattr(exc, "details", None))
    if status == 401 or "api_key_invalid" in marker or "unauthenticated" in marker:
        return Failure(AUTH, provider, code, status)
    if status == 402 or "payment_required" in marker:
        return Failure(NO_CREDIT, provider, code or "payment_required", status)
    if status == 400 and "failed_precondition" in marker:
        return Failure(NO_CREDIT, provider, code or "failed_precondition", status)
    if status == 403 or "permission_denied" in marker:
        return Failure(PERMISSION, provider, code, status)
    if status == 429:
        if "quota_exceeded" in marker or "perday" in marker or "per_day" in marker:
            return Failure(DAILY_QUOTA, provider, code or "quota_exceeded", status)
        return Failure(RATE_LIMIT, provider, code, status, retry_after)
    if status == 408 or status == 504 or "deadline_exceeded" in marker:
        return Failure(TIMEOUT, provider, code, status, retry_after)
    if status >= 500:
        return Failure(SERVER, provider, code, status, retry_after)
    return Failure(BAD_REQUEST, provider, code, status)


def _google_retry_delay(details):
    """retryDelay such as "17s" from a google.rpc.RetryInfo detail, if present."""
    stack = [details]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            value = item.get("retryDelay")
            if isinstance(value, str) and value.endswith("s"):
                try:
                    return max(float(value[:-1]), 0.0)
                except ValueError:
                    pass
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def safe_code(value):
    """A short identifier only: no spaces, no key-like text, at most 80 characters."""
    text = str(value or "").strip()
    if not text or len(text) > 80 or any(char.isspace() for char in text):
        return ""
    lowered = text.lower()
    if "sk-" in lowered or lowered.startswith("aiza"):
        return ""
    return text


def retry_after_seconds(response):
    """Seconds from Retry-After / retry-after-ms headers, or None."""
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        millis = headers.get("retry-after-ms")
        if millis is not None:
            return max(float(millis) / 1000.0, 0.0)
        value = headers.get("retry-after")
    except Exception:
        return None
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        pass
    try:
        when = email.utils.parsedate_to_datetime(str(value))
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    return max(when.timestamp() - time.time(), 0.0)


def _with_code(failure):
    return f" ({failure.code})" if failure.code else ""


def user_message(failure):
    """A short message with the next action. Never includes the exception text."""
    name = failure.provider
    if failure.kind == AUTH:
        return f"{name} rejected the API key. Check the key in .env, then restart the app."
    if failure.kind == PERMISSION:
        return f"{name} refused this request{_with_code(failure)}. The key may not have access to the configured model."
    if failure.kind == NO_CREDIT:
        return (
            f"{name} reports no available credit, inactive billing or a reached spend limit{_with_code(failure)}. "
            f"Add credit, enable billing or raise the limit in the {name} billing settings. Retrying will not help."
        )
    if failure.kind == DAILY_QUOTA:
        return (
            f"{name} daily quota reached{_with_code(failure)}. Wait for the daily reset or request a higher quota. "
            "Retrying now will not help."
        )
    if failure.kind == NOT_READY:
        return f"{name} is not ready for this call. Follow the setup step shown in the app."
    if failure.kind == RATE_LIMIT:
        return f"{name} rate limit reached{_with_code(failure)}. Wait a minute, then try again."
    if failure.kind == SERVER:
        status = f"HTTP {failure.status}" if failure.status else "a server error"
        return f"{name} returned {status}, a temporary server error. Try again later."
    if failure.kind == TIMEOUT:
        return f"The request to {name} timed out. Check the network or raise the timeout, then try again."
    if failure.kind == NETWORK:
        return f"Could not reach {name}. Check the network connection, then try again."
    if failure.kind == BUDGET:
        return "The approved call limit for this run was reached, so no further calls were made."
    if failure.kind == BAD_REQUEST:
        status = f"HTTP {failure.status}" if failure.status else "an error"
        return f"{name} rejected the request with {status}{_with_code(failure)}."
    return f"The {name} request failed."


def comparison_warning(failure):
    """Clause warning. It always contains a phrase that risk.py treats as a failure."""
    return f"{user_message(failure)} No comparison was made for this clause."
