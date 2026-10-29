"""Bounded retries and an optional call budget for provider calls.

The SDK clients are created with max_retries=0, so this module is the only
place a call is repeated. Only rate limits, temporary server errors and
network failures are retried: at most three times, waiting 1, 2 and 4
seconds, or longer when the provider sends Retry-After. A rejected key, no
credit, or a bad request is never retried.

Every attempt, including a retry, counts against the budget. When a budget
is set, a call that would go over it is not made.
"""

import os as OS
import time

import errors as Errors

DEFAULT_DELAYS = (1.0, 2.0, 4.0)
# Indirection so tests can record waits instead of sleeping.
SLEEP = time.sleep
MAX_RETRIES_CAP = 3


def configured_max_retries():
    """AEGIS_MAX_RETRIES, clamped to 0..3. Default 3."""
    raw = OS.getenv("AEGIS_MAX_RETRIES", "").strip()
    if not raw:
        return MAX_RETRIES_CAP
    try:
        value = int(raw)
    except ValueError:
        return MAX_RETRIES_CAP
    return max(0, min(value, MAX_RETRIES_CAP))


class CallBudget:
    """Counts calls by kind ("embed", "chat") and refuses one over the limit."""

    def __init__(self, embed=None, chat=None):
        self.limits = {"embed": embed, "chat": chat}
        self.used = {"embed": 0, "chat": 0}

    def take(self, kind):
        limit = self.limits.get(kind)
        if limit is not None and self.used.get(kind, 0) >= limit:
            raise Errors.BudgetExceeded(kind, limit)
        self.used[kind] = self.used.get(kind, 0) + 1

    def remaining(self, kind):
        limit = self.limits.get(kind)
        if limit is None:
            return None
        return max(limit - self.used.get(kind, 0), 0)


class CallPolicy:
    def __init__(self, max_retries=None, delays=DEFAULT_DELAYS, sleep=None, budget=None, provider="OpenAI"):
        if max_retries is None:
            max_retries = configured_max_retries()
        self.max_retries = max(0, min(int(max_retries), MAX_RETRIES_CAP))
        self.delays = tuple(delays) or DEFAULT_DELAYS
        self.sleep = sleep or (lambda seconds: SLEEP(seconds))
        self.budget = budget
        self.provider = provider
        self.attempts = {"embed": 0, "chat": 0}
        self.retries = []

    def call(self, kind, function, *args, **kwargs):
        attempt = 0
        while True:
            if self.budget is not None:
                self.budget.take(kind)
            self.attempts[kind] = self.attempts.get(kind, 0) + 1
            try:
                return function(*args, **kwargs)
            except Errors.BudgetExceeded:
                raise
            except Exception as exc:
                failure = Errors.classify(exc, self.provider)
                if failure is None or not failure.retryable or attempt >= self.max_retries:
                    raise
                if self.budget is not None and self.budget.remaining(kind) == 0:
                    # No budget left for a retry: report the real failure, not the budget.
                    raise
                delay = self.delays[min(attempt, len(self.delays) - 1)]
                if failure.retry_after is not None:
                    if failure.retry_after > Errors.MAX_RETRY_AFTER_SECONDS:
                        raise
                    delay = max(delay, failure.retry_after)
                self.retries.append({"kind": kind, "reason": failure.kind, "code": failure.code, "wait": delay})
                self.sleep(delay)
                attempt += 1


def default_policy():
    return CallPolicy()
