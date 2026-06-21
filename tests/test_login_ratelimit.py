"""Login brute-force throttling (MemoryLoginRateLimiter + route wiring).

Unit tests drive the limiter with an injected clock (deterministic, no
sleeps); the integration tests prove the local login endpoint locks after
the configured number of failures and that a success resets the counter.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inqtrix.auth.api_key import build_local_provider
from inqtrix.auth.ratelimit import MemoryLoginRateLimiter
from inqtrix.server.routers.auth import build_auth_router
from inqtrix.settings import AuthSettings, Settings

KEY = "local:alice@example.com:1.2.3.4"


class _FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _limiter(clock: _FakeClock, *, max_attempts: int = 3) -> MemoryLoginRateLimiter:
    return MemoryLoginRateLimiter(
        max_attempts=max_attempts,
        window_seconds=300,
        lockout_seconds=60,
        clock=clock,
    )


def test_locks_after_max_attempts_and_lockout_expires():
    clock = _FakeClock()
    rl = _limiter(clock)
    assert rl.locked(KEY) is False
    rl.record_failure(KEY)
    rl.record_failure(KEY)
    assert rl.locked(KEY) is False  # 2 < 3
    rl.record_failure(KEY)
    assert rl.locked(KEY) is True  # 3 >= 3 -> locked
    clock.advance(59)
    assert rl.locked(KEY) is True
    clock.advance(2)  # 61 > lockout 60
    assert rl.locked(KEY) is False  # lockout elapsed


def test_failures_outside_the_window_do_not_accumulate():
    clock = _FakeClock()
    rl = _limiter(clock)
    rl.record_failure(KEY)
    rl.record_failure(KEY)
    clock.advance(301)  # the two failures fall out of the 300s window
    rl.record_failure(KEY)
    assert rl.locked(KEY) is False  # only 1 failure is in-window


def test_reset_clears_the_counter():
    rl = _limiter(_FakeClock(), max_attempts=2)
    rl.record_failure(KEY)
    rl.record_failure(KEY)
    assert rl.locked(KEY) is True
    rl.reset(KEY)
    assert rl.locked(KEY) is False


def test_stale_keys_are_pruned_on_write():
    # A key that accrues failures but never locks and never logs in must not
    # linger forever — the next write sweeps it once its window has elapsed
    # (bounds memory against identifier/IP rotation).
    clock = _FakeClock()
    rl = _limiter(clock)
    rl.record_failure("local:a@x:1.1.1.1")  # 1 failure, never locks
    clock.advance(301)  # its window elapses
    rl.record_failure("local:b@x:2.2.2.2")  # this write sweeps the stale key
    assert set(rl._entries.keys()) == {"local:b@x:2.2.2.2"}


def test_max_keys_cap_evicts_least_recently_touched():
    rl = MemoryLoginRateLimiter(
        max_attempts=3,
        window_seconds=300,
        lockout_seconds=60,
        max_keys=2,
        clock=_FakeClock(),
    )
    rl.record_failure("a")
    rl.record_failure("b")
    rl.record_failure("c")  # exceeds the cap -> oldest ("a") evicted
    assert set(rl._entries.keys()) == {"b", "c"}


def _client(max_attempts: int) -> TestClient:
    settings = Settings(
        auth=AuthSettings(
            mode="local",
            session_secret="s" * 32,
            pat_pepper="p" * 32,
            oidc_insecure_dev_cookies=True,
            login_rate_limit_max_attempts=max_attempts,
        )
    )
    provider = build_local_provider(settings)
    app = FastAPI()
    app.include_router(build_auth_router(provider))
    return TestClient(app, base_url="http://127.0.0.1:5100")


def test_login_locks_after_repeated_failures():
    client = _client(max_attempts=3)
    client.post(
        "/api/setup/owner",
        json={"email": "o@example.com", "password": "correct-horse-battery"},
    )
    for _ in range(3):
        assert (
            client.post(
                "/api/auth/login/local",
                json={"email": "o@example.com", "password": "wrong-password"},
            ).status_code
            == 401
        )
    # The next attempt is throttled — and even the CORRECT password is now
    # refused, because the lockout is checked before the credential check.
    assert (
        client.post(
            "/api/auth/login/local",
            json={"email": "o@example.com", "password": "wrong-password"},
        ).status_code
        == 429
    )
    assert (
        client.post(
            "/api/auth/login/local",
            json={"email": "o@example.com", "password": "correct-horse-battery"},
        ).status_code
        == 429
    )


def test_successful_login_resets_the_failure_count():
    client = _client(max_attempts=3)
    client.post(
        "/api/setup/owner",
        json={"email": "o@example.com", "password": "correct-horse-battery"},
    )
    for _ in range(2):  # 2 < 3, not yet locked
        assert (
            client.post(
                "/api/auth/login/local",
                json={"email": "o@example.com", "password": "wrong-password"},
            ).status_code
            == 401
        )
    # A correct login resets the counter...
    assert (
        client.post(
            "/api/auth/login/local",
            json={"email": "o@example.com", "password": "correct-horse-battery"},
        ).status_code
        == 200
    )
    # ...so two further failures still do not lock (would have without reset).
    for _ in range(2):
        assert (
            client.post(
                "/api/auth/login/local",
                json={"email": "o@example.com", "password": "wrong-password"},
            ).status_code
            == 401
        )
    assert (
        client.post(
            "/api/auth/login/local",
            json={"email": "o@example.com", "password": "correct-horse-battery"},
        ).status_code
        == 200
    )
