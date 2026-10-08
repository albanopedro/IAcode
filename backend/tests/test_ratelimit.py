import pytest

from jarvis.agents.ratelimit import parse_duration, parse_rate_limit


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("2m59.56s", 179.56),
        ("120ms", 0.12),
        ("14h12m", 51120.0),
        ("1.2s", 1.2),
        ("30", 30.0),
        ("", None),
        ("soon", None),
        ("5x", None),
    ],
)
def test_parse_duration(value, seconds):
    result = parse_duration(value)
    assert result == (pytest.approx(seconds) if seconds is not None else None)


def test_groq_style_headers():
    info = parse_rate_limit(
        {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "998",
            "x-ratelimit-reset-requests": "2m59.56s",
        }
    )
    assert (info.limit, info.remaining) == (1000, 998)
    assert info.reset_seconds == pytest.approx(179.56)


def test_openrouter_style_headers_with_epoch_ms_reset():
    now = 1_800_000_000.0
    info = parse_rate_limit(
        {
            "X-RateLimit-Limit": "50",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int((now + 3600) * 1000)),
        },
        now=now,
    )
    assert (info.limit, info.remaining) == (50, 0)
    assert info.reset_seconds == pytest.approx(3600)


def test_ietf_style_headers():
    headers = {"ratelimit-limit": "2", "ratelimit-remaining": "1", "ratelimit-reset": "19"}
    info = parse_rate_limit(headers)
    assert (info.limit, info.remaining, info.reset_seconds) == (2, 1, 19)


def test_no_quota_headers():
    assert parse_rate_limit({"content-type": "application/json"}) is None
