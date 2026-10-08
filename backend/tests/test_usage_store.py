from jarvis.core.usage_store import StoredState, UsageStore


def test_counts_calls_per_agent_and_day(tmp_path):
    store = UsageStore(tmp_path / "sub" / "jarvis.db")  # parent folder is created
    store.record_call("a", "2026-10-08", ok=True)
    store.record_call("a", "2026-10-08", ok=False)
    store.record_call("a", "2026-10-09", ok=True)
    state = store.load("a", "2026-10-08")
    assert (state.requests_today, state.successes_today, state.failures_today) == (2, 1, 1)
    assert store.load("a", "2026-10-09").requests_today == 1
    assert store.load("b", "2026-10-08") == StoredState()


def test_state_survives_reopening(tmp_path):
    path = tmp_path / "jarvis.db"
    store = UsageStore(path)
    store.record_call("a", "d", ok=True)
    store.save_state("a", now=1.0, cooldown_until=99.0, blocked_reason=None, last_error="429")
    store.close()

    again = UsageStore(path)
    state = again.load("a", "d")
    assert state.requests_today == 1
    assert state.cooldown_until == 99.0
    assert state.last_error == "429"


def test_block_and_unblock(tmp_path):
    store = UsageStore()
    store.save_state("a", now=1.0, cooldown_until=None, blocked_reason="402", last_error="402")
    assert store.blocked() == {"a": "402"}
    assert store.unblock("a") is True
    assert store.blocked() == {}
    assert store.unblock("a") is False  # nothing left to lift
    assert store.load("a", "d").blocked_reason is None
