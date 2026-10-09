"""Server tests: REST, WebSocket text chat, the voice loop over WebSocket, security."""

import io
import json
import wave

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from jarvis.config import Settings  # noqa: E402
from jarvis.core.agent_manager import AgentManager  # noqa: E402
from jarvis.core.cost_guard import CostGuard  # noqa: E402
from jarvis.core.errors import ProviderUnavailableError  # noqa: E402
from jarvis.core.orchestrator import Orchestrator  # noqa: E402
from jarvis.memory.assistant import Assistant  # noqa: E402
from jarvis.memory.store import MemoryStore  # noqa: E402
from jarvis.server.app import create_app  # noqa: E402
from jarvis.server.connection import MAX_TEXT  # noqa: E402
from jarvis.server.runtime import JarvisRuntime, VoiceEngines  # noqa: E402
from jarvis.server.ws_audio import QueueAudioSource  # noqa: E402
from jarvis.voice.audio import AudioClip  # noqa: E402
from jarvis.voice.stt import STTProvider, Transcript  # noqa: E402
from jarvis.voice.tts import TTSProvider  # noqa: E402
from tests.fakes import FakeAgent, FakeClock  # noqa: E402

ORIGIN = "http://127.0.0.1:5300"
WS_URL = "ws://127.0.0.1:8300/ws"
RATE = 16_000


class ScriptedSTT(STTProvider):
    name = "scripted-stt"

    def __init__(self, texts):
        self.texts = list(texts)

    async def transcribe(self, clip):
        return Transcript(self.texts.pop(0), "pt", clip.duration, 5.0)


class ToneTTS(TTSProvider):
    name = "tone-tts"

    def __init__(self):
        self.sentences = []

    async def synthesize(self, text):
        self.sentences.append(text)
        t = np.arange(int(0.2 * 22_050)) / 22_050
        return AudioClip((0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), 22_050)


def make_client(agents, *, stt_texts=(), tmp_path=None, memory=None, tools=None):
    settings = Settings(data_dir=tmp_path) if tmp_path else Settings()
    settings.voice.barge_in = False  # tests opt in with a fake detector (see test_barge_in)
    manager = AgentManager(agents, CostGuard(), clock=FakeClock())
    tts = ToneTTS()
    voice = VoiceEngines(ScriptedSTT(stt_texts), {"say": tts, "piper": tts})
    assistant = Assistant(Orchestrator(manager), memory, summarize=False, tools=tools)
    runtime = JarvisRuntime(settings, manager, assistant, voice=voice)
    app = create_app(runtime, web_dist=None)
    return TestClient(app, base_url="http://127.0.0.1:8300"), tts


def receive_until(ws, predicate, limit=60):
    """Collect server messages until ``predicate(message)``; JSON is decoded."""
    seen = []
    for _ in range(limit):
        raw = ws.receive()
        message = raw["bytes"] if raw.get("bytes") is not None else json.loads(raw["text"])
        seen.append(message)
        if predicate(message):
            return seen
    raise AssertionError(f"condition not met; got {seen!r}")


def is_type(kind, **fields):
    return lambda m: (
        isinstance(m, dict)
        and m.get("type") == kind
        and all(m.get(k) == v for k, v in fields.items())
    )


def pcm_frames(clip_seconds_tone=0.8):
    frame = RATE * 30 // 1000
    silence = np.zeros(int(0.6 * RATE), dtype=np.float32)
    t = np.arange(int(clip_seconds_tone * RATE)) / RATE
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    samples = np.concatenate([silence, tone, np.zeros(int(1.5 * RATE), dtype=np.float32)])
    ints = (samples * 32767).astype(np.int16)
    return [ints[i : i + frame].tobytes() for i in range(0, len(ints) - frame + 1, frame)]


# -- REST ---------------------------------------------------------------------------


def test_health_and_status():
    client, _ = make_client([FakeAgent("a")])
    with client:
        assert client.get("/api/health").json() == {"ok": True}
        status = client.get("/api/status").json()
    assert status["cost_mode"] == "FREE_ONLY"
    assert [a["id"] for a in status["agents"]] == ["a"]
    assert status["voice"]["ready"] is True


def test_foreign_origins_are_rejected():
    client, _ = make_client([FakeAgent("a")])
    evil = {"origin": "https://evil.example"}
    with client:
        assert client.get("/api/status", headers=evil).status_code == 403
        assert client.get("/api/status", headers={"origin": ORIGIN}).status_code == 200
        with (
            pytest.raises(WebSocketDisconnect) as info,
            client.websocket_connect(WS_URL, headers=evil) as ws,
        ):
            ws.receive()
        assert info.value.code == 1008


def test_no_secrets_in_status(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_super_secret_value")
    client, _ = make_client([FakeAgent("a")])
    with client:
        assert "gsk_super_secret_value" not in client.get("/api/status").text


# -- WebSocket: text ----------------------------------------------------------------


def test_text_chat_keeps_context_and_reports_the_agent():
    agent = FakeAgent("a", ["Docker é uma plataforma.", "Use o Docker Desktop."])
    client, _ = make_client([agent])
    with client, client.websocket_connect(WS_URL, headers={"origin": ORIGIN}) as ws:
        hello = receive_until(ws, is_type("state", state="idle"))
        assert hello[0]["type"] == "hello"
        assert any(m.get("type") == "agents" for m in hello)

        ws.send_json({"type": "text", "text": "explique Docker"})
        first = receive_until(ws, is_type("answer"))[-1]
        assert first["text"] == "Docker é uma plataforma."
        assert first["agent_id"] == "a" and first["mode"] == "text" and first["cost"] == 0

        ws.send_json({"type": "text", "text": "e no Mac?"})
        receive_until(ws, is_type("answer"))
    sent = [(m.role, m.content) for m in agent.calls[1].messages[1:]]
    assert sent == [
        ("user", "explique Docker"),
        ("assistant", "Docker é uma plataforma."),
        ("user", "e no Mac?"),
    ]


def test_text_answer_can_be_spoken_as_wav():
    client, tts = make_client([FakeAgent("a", ["Olá. Tudo bem?"])])
    with client, client.websocket_connect(WS_URL, headers={"origin": ORIGIN}) as ws:
        receive_until(ws, is_type("state", state="idle"))
        ws.send_json({"type": "text", "text": "oi", "speak": True})
        seen = receive_until(ws, lambda m: isinstance(m, bytes))
        assert any(is_type("state", state="speaking")(m) for m in seen)
        with wave.open(io.BytesIO(seen[-1])) as wav:
            assert wav.getframerate() == 22_050
    assert tts.sentences == ["Olá.", "Tudo bem?"]


def test_failures_and_bad_messages_become_error_events():
    client, _ = make_client([FakeAgent("a", [ProviderUnavailableError("down")])])
    with client, client.websocket_connect(WS_URL, headers={"origin": ORIGIN}) as ws:
        receive_until(ws, is_type("state", state="idle"))
        ws.send_json({"type": "text", "text": "oi"})
        error = receive_until(ws, is_type("error"))[-1]
        assert "nenhum agente" in error["message"] and error["attempts"]
        ws.send_text("{not json")
        assert "inválida" in receive_until(ws, is_type("error"))[-1]["message"]
        ws.send_json({"type": "text", "text": "x" * (MAX_TEXT + 1)})
        receive_until(ws, is_type("error"))
        ws.send_json({"type": "hack"})
        assert "desconhecido" in receive_until(ws, is_type("error"))[-1]["message"]


# -- WebSocket: voice ---------------------------------------------------------------


def test_voice_loop_over_websocket():
    agent = FakeAgent("a", ["Docker é uma plataforma."])
    client, tts = make_client([agent], stt_texts=["JARVIS, explique Docker."])
    with client, client.websocket_connect(WS_URL, headers={"origin": ORIGIN}) as ws:
        receive_until(ws, is_type("state", state="idle"))
        ws.send_json({"type": "voice_start"})
        receive_until(ws, is_type("state", state="listening"))
        for frame in pcm_frames():
            ws.send_bytes(frame)
        seen = receive_until(ws, lambda m: isinstance(m, bytes))
        kinds = [m["type"] if isinstance(m, dict) else "audio" for m in seen]
        assert kinds.index("transcript") < kinds.index("answer") < kinds.index("audio")
        transcript = next(m for m in seen if is_type("transcript")(m))
        assert transcript["text"] == "JARVIS, explique Docker."
        answer = next(m for m in seen if is_type("answer")(m))
        assert answer["mode"] == "voice"
        ws.send_json({"type": "played", "count": 1})  # the browser finished the sentence
        # Back to listening for the next turn (continuous conversation).
        receive_until(ws, is_type("state", state="listening"))
        ws.send_json({"type": "voice_stop"})
        receive_until(ws, is_type("state", state="idle"))
    # The voice turn asked the agents for a short, spoken-style answer.
    assert "três frases curtas" in agent.calls[0].messages[0].content
    assert tts.sentences == ["Docker é uma plataforma."]


def test_audio_is_ignored_unless_the_voice_loop_runs():
    agent = FakeAgent("a")
    client, _ = make_client([agent], stt_texts=["nunca"])
    with client, client.websocket_connect(WS_URL, headers={"origin": ORIGIN}) as ws:
        receive_until(ws, is_type("state", state="idle"))
        for frame in pcm_frames():
            ws.send_bytes(frame)
        ws.send_json({"type": "status"})
        receive_until(ws, is_type("agents"))
    assert agent.calls == []


def test_queue_source_rejects_bad_frames():
    source = QueueAudioSource()
    assert source.push(b"\x00\x01" * 480)
    assert not source.push(b"")
    assert not source.push(b"\x00")  # odd length is not PCM16
    assert not source.push(b"\x00" * (64 * 1024 + 2))


# -- memory over WebSocket ------------------------------------------------------------


class KeepOpenStore(MemoryStore):
    """The app closes its store on shutdown; tests reuse one across clients."""

    def close(self):
        pass


def connect(client):
    ws = client.websocket_connect(WS_URL, headers={"origin": ORIGIN})
    return ws


def test_reconnecting_resumes_the_last_conversation():
    store = KeepOpenStore()
    client, _ = make_client([FakeAgent("a", ["Docker é uma plataforma."])], memory=store)
    with client, connect(client) as ws:
        receive_until(ws, is_type("state", state="idle"))
        ws.send_json({"type": "text", "text": "explique Docker"})
        receive_until(ws, is_type("answer"))

    client, _ = make_client([FakeAgent("a")], memory=store)
    with client, connect(client) as ws:
        history = next(
            m for m in receive_until(ws, is_type("state", state="idle")) if is_type("history")(m)
        )
    assert [(m["role"], m["text"]) for m in history["messages"]] == [
        ("user", "explique Docker"),
        ("assistant", "Docker é uma plataforma."),
    ]
    assert history["title"] == "explique Docker"


def test_new_and_reopened_conversations():
    store = KeepOpenStore()
    client, _ = make_client([FakeAgent("a", ["resposta antiga", "resposta nova"])], memory=store)
    with client, connect(client) as ws:
        receive_until(ws, is_type("state", state="idle"))
        ws.send_json({"type": "text", "text": "assunto antigo"})
        receive_until(ws, is_type("answer"))
        old_id = store.latest_conversation().id

        ws.send_json({"type": "new_conversation"})
        seen = receive_until(ws, is_type("cleared"))
        assert next(m for m in seen if is_type("history")(m))["messages"] == []
        listed = next(m for m in seen if is_type("conversations")(m))["conversations"]
        assert [c["title"] for c in listed] == ["assunto antigo"]

        ws.send_json({"type": "open_conversation", "id": old_id})
        reopened = receive_until(ws, is_type("history"))[-1]
        assert reopened["conversation_id"] == old_id
        assert reopened["messages"][0]["text"] == "assunto antigo"

        ws.send_json({"type": "open_conversation", "id": "nao-existe"})
        receive_until(ws, is_type("error"))
        ws.send_json({"type": "delete_conversation", "id": old_id})
        after = receive_until(ws, is_type("conversations"))[-1]
        assert after["conversations"] == []


def test_facts_are_managed_from_the_interface():
    store = KeepOpenStore()
    agent = FakeAgent("a")
    client, _ = make_client([agent], memory=store)
    with client, connect(client) as ws:
        receive_until(ws, is_type("state", state="idle"))
        ws.send_json({"type": "text", "text": "JARVIS, lembre que eu uso macOS"})
        answer = receive_until(ws, is_type("answer"))[-1]
        assert answer["agent_id"] == "jarvis:memoria" and answer["cost"] == 0
        facts = receive_until(ws, is_type("facts"))[-1]["facts"]
        assert [f["text"] for f in facts] == ["eu uso macOS"]

        ws.send_json({"type": "clear_facts"})  # without confirm: nothing happens
        ws.send_json({"type": "list_facts"})
        assert receive_until(ws, is_type("facts"))[-1]["facts"]

        ws.send_json({"type": "forget_fact", "id": facts[0]["id"]})
        assert receive_until(ws, is_type("facts"))[-1]["facts"] == []
        ws.send_json({"type": "forget_fact", "id": "1; DROP TABLE facts"})
        receive_until(ws, is_type("error"))
    assert agent.calls == []


# -- tool confirmations over WebSocket --------------------------------------------------


def _toolkit():
    from jarvis.tools.base import Policy
    from jarvis.tools.builtin.basic import CalculatorTool
    from jarvis.tools.executor import ToolExecutor
    from jarvis.tools.toolkit import ToolKit

    calculator = CalculatorTool()
    return ToolKit(ToolExecutor([calculator], policies={"calculator": Policy.CONFIRM}))


CALL = '```tool\n{"name": "calculator", "args": {"expression": "6 * 7"}}\n```'


@pytest.mark.parametrize("approved", [True, False])
def test_tool_confirmation_in_the_browser(approved):
    agent = FakeAgent("a", [CALL, "Pronto."])
    client, _ = make_client([agent], tools=_toolkit())
    with client, connect(client) as ws:
        receive_until(ws, is_type("state", state="idle"))
        ws.send_json({"type": "text", "text": "quanto é 6 vezes 7?"})
        confirm = receive_until(ws, is_type("confirm"))[-1]
        assert confirm["tool"] == "calculator" and "6 * 7" in confirm["description"]
        ws.send_json({"type": "confirm_reply", "id": confirm["id"], "approved": approved})
        seen = receive_until(ws, is_type("answer"))
        answer = seen[-1]
        assert answer["tools"][0]["decision"] == ("confirmed" if approved else "denied_by_user")
        assert any(is_type("confirm_closed")(m) for m in seen)
        assert any(is_type("tool", phase="end")(m) for m in seen)
    tool_result = agent.calls[1].messages[-1].content
    assert ("42" in tool_result) is approved


def test_local_only_switch_keeps_messages_on_this_mac():
    from jarvis.core.types import CostClass, Privacy

    local = FakeAgent("local", ["resposta local"], priority=10, privacy=Privacy.LOCAL)
    local.info = local.info.model_copy(update={"is_local": True, "cost_class": CostClass.LOCAL})
    online = FakeAgent("online", ["resposta online"], priority=90)
    client, _ = make_client([local, online])
    with client, connect(client) as ws:
        hello = receive_until(ws, is_type("state", state="idle"))[0]
        assert hello["local_only"] is False
        ws.send_json({"type": "set_local_only", "value": True})
        assert receive_until(ws, is_type("local_only"))[-1]["value"] is True
        ws.send_json({"type": "text", "text": "segredo"})
        assert receive_until(ws, is_type("answer"))[-1]["agent_id"] == "local"
        ws.send_json({"type": "set_local_only", "value": False})
        receive_until(ws, is_type("local_only"))
        ws.send_json({"type": "text", "text": "pergunta comum"})
        assert receive_until(ws, is_type("answer"))[-1]["agent_id"] == "online"
    assert [m.content for m in online.calls[0].messages if m.role == "user"][-1] == "pergunta comum"
    assert all("segredo" not in m.content for m in online.calls[0].messages)


# -- hardening (Phase 9) ------------------------------------------------------------------


def test_dns_rebinding_hosts_are_rejected():
    client, _ = make_client([FakeAgent("a")])
    with client:
        # A rebinding page makes same-origin GETs (no Origin header) with its own Host.
        assert client.get("/api/status", headers={"host": "evil.example:8300"}).status_code == 403
        assert client.get("/api/status", headers={"host": "localhost:8300"}).status_code == 200
        with (
            pytest.raises(WebSocketDisconnect) as info,
            client.websocket_connect(WS_URL, headers={"host": "evil.example"}) as ws,
        ):
            ws.receive()
        assert info.value.code == 1008


def test_security_headers_block_framing_and_sniffing():
    client, _ = make_client([FakeAgent("a")])
    with client:
        for response in (
            client.get("/api/health"),
            client.get("/api/status", headers={"origin": "https://evil.example"}),  # also on 403
        ):
            assert response.headers["x-frame-options"] == "DENY"
            assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["referrer-policy"] == "no-referrer"


def test_wake_word_over_websocket():
    from tests.test_wakeword import LoudnessWakeWord

    client, tts = make_client([FakeAgent("a")], stt_texts=["oi"])
    with client, connect(client) as ws:
        rt = client.app.state.runtime

        async def fake_wake_word():
            return LoudnessWakeWord()

        rt.wake_word = fake_wake_word
        receive_until(ws, is_type("state", state="idle"))
        ws.send_json({"type": "voice_start", "wake": True})
        voice = receive_until(ws, is_type("voice"))[-1]
        assert voice["active"] is True and voice["wake"] is True
        receive_until(ws, is_type("state", state="sleeping"))
        t = np.arange(int(0.5 * RATE)) / RATE
        loud = (0.4 * np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
        frame = RATE * 30 // 1000
        for i in range(0, len(loud) - frame + 1, frame):
            ws.send_bytes(loud[i : i + frame].tobytes())
        wake = receive_until(ws, is_type("wake"))[-1]
        assert wake["score"] > 0.5
        receive_until(ws, lambda m: isinstance(m, bytes))  # "Sim?" is spoken
        ws.send_json({"type": "voice_stop"})
        receive_until(ws, is_type("state", state="idle"))
    assert tts.sentences[0] == "Sim?"
