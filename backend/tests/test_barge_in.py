"""Barge-in: "Hey Jarvis" while JARVIS talks cuts it off, and it listens again."""

import asyncio

import pytest

np = pytest.importorskip("numpy")

from jarvis.core.agent_manager import AgentManager  # noqa: E402
from jarvis.core.cost_guard import CostGuard  # noqa: E402
from jarvis.core.orchestrator import Orchestrator  # noqa: E402
from jarvis.server.ws_audio import DRAIN_SLACK, WebSocketSink  # noqa: E402
from jarvis.voice.devices import ClipSource, MemorySink  # noqa: E402
from jarvis.voice.session import Phase, TurnOutcome, VoiceSession  # noqa: E402
from jarvis.voice.text import split_sentences  # noqa: E402
from jarvis.voice.vad import VadConfig  # noqa: E402
from tests.fakes import FakeAgent, FakeClock  # noqa: E402
from tests.test_voice import ScriptedSTT, ToneTTS, tone  # noqa: E402
from tests.test_wakeword import LoudnessWakeWord, wake_tone  # noqa: E402

ANSWER = "Primeira frase. Segunda frase. Terceira frase. Quarta frase."


class SlowSink(MemorySink):
    """Takes a while per sentence, like a real speaker."""

    async def play(self, clip):
        await asyncio.sleep(0.05)
        await super().play(clip)


class FailingTTS(ToneTTS):
    async def synthesize(self, text):
        raise RuntimeError("voz quebrada")


def make(texts, clips, *, tts=None, barge_in=True):
    manager = AgentManager([FakeAgent("a", [ANSWER] * 3)], CostGuard(), clock=FakeClock())
    events, sink, detector = [], SlowSink(), LoudnessWakeWord()
    session = VoiceSession(
        Orchestrator(manager),
        ScriptedSTT(texts),
        tts or ToneTTS(),
        ClipSource(clips),
        sink,
        vad_config=VadConfig(no_speech_timeout=4.0),
        on_event=lambda phase, data: events.append(phase),
        barge_in=detector if barge_in else None,
    )
    return session, sink, events, detector


async def test_hey_jarvis_while_speaking_stops_the_voice_and_listens_again():
    clips = [tone(0.8), wake_tone(), tone(0.8)]
    session, sink, events, detector = make(["explique Docker", "e no Mac?"], clips)

    first = await session.turn()
    assert first.outcome is TurnOutcome.ANSWERED and first.interrupted
    assert sink.stopped == 1
    assert len(sink.played) < 4  # cut off before the end
    assert events.index(Phase.SPEAKING) < events.index(Phase.INTERRUPTED)
    assert detector.resets == 1

    second = await session.turn()  # JARVIS heard the next question
    assert second.transcript.text == "e no Mac?"
    assert not second.interrupted  # nobody said "Hey Jarvis": the whole answer plays
    assert len(sink.played) >= 4


async def test_ordinary_speech_does_not_interrupt():
    session, sink, events, _ = make(["explique Docker"], [tone(0.8), tone(0.8)])
    result = await session.turn()
    assert not result.interrupted
    assert len(sink.played) == 4 and sink.stopped == 0
    assert Phase.INTERRUPTED not in events


async def test_without_barge_in_the_microphone_stays_closed():
    session, sink, _, detector = make(["explique Docker"], [tone(0.8), wake_tone()], barge_in=False)
    result = await session.turn()
    assert not result.interrupted and len(sink.played) == 4
    assert detector.resets == 0


async def test_speech_errors_still_surface_with_barge_in():
    session, *_ = make([], [], tts=FailingTTS())
    with pytest.raises(RuntimeError, match="voz quebrada"):
        await session.speak("Olá.")


async def test_wake_word_mode_stays_awake_after_an_interruption():
    clips = [wake_tone(), tone(0.8), wake_tone(), tone(0.8)]
    detector = LoudnessWakeWord()
    session, sink, events, _ = make(["explique Docker", "e no Mac?"], clips)
    session.wake_word = session.barge_in = detector  # one detector, used for both
    assert await asyncio.wait_for(session.run(max_turns=2), 5) == 2
    assert events.count(Phase.SLEEPING) == 1  # never went back to sleep
    assert events.count(Phase.INTERRUPTED) == 1


# -- the browser's speaker ----------------------------------------------------------


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


async def test_web_sink_waits_until_the_browser_played_everything():
    sent = []

    async def send(wav, duration):
        sent.append(duration)

    sink = WebSocketSink(send)
    await sink.play(tone(0.2))
    await sink.play(tone(0.2))
    waiting = asyncio.create_task(sink.drained())
    await asyncio.sleep(0.01)
    sink.mark_played(1)
    await asyncio.sleep(0.01)
    assert not waiting.done()
    sink.mark_played(2)
    await asyncio.wait_for(waiting, 1)
    sink.mark_played(99)  # never more than was sent
    assert sink.played == sink.sent == 2


async def test_web_sink_gives_up_on_a_silent_browser_and_stops_on_interrupt():
    async def send(wav, duration):
        pass

    clock = Clock()
    sink = WebSocketSink(send, clock=clock)
    await sink.play(tone(0.2))
    clock.now += 0.2 + DRAIN_SLACK  # the browser should be done by now
    await asyncio.wait_for(sink.drained(), 1)

    await sink.play(tone(0.2))
    waiting = asyncio.create_task(sink.drained())
    await asyncio.sleep(0.01)
    sink.stop()
    await asyncio.wait_for(waiting, 1)
    await sink.play(tone(0.2))  # stopped: nothing more is sent
    assert sink.sent == 2


# -- over the WebSocket (the browser is the microphone and the speaker) ---------------


def test_barge_in_over_websocket():
    pytest.importorskip("fastapi")
    from tests.test_server import RATE, connect, is_type, make_client, pcm_frames, receive_until

    client, tts = make_client([FakeAgent("a", [ANSWER])], stt_texts=["explique Docker"])
    with client, connect(client) as ws:
        rt = client.app.state.runtime
        rt.settings.voice.barge_in = True

        async def fake_wake_word():
            return LoudnessWakeWord()

        rt.wake_word = fake_wake_word
        receive_until(ws, is_type("state", state="idle"))
        ws.send_json({"type": "voice_start"})  # no wake word: barge-in still works
        voice = receive_until(ws, is_type("voice"))[-1]
        assert voice["barge_in"] is True and voice["wake"] is False
        receive_until(ws, is_type("state", state="listening"))
        for frame in pcm_frames():
            ws.send_bytes(frame)
        receive_until(ws, lambda m: isinstance(m, bytes))  # JARVIS starts talking
        # The browser keeps the microphone open and the user says "Hey Jarvis".
        t = np.arange(int(0.5 * RATE)) / RATE
        loud = (0.4 * np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
        frame = RATE * 30 // 1000
        for i in range(0, len(loud) - frame + 1, frame):
            ws.send_bytes(loud[i : i + frame].tobytes())
        seen = receive_until(ws, is_type("interrupted"))
        assert any(is_type("stop_audio")(m) for m in seen)
        receive_until(ws, is_type("state", state="listening"))  # ready for the next question
        ws.send_json({"type": "played", "count": True})  # ignored: not a count
        ws.send_json({"type": "voice_stop"})
        receive_until(ws, is_type("state", state="idle"))
    # The sentences were already sent; stop_audio makes the browser drop what is left.
    assert tts.sentences == split_sentences(ANSWER)
