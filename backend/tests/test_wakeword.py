"""Wake word: sleep until "Hey Jarvis", then talk, then sleep again."""

import pytest

np = pytest.importorskip("numpy")

from jarvis.config import PROJECT_ROOT  # noqa: E402
from jarvis.core.agent_manager import AgentManager  # noqa: E402
from jarvis.core.cost_guard import CostGuard  # noqa: E402
from jarvis.core.orchestrator import Orchestrator  # noqa: E402
from jarvis.voice.audio import AudioClip  # noqa: E402
from jarvis.voice.devices import ClipSource, MemorySink  # noqa: E402
from jarvis.voice.session import WAKE_REPLY, Phase, VoiceSession  # noqa: E402
from jarvis.voice.vad import VadConfig  # noqa: E402
from jarvis.voice.wakeword import OpenWakeWord, is_downloaded  # noqa: E402
from tests.fakes import FakeAgent, FakeClock  # noqa: E402
from tests.test_voice import ScriptedSTT, ToneTTS, silence, tone  # noqa: E402

RATE = 16_000


class LoudnessWakeWord:
    """Fake detector: 'hears' the wake word in loud high-pitched audio (880 Hz tone)."""

    name = "fake"

    def __init__(self):
        self.resets = 0

    def feed(self, frame):
        spectrum = np.abs(np.fft.rfft(frame))
        peak_hz = np.argmax(spectrum) * RATE / len(frame) if len(frame) else 0
        return 0.99 if peak_hz > 700 and np.abs(frame).max() > 0.1 else 0.0

    def reset(self):
        self.resets += 1


def wake_tone():
    return tone(0.5, freq=880)


def make(texts, clips, agent_script):
    agent = FakeAgent("a", agent_script)
    manager = AgentManager([agent], CostGuard(), clock=FakeClock())
    events, tts, sink = [], ToneTTS(), MemorySink()
    session = VoiceSession(
        Orchestrator(manager),
        ScriptedSTT(texts),
        tts,
        ClipSource(clips),
        sink,
        vad_config=VadConfig(no_speech_timeout=4.0),  # > the silences between test clips
        on_event=lambda phase, data: events.append(phase),
        wake_word=LoudnessWakeWord(),
    )
    return session, agent, tts, events


async def test_ordinary_speech_does_not_wake_jarvis():
    session, agent, tts, events = make(["nunca"], [tone(1.0, freq=220)], ["x"])
    # The source then yields only silence: wait_for_wake would block forever; stop after a bit.
    import asyncio

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(session.wait_for_wake(), 0.5)
    assert agent.calls == [] and tts.sentences == []
    assert events == [Phase.SLEEPING]


async def test_wake_word_then_conversation_then_sleep():
    clips = [wake_tone(), tone(0.8), tone(0.8)]
    session, agent, tts, events = make(
        ["JARVIS, explique Docker.", "Tchau JARVIS"], clips, ["Docker é uma plataforma."]
    )
    import asyncio

    with pytest.raises(TimeoutError):  # after "tchau" it goes back to sleep and waits forever
        await asyncio.wait_for(session.run(), 3)
    assert tts.sentences[0] == WAKE_REPLY
    assert "Docker é uma plataforma." in tts.sentences
    assert len(agent.calls) == 1
    assert events.count(Phase.SLEEPING) == 2  # asleep, awake, …, asleep again
    assert Phase.AWAKE in events


async def test_silence_after_waking_goes_back_to_sleep():
    session, agent, tts, events = make(["nunca"], [wake_tone(), silence(0.1)], ["x"])
    import asyncio

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(session.run(), 3)
    assert agent.calls == []
    assert events.count(Phase.SLEEPING) >= 2


# -- the real model, with synthesized speech (macOS say) --------------------------------

MODELS = PROJECT_ROOT / "data" / "models" / "wakeword"


@pytest.mark.skipif(not is_downloaded(MODELS, "hey_jarvis_v0.1"), reason="models not downloaded")
async def test_real_model_wakes_on_hey_jarvis_only():
    import shutil

    if shutil.which("say") is None:
        pytest.skip("macOS say not available")
    from jarvis.voice.tts import MacSayTTS

    detector = OpenWakeWord(MODELS)

    async def best(voice, text):
        clip = (await MacSayTTS(voice).synthesize(text)).resampled(RATE)
        padded = np.concatenate(
            [np.zeros(RATE, np.float32), clip.samples, np.zeros(RATE, np.float32)]
        )
        detector.reset()
        return max(detector.feed(padded[i : i + 480]) for i in range(0, len(padded) - 480, 480))

    assert await best("Samantha", "Hey Jarvis") > 0.9
    assert await best("Luciana", "Explique o que é Docker") < 0.1
    assert await best("Luciana", "Jarvis, que horas são?") < 0.5
    assert isinstance(AudioClip, type)
