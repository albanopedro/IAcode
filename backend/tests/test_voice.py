"""Voice pipeline tests with synthetic audio: no microphone, speaker or model needed."""

import shutil

import pytest

np = pytest.importorskip("numpy")

from jarvis.core.agent_manager import AgentManager  # noqa: E402
from jarvis.core.cost_guard import CostGuard  # noqa: E402
from jarvis.core.errors import ProviderUnavailableError  # noqa: E402
from jarvis.core.orchestrator import Orchestrator  # noqa: E402
from jarvis.voice.audio import AudioClip, concat, read_wav, write_wav  # noqa: E402
from jarvis.voice.devices import ClipSource, MemorySink  # noqa: E402
from jarvis.voice.session import (  # noqa: E402
    FAILURE_MESSAGE,
    GOODBYE_MESSAGE,
    Phase,
    TurnOutcome,
    VoiceSession,
)
from jarvis.voice.stt import STTProvider, Transcript  # noqa: E402
from jarvis.voice.text import CODE_PLACEHOLDER, speakable, split_sentences  # noqa: E402
from jarvis.voice.tts import MacSayTTS, TTSProvider  # noqa: E402
from jarvis.voice.vad import FRAME_MS, EnergyVad, VadConfig, VadEvent  # noqa: E402
from tests.fakes import FakeAgent, FakeClock  # noqa: E402

RATE = 16_000


def tone(seconds, amplitude=0.3, freq=220.0, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return AudioClip((amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), rate)


def silence(seconds, rate=RATE):
    return AudioClip(np.zeros(int(seconds * rate), dtype=np.float32), rate)


# -- text -------------------------------------------------------------------------


def test_speakable_strips_markdown_and_code():
    text = (
        "## Docker\n"
        "**Docker** é uma *plataforma*. Veja [a doc](https://docs.docker.com).\n"
        "- instale o `docker`\n"
        "1. rode https://example.com\n"
        "```bash\nbrew install docker\n```\n"
        "Pronto!"
    )
    spoken = speakable(text)
    assert "#" not in spoken and "*" not in spoken and "`" not in spoken
    assert "brew install" not in spoken
    assert CODE_PLACEHOLDER.strip() in spoken
    assert "a doc" in spoken and "https" not in spoken
    assert "instale o docker." in spoken


def test_split_sentences_and_long_sentences():
    assert split_sentences("Olá! Tudo bem? Sim.") == ["Olá!", "Tudo bem?", "Sim."]
    long = "palavra, " * 60
    parts = split_sentences(long, max_chars=100)
    assert all(len(p) <= 101 for p in parts)
    assert " ".join(parts).replace(" ", "") == long.replace(" ", "").strip()


# -- audio ------------------------------------------------------------------------


def test_wav_round_trip_and_resampling(tmp_path):
    clip = tone(0.5, rate=22_050)
    path = tmp_path / "x.wav"
    write_wav(clip, path)
    back = read_wav(path)
    assert back.sample_rate == 22_050
    assert np.allclose(back.samples, clip.samples, atol=1e-3)
    resampled = back.resampled(RATE)
    assert resampled.sample_rate == RATE
    assert abs(resampled.duration - 0.5) < 0.01
    assert concat([tone(0.2), silence(0.3)]).duration == pytest.approx(0.5)


# -- VAD --------------------------------------------------------------------------


def frames_of(*clips):
    samples = concat(list(clips)).samples
    size = RATE * FRAME_MS // 1000
    return [samples[i : i + size] for i in range(0, len(samples) - size + 1, size)]


def run_vad(vad, frames):
    events = []
    for index, frame in enumerate(frames):
        event = vad.feed(frame)
        if event is not VadEvent.NONE:
            events.append((event, index))
            if event in (VadEvent.SPEECH_END, VadEvent.TIMEOUT):
                break
    return events


def test_vad_detects_start_and_end_of_speech():
    vad = EnergyVad(VadConfig(silence_ms=600))
    noise = AudioClip(np.random.default_rng(1).normal(0, 0.002, RATE).astype(np.float32), RATE)
    events = run_vad(vad, frames_of(noise, tone(1.0), silence(1.0)))
    assert [e for e, _ in events] == [VadEvent.SPEECH_START, VadEvent.SPEECH_END]
    utterance = vad.utterance()
    # Speech (1 s) + pre-roll (~0.3 s) + the silence that ended it (~0.6 s).
    assert 1.3 <= len(utterance) / RATE <= 2.1


def test_vad_times_out_without_speech():
    vad = EnergyVad(VadConfig(no_speech_timeout=1.0))
    events = run_vad(vad, frames_of(silence(2.0)))
    assert events == [(VadEvent.TIMEOUT, events[0][1])]
    assert vad.utterance().size == 0


def test_vad_ignores_a_short_click():
    vad = EnergyVad(VadConfig(no_speech_timeout=1.5))
    events = run_vad(vad, frames_of(silence(0.5), tone(0.03), silence(1.5)))
    assert [e for e, _ in events] == [VadEvent.TIMEOUT]


def test_vad_caps_very_long_speech():
    vad = EnergyVad(VadConfig(max_seconds=1.0))
    events = run_vad(vad, frames_of(silence(0.4), tone(3.0)))
    assert [e for e, _ in events] == [VadEvent.SPEECH_START, VadEvent.SPEECH_END]


# -- session (fake STT/TTS, real orchestrator with fake agents) ---------------------


class ScriptedSTT(STTProvider):
    name = "scripted"

    def __init__(self, texts):
        self.texts = list(texts)
        self.heard = []

    async def transcribe(self, clip):
        self.heard.append(clip)
        return Transcript(self.texts.pop(0), "pt", clip.duration, 1.0)


class ToneTTS(TTSProvider):
    name = "tone"

    def __init__(self):
        self.sentences = []

    async def synthesize(self, text):
        self.sentences.append(text)
        return tone(0.1)


def make_session(texts, agents, clips=None):
    manager = AgentManager(agents, CostGuard(), clock=FakeClock())
    events = []
    source = ClipSource(clips or [tone(0.8) for _ in texts])
    sink = MemorySink()
    stt, tts = ScriptedSTT(texts), ToneTTS()
    session = VoiceSession(
        Orchestrator(manager),
        stt,
        tts,
        source,
        sink,
        vad_config=VadConfig(no_speech_timeout=2.0),
        on_event=lambda phase, data: events.append(phase),
    )
    return session, stt, tts, sink, events


async def test_one_voice_turn_end_to_end():
    agent = FakeAgent("a", ["Docker é uma plataforma. Ela usa contêineres."])
    session, stt, tts, sink, events = make_session(["JARVIS, explique Docker."], [agent])
    result = await session.turn()
    assert result.outcome is TurnOutcome.ANSWERED
    assert result.transcript.text == "JARVIS, explique Docker."
    assert agent.calls[0].messages[-1].content == "JARVIS, explique Docker."
    assert tts.sentences == ["Docker é uma plataforma.", "Ela usa contêineres."]
    assert len(sink.played) == 2
    assert events == [
        Phase.LISTENING,
        Phase.HEARD,
        Phase.THINKING,
        Phase.ANSWER,
        Phase.SPEAKING,
        Phase.IDLE,
    ]
    assert {"stt_ms", "think_ms", "speak_ms"} <= result.timings.keys()
    assert 0.8 <= stt.heard[0].duration <= 2.2


async def test_continuous_conversation_keeps_context_until_goodbye():
    agent = FakeAgent("a", ["Docker é uma plataforma.", "Use o Docker Desktop."])
    session, _, tts, _, _ = make_session(
        ["JARVIS, explique Docker.", "E como instalo no Mac?", "Tchau, JARVIS!"], [agent]
    )
    answered = await session.run()
    assert answered == 2
    second = [(m.role, m.content) for m in agent.calls[1].messages[1:]]
    assert second == [
        ("user", "JARVIS, explique Docker."),
        ("assistant", "Docker é uma plataforma."),
        ("user", "E como instalo no Mac?"),
    ]
    assert tts.sentences[-1] == GOODBYE_MESSAGE
    assert len(agent.calls) == 2  # the goodbye never reached an agent


async def test_whisper_hallucinations_are_ignored():
    agent = FakeAgent("a")
    session, *_ = make_session(["Legendas pela comunidade Amara.org"], [agent])
    assert (await session.turn()).outcome is TurnOutcome.NOTHING_HEARD
    assert agent.calls == []


async def test_silence_means_nothing_heard():
    session, stt, *_ = make_session(["nunca usado"], [FakeAgent("a")], clips=[silence(0.1)])
    assert (await session.turn()).outcome is TurnOutcome.NOTHING_HEARD
    assert stt.heard == []
    assert await session.run(stop_on_silence=True) == 0


async def test_all_agents_failing_is_spoken_politely():
    agent = FakeAgent("a", [ProviderUnavailableError("down")])
    session, _, tts, _, events = make_session(["oi"], [agent])
    assert (await session.turn()).outcome is TurnOutcome.FAILED
    assert tts.sentences == [FAILURE_MESSAGE]
    assert Phase.ERROR in events


async def test_microphone_is_closed_while_speaking():
    """Half-duplex: the source iterator is closed before JARVIS starts talking."""
    opened = []

    class TrackingSource(ClipSource):
        async def frames(self):
            opened.append("open")
            try:
                async for frame in super().frames():
                    yield frame
            finally:
                opened.append("closed")

    class CheckingSink(MemorySink):
        async def play(self, clip):
            assert opened[-1] == "closed"
            await super().play(clip)

    manager = AgentManager([FakeAgent("a", ["Olá."])], CostGuard(), clock=FakeClock())
    session = VoiceSession(
        Orchestrator(manager),
        ScriptedSTT(["oi"]),
        ToneTTS(),
        TrackingSource([tone(0.8)]),
        CheckingSink(),
        vad_config=VadConfig(no_speech_timeout=2.0),
    )
    assert (await session.turn()).outcome is TurnOutcome.ANSWERED
    assert opened == ["open", "closed"]


def test_stop_phrases_ignore_case_and_accents():
    session, *_ = make_session([], [FakeAgent("a")])
    assert session.is_stop("Tchau, JARVIS!")
    assert session.is_stop("ok, pode parar jarvis")
    assert not session.is_stop("tchau por hoje, mas continue")


# -- real macOS voice (local, free) --------------------------------------------------


@pytest.mark.skipif(shutil.which("say") is None, reason="macOS say not available")
async def test_macos_say_produces_audio():
    clip = await MacSayTTS("Luciana").synthesize("-Olá, teste.")  # leading dash is safe
    assert clip.sample_rate == 22_050
    assert clip.duration > 0.3
    assert float(np.abs(clip.samples).max()) > 0.01


def test_stop_phrases_accept_whisper_spelling_variants():
    session, *_ = make_session([], [FakeAgent("a")])
    assert session.is_stop("Xau, Jarvis.")
    assert session.is_stop("chau jarbas")


async def test_voice_turns_ask_for_short_spoken_answers():
    agent = FakeAgent("a", ["Ok."])
    session, *_ = make_session(["oi"], [agent])
    await session.turn()
    system = agent.calls[0].messages[0]
    assert system.role == "system"
    assert "três frases curtas" in system.content


def test_stop_phrases_tolerate_small_transcription_errors_in_short_utterances():
    session, *_ = make_session([], [FakeAgent("a")])
    assert session.is_stop("Jao, JARVIS.")
    assert not session.is_stop("JARVIS")
    assert not session.is_stop("Jarvis, qual a capital da França?")
    assert not session.is_stop("tchau")
