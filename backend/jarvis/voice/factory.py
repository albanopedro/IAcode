"""Build the voice components from settings."""

from __future__ import annotations

from jarvis.config import Settings, VoiceConfig
from jarvis.voice.stt import STTProvider, WhisperCppSTT
from jarvis.voice.tts import MacSayTTS, PiperTTS, TTSProvider
from jarvis.voice.vad import VadConfig


def build_stt(settings: Settings) -> STTProvider:
    voice = settings.voice
    return WhisperCppSTT(
        voice.stt_model,
        models_dir=settings.data_dir / "models" / "whisper",
        language=voice.language,
        threads=voice.stt_threads,
        prompt=voice.stt_prompt,
    )


def build_tts(settings: Settings, engine: str | None = None) -> TTSProvider:
    voice = settings.voice
    engine = engine or voice.tts_engine
    if engine == "say":
        return MacSayTTS(voice.say_voice, voice.say_rate)
    if engine == "piper":
        return PiperTTS(voice.piper_voice, settings.data_dir / "models" / "piper")
    raise ValueError(f"unknown TTS engine {engine!r} (use 'say' or 'piper')")


def vad_config(voice: VoiceConfig) -> VadConfig:
    return VadConfig(
        silence_ms=voice.silence_ms,
        no_speech_timeout=voice.no_speech_timeout,
        max_seconds=voice.max_utterance_seconds,
    )
