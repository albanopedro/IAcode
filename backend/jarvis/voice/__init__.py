"""Voice for JARVIS: microphone → speech-to-text → orchestrator → text-to-speech.

Everything here runs locally and costs nothing:
- STT: whisper.cpp (pywhispercpp, MIT) with a quantized Whisper model;
- TTS: macOS ``say`` (built in) or Piper (open source, GPL-3.0);
- VAD: a small energy-based detector (no extra dependency).

Heavy libraries are imported lazily, so the text-only JARVIS works without the
``[voice]`` extra installed.
"""
