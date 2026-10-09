import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { AgentPanel } from "./components/AgentPanel";
import { ConfirmDialog } from "./components/ConfirmDialog";
import { Conversation } from "./components/Conversation";
import { HistoryBar } from "./components/HistoryBar";
import { MemoryPanel } from "./components/MemoryPanel";
import { Orb } from "./components/Orb";
import { Microphone } from "./lib/mic";
import { VoicePlayer } from "./lib/player";
import type { TtsEngine } from "./lib/protocol";
import { JarvisSocket, defaultSocketUrl } from "./lib/socket";
import { STATE_LABELS, initialState, reducer } from "./lib/store";

const STATUS_POLL_MS = 15_000;

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [playerBusy, setPlayerBusy] = useState(false);
  const [speakAnswers, setSpeakAnswers] = useState(false);
  const [tts, setTts] = useState<TtsEngine>("say");
  const [wakeWord, setWakeWord] = useState(false);
  const [draft, setDraft] = useState("");
  const [micError, setMicError] = useState<string | null>(null);

  const socket = useRef<JarvisSocket | null>(null);
  const mic = useRef<Microphone | null>(null);
  const player = useRef<VoicePlayer | null>(null);

  useEffect(() => {
    const microphone = new Microphone((frame) => socket.current?.sendAudio(frame));
    const voice = new VoicePlayer((busy) => {
      setPlayerBusy(busy);
      microphone.muted = busy; // half-duplex: never send JARVIS's own voice back
    });
    const ws = new JarvisSocket(defaultSocketUrl(), {
      onEvent: (event) => {
        if (event.type === "stop_audio") voice.stop();
        if (event.type === "voice" && !event.active) void microphone.stop();
        dispatch({ kind: "server", event });
      },
      onAudio: (wav) => voice.enqueue(wav),
      onStatus: (status) => {
        if (status !== "open") void microphone.stop();
        dispatch({ kind: "connection", status });
      },
    });
    mic.current = microphone;
    player.current = voice;
    socket.current = ws;
    ws.connect();
    const poll = setInterval(() => ws.send({ type: "status" }), STATUS_POLL_MS);
    return () => {
      clearInterval(poll);
      ws.close();
      voice.stop();
      void microphone.stop();
    };
  }, []);

  const online = state.connection === "open";
  const orbState = !online ? "offline" : playerBusy ? "speaking" : state.serverState;

  const level = useCallback(() => {
    if (player.current?.busy) return player.current.level;
    return mic.current?.active ? mic.current.level : 0;
  }, []);

  async function toggleVoice() {
    setMicError(null);
    if (state.voiceActive) {
      socket.current?.send({ type: "voice_stop" });
      await mic.current?.stop();
      return;
    }
    try {
      await player.current?.unlock();
      await mic.current?.start();
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      setMicError(`Não consegui acessar o microfone (${reason}).`);
      return;
    }
    socket.current?.send({ type: "voice_start", tts, wake: wakeWord });
  }

  function interrupt() {
    player.current?.stop();
    socket.current?.send({ type: "interrupt" });
  }

  async function sendText() {
    const text = draft.trim();
    if (!text || !online) return;
    if (speakAnswers) await player.current?.unlock();
    if (socket.current?.send({ type: "text", text, speak: speakAnswers })) {
      dispatch({ kind: "user", text });
      setDraft("");
    }
  }

  const busy = state.serverState === "thinking";
  const label = STATE_LABELS[orbState];

  return (
    <div className="app">
      <header className="topbar">
        <h1>
          J.A.R.V.I.S. <span>assistente multimodelo · 100% gratuito</span>
        </h1>
        <span className={`connection ${state.connection}`}>
          {state.connection === "open" ? "conectado" : state.connection === "connecting" ? "conectando…" : "desconectado"}
        </span>
      </header>

      <main className="layout">
        <section className="center">
          <Orb state={orbState} level={level} />
          <p className={`state-label state-${orbState}`} aria-live="polite">
            {label}
          </p>
          {state.voiceActive && state.voiceEngines && <p className="engines">{state.voiceEngines}</p>}
          {state.activeTool && <p className="tool-running">🔧 usando {state.activeTool}…</p>}

          <div className="voice-controls">
            <button
              type="button"
              className={`mic ${state.voiceActive ? "on" : ""}`}
              onClick={() => void toggleVoice()}
              disabled={!online || state.serverState === "loading"}
              aria-pressed={state.voiceActive}
            >
              {state.voiceActive ? "⏹ Parar de ouvir" : "🎙 Conversar por voz"}
            </button>
            {(playerBusy || state.serverState === "speaking") && (
              <button type="button" className="secondary" onClick={interrupt}>
                ✋ Interromper
              </button>
            )}
          </div>
          {micError && <p className="mic-error">{micError}</p>}

          <div className="settings">
            <label>
              Voz
              <select value={tts} onChange={(e) => setTts(e.target.value as TtsEngine)} disabled={state.voiceActive}>
                <option value="say">macOS (Luciana)</option>
                <option value="piper">Piper (open source)</option>
              </select>
            </label>
            <label>
              <input type="checkbox" checked={speakAnswers} onChange={(e) => setSpeakAnswers(e.target.checked)} />
              Falar respostas escritas
            </label>
            <label title="O microfone fica ligado, mas nada é gravado nem enviado até você dizer “Hey Jarvis”">
              <input
                type="checkbox"
                checked={wakeWord}
                disabled={state.voiceActive}
                onChange={(e) => setWakeWord(e.target.checked)}
              />
              Ativar por “Hey Jarvis”
            </label>
            <label title="Só o modelo que roda neste Mac responde; nada é enviado para a internet">
              <input
                type="checkbox"
                checked={state.localOnly === true}
                disabled={!online}
                onChange={(e) => socket.current?.send({ type: "set_local_only", value: e.target.checked })}
              />
              🔒 Só local (nada sai do Mac)
            </label>
          </div>
        </section>

        <section className="chat">
          <HistoryBar
            conversations={state.conversations}
            currentId={state.conversationId}
            hasSummary={state.hasSummary}
            disabled={!online || busy}
            onOpen={(id) => socket.current?.send({ type: "open_conversation", id })}
            onNew={() => socket.current?.send({ type: "new_conversation" })}
            onDelete={(id) => socket.current?.send({ type: "delete_conversation", id })}
          />
          <Conversation entries={state.entries} />
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              void sendText();
            }}
          >
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void sendText();
                }
              }}
              maxLength={state.maxText}
              placeholder={online ? "Escreva para o JARVIS… (Enter envia)" : "Aguardando o servidor…"}
              rows={2}
              disabled={!online}
              aria-label="Mensagem"
            />
            <button type="submit" disabled={!online || busy || !draft.trim()}>
              Enviar
            </button>
          </form>
        </section>

        <div className="side">
          <AgentPanel agents={state.agents} currentAgentId={state.currentAgentId} costMode={state.costMode} />
          <MemoryPanel
            enabled={state.memoryEnabled}
            facts={state.facts}
            onForget={(id) => socket.current?.send({ type: "forget_fact", id })}
            onForgetAll={() => socket.current?.send({ type: "clear_facts", confirm: true })}
          />
        </div>
      </main>

      {state.confirms[0] && (
        <ConfirmDialog
          key={state.confirms[0].id}
          request={state.confirms[0]}
          onAnswer={(approved) =>
            socket.current?.send({ type: "confirm_reply", id: state.confirms[0].id, approved })
          }
        />
      )}
    </div>
  );
}
