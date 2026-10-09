import { useEffect, useRef } from "react";
import type { ChatEntry } from "../lib/store";

const TASK_LABELS: Record<string, string> = {
  chat: "conversa",
  code: "código",
  math: "matemática",
  research: "pesquisa",
};

function shortAgent(id: string): string {
  return id.replace(/^opencode:/, "").replace(/-free$/, "").replace(/^local:/, "💻 ");
}

export function Conversation({ entries }: { entries: ChatEntry[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [entries.length]);

  if (entries.length === 0) {
    return (
      <div className="conversation empty">
        <p>Fale com o JARVIS pelo microfone ou escreva abaixo.</p>
        <p className="hint">Diga “tchau JARVIS” para encerrar a conversa por voz.</p>
        <p className="hint">Para eu guardar algo: “JARVIS, lembre que…”.</p>
      </div>
    );
  }

  return (
    <div className="conversation" aria-live="polite">
      {entries.map((entry) => (
        <article key={entry.id} className={`entry entry-${entry.role}`}>
          <header>
            {entry.role === "user" ? "Você" : entry.role === "jarvis" ? "JARVIS" : "Aviso"}
            {entry.agentId === "jarvis:memoria" && <span className="tag memory">🧠 memória</span>}
            {entry.mode === "voice" && <span className="tag">🎙 voz</span>}
            {entry.private && (
              <span className="tag private" title="Só local: nunca enviado para fora deste Mac">
                🔒 privado
              </span>
            )}
            {entry.agentId && entry.agentId !== "jarvis:memoria" && (
              <span className="tag agent">{shortAgent(entry.agentId)}</span>
            )}
            {entry.task && <span className="tag">{TASK_LABELS[entry.task] ?? entry.task}</span>}
            {entry.latencyMs !== undefined && (
              <span className="tag muted">{(entry.latencyMs / 1000).toFixed(1)}s</span>
            )}
          </header>
          <p>{entry.text}</p>
          {entry.tools && entry.tools.length > 0 && (
            <ul className="tools-used" aria-label="Ferramentas usadas">
              {entry.tools.map((t, i) => (
                <li key={i} className={t.ok ? "ok" : "failed"} title={t.decision}>
                  🔧 {t.title}
                  {t.decision === "denied_by_user" && " · negada por você"}
                  {t.decision === "confirmed" && " · autorizada"}
                </li>
              ))}
            </ul>
          )}
          {entry.attempts && entry.attempts.some((a) => !a.ok) && (
            <details>
              <summary>
                {entry.attempts.filter((a) => !a.ok).length} agente(s) falharam antes
              </summary>
              <ul>
                {entry.attempts
                  .filter((a) => !a.ok)
                  .map((a, i) => (
                    <li key={i}>
                      <strong>{shortAgent(a.agent_id)}</strong>: {a.error}
                    </li>
                  ))}
              </ul>
            </details>
          )}
        </article>
      ))}
      <div ref={endRef} />
    </div>
  );
}
