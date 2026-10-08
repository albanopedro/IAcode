import type { ConversationSummary } from "../lib/protocol";

interface Props {
  conversations: ConversationSummary[];
  currentId: string | null;
  hasSummary: boolean;
  disabled: boolean;
  onOpen: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

function label(c: ConversationSummary): string {
  const when = new Date(c.updated_at * 1000).toLocaleString("pt-BR", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
  return `${when} · ${c.title || "(sem título)"} (${c.messages})`;
}

export function HistoryBar({ conversations, currentId, hasSummary, disabled, onOpen, onNew, onDelete }: Props) {
  const current = conversations.find((c) => c.id === currentId);
  return (
    <div className="history-bar">
      <select
        aria-label="Conversas salvas"
        value={current ? current.id : ""}
        disabled={disabled || conversations.length === 0}
        onChange={(e) => e.target.value && onOpen(e.target.value)}
      >
        {!current && <option value="">Nova conversa</option>}
        {conversations.map((c) => (
          <option key={c.id} value={c.id}>
            {label(c)}
          </option>
        ))}
      </select>
      <button type="button" className="secondary" onClick={onNew} disabled={disabled}>
        ＋ Nova
      </button>
      {current && (
        <button
          type="button"
          className="secondary"
          disabled={disabled}
          onClick={() => {
            if (window.confirm(`Apagar a conversa “${current.title}”?`)) onDelete(current.id);
          }}
          aria-label="Apagar esta conversa"
          title="Apagar esta conversa"
        >
          🗑
        </button>
      )}
      {hasSummary && (
        <span className="summary-tag" title="As mensagens mais antigas foram resumidas para o contexto">
          resumida
        </span>
      )}
    </div>
  );
}
