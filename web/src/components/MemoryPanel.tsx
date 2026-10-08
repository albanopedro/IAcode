import type { Fact } from "../lib/protocol";

interface Props {
  enabled: boolean;
  facts: Fact[];
  onForget: (id: number) => void;
  onForgetAll: () => void;
}

export function MemoryPanel({ enabled, facts, onForget, onForgetAll }: Props) {
  return (
    <section className="memory" aria-label="Memória de longo prazo">
      <header>
        <h2>Memória</h2>
        {facts.length > 0 && (
          <button
            type="button"
            className="link danger"
            onClick={() => {
              if (window.confirm("Apagar TODAS as lembranças de longo prazo?")) onForgetAll();
            }}
          >
            apagar tudo
          </button>
        )}
      </header>
      {!enabled ? (
        <p className="hint">Memória desligada na configuração.</p>
      ) : facts.length === 0 ? (
        <p className="hint">
          Nada guardado. Diga ou escreva: <em>“JARVIS, lembre que eu prefiro respostas curtas.”</em>
        </p>
      ) : (
        <ul>
          {facts.map((fact) => (
            <li key={fact.id}>
              <span>{fact.text}</span>
              <button
                type="button"
                className="icon"
                aria-label={`Esquecer: ${fact.text}`}
                title="Esquecer"
                onClick={() => onForget(fact.id)}
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
      <p className="hint small">
        🔒 Só agentes sem retenção de dados recebem estas lembranças.
      </p>
    </section>
  );
}
