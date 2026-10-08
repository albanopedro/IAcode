import type { AgentStatus, Health } from "../lib/protocol";

const HEALTH_LABEL: Record<Health, string> = {
  available: "disponível",
  cooldown: "em pausa",
  offline: "offline",
  unconfigured: "sem chave",
  blocked: "bloqueado",
  unknown: "verificando",
};

function details(agent: AgentStatus): string[] {
  const parts: string[] = [];
  if (agent.cooldown_until) {
    const seconds = Math.max(0, Math.round((Date.parse(agent.cooldown_until) - Date.now()) / 1000));
    parts.push(`volta em ${seconds}s`);
  }
  if (agent.remaining_usage !== null) {
    parts.push(`restam ${agent.remaining_usage}${agent.rate_limit ? `/${agent.rate_limit}` : ""}`);
  }
  if (agent.success_rate !== null) parts.push(`${Math.round(agent.success_rate * 100)}% ok`);
  if (agent.avg_latency_ms !== null) parts.push(`${(agent.avg_latency_ms / 1000).toFixed(1)}s`);
  if (agent.privacy === "zero_retention") parts.push("sem retenção");
  return parts;
}

interface Props {
  agents: AgentStatus[];
  currentAgentId: string | null;
  costMode: string;
}

export function AgentPanel({ agents, currentAgentId, costMode }: Props) {
  const usable = agents.filter((a) => a.health === "available").length;
  const sorted = [...agents].sort(
    (a, b) => Number(b.health === "available") - Number(a.health === "available") || b.priority - a.priority,
  );
  return (
    <aside className="agents" aria-label="Agentes">
      <header>
        <h2>Agentes</h2>
        <span className="cost-mode" title="Só agentes gratuitos podem ser usados">
          🔒 {costMode}
        </span>
      </header>
      <p className="summary">
        {usable} de {agents.length} disponíveis
      </p>
      <ul>
        {sorted.map((agent) => (
          <li
            key={agent.id}
            className={`agent health-${agent.health}${agent.id === currentAgentId ? " current" : ""}`}
            title={agent.last_error ?? agent.name}
          >
            <span className="dot" aria-hidden />
            <div>
              <div className="name">
                {agent.name}
                {agent.id === currentAgentId && <span className="badge">em uso</span>}
              </div>
              <div className="meta">
                {HEALTH_LABEL[agent.health]}
                {details(agent).map((d) => ` · ${d}`)}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </aside>
  );
}
