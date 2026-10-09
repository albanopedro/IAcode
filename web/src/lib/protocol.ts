// Messages exchanged with the JARVIS server over /ws (see backend/jarvis/server/connection.py).

export type JarvisState = "idle" | "loading" | "listening" | "thinking" | "speaking";

export type Health = "available" | "cooldown" | "offline" | "unconfigured" | "blocked" | "unknown";

export interface AgentStatus {
  id: string;
  name: string;
  provider: string;
  model: string;
  health: Health;
  available: boolean;
  remaining_usage: number | null;
  remaining_source: "provider" | "local" | null;
  rate_limit: number | null;
  rpm_limit: number | null;
  requests_last_minute: number;
  cooldown_until: string | null;
  last_error: string | null;
  priority: number;
  privacy: string;
  success_rate: number | null;
  avg_latency_ms: number | null;
  requests_today: number;
}

export interface Attempt {
  agent_id: string;
  ok: boolean;
  error: string | null;
  latency_ms: number;
}

export interface ToolUse {
  name: string;
  title: string;
  decision: string;
  ok: boolean;
}

export interface ConfirmRequest {
  type: "confirm";
  id: string;
  tool: string;
  title: string;
  risk: "safe" | "sensitive" | "dangerous";
  description: string;
  reason: string;
  timeout: number;
}

export interface AnswerEvent {
  type: "answer";
  mode: "text" | "voice";
  text: string;
  agent_id: string;
  model: string;
  task: string;
  latency_ms: number;
  cost: number;
  attempts: Attempt[];
  ranking: { agent_id: string; score: number }[];
  tools?: ToolUse[];
}

export interface HistoryMessage {
  role: "user" | "assistant";
  text: string;
  agent_id: string | null;
  created_at: number;
}

export interface ConversationSummary {
  id: string;
  title: string;
  updated_at: number;
  messages: number;
}

export interface Fact {
  id: number;
  text: string;
  created_at: number;
}

export type ServerEvent =
  | { type: "hello"; voice_ready: boolean; tts_engine: string; max_text: number }
  | { type: "state"; state: JarvisState }
  | { type: "transcript"; text: string; stt_ms: number }
  | AnswerEvent
  | { type: "agents"; cost_mode: string; agents: AgentStatus[] }
  | { type: "voice"; active: boolean; stt?: string; tts?: string }
  | { type: "stop_audio" }
  | { type: "cleared" }
  | { type: "error"; message: string; attempts?: Attempt[] }
  | {
      type: "history";
      conversation_id: string | null;
      title: string;
      has_summary: boolean;
      messages: HistoryMessage[];
    }
  | { type: "conversations"; current?: string | null; conversations: ConversationSummary[] }
  | { type: "facts"; enabled: boolean; facts: Fact[] }
  | { type: "tool"; phase: "start" | "end"; tool: string; title: string; ok?: boolean }
  | ConfirmRequest
  | { type: "confirm_closed"; id: string };

export type TtsEngine = "say" | "piper";

export type ClientMessage =
  | { type: "text"; text: string; speak?: boolean }
  | { type: "voice_start"; tts?: TtsEngine }
  | { type: "voice_stop" }
  | { type: "interrupt" }
  | { type: "clear" }
  | { type: "new_conversation" }
  | { type: "open_conversation"; id: string }
  | { type: "delete_conversation"; id: string }
  | { type: "list_conversations" }
  | { type: "list_facts" }
  | { type: "forget_fact"; id: number }
  | { type: "clear_facts"; confirm: true }
  | { type: "confirm_reply"; id: string; approved: boolean }
  | { type: "status" };

export function parseServerEvent(raw: string): ServerEvent | null {
  try {
    const value = JSON.parse(raw) as { type?: unknown };
    return typeof value?.type === "string" ? (value as ServerEvent) : null;
  } catch {
    return null;
  }
}
