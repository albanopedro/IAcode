// UI state: a pure reducer over server events, so it can be unit-tested.
import type {
  AgentStatus,
  Attempt,
  ConfirmRequest,
  ConversationSummary,
  Fact,
  JarvisState,
  ServerEvent,
  ToolUse,
} from "./protocol";

export type Connection = "connecting" | "open" | "closed";

export interface ChatEntry {
  id: number;
  role: "user" | "jarvis" | "error";
  text: string;
  mode: "text" | "voice";
  agentId?: string;
  task?: string;
  latencyMs?: number;
  attempts?: Attempt[];
  tools?: ToolUse[];
  private?: boolean; // exchanged in local-only mode: never sent outside this Mac
}

export interface UiState {
  connection: Connection;
  serverState: JarvisState;
  voiceActive: boolean;
  voiceEngines: string | null;
  voiceReady: boolean;
  ttsEngine: string;
  maxText: number;
  costMode: string;
  agents: AgentStatus[];
  entries: ChatEntry[];
  currentAgentId: string | null;
  nextId: number;
  conversationId: string | null;
  conversationTitle: string;
  hasSummary: boolean;
  conversations: ConversationSummary[];
  memoryEnabled: boolean;
  facts: Fact[];
  activeTool: string | null;
  confirms: ConfirmRequest[];
  localOnly: boolean;
}

export const initialState: UiState = {
  connection: "connecting",
  serverState: "idle",
  voiceActive: false,
  voiceEngines: null,
  voiceReady: false,
  ttsEngine: "say",
  maxText: 4000,
  costMode: "FREE_ONLY",
  agents: [],
  entries: [],
  currentAgentId: null,
  nextId: 1,
  conversationId: null,
  conversationTitle: "",
  hasSummary: false,
  conversations: [],
  memoryEnabled: false,
  facts: [],
  activeTool: null,
  confirms: [],
  localOnly: false,
};

export type Action =
  | { kind: "server"; event: ServerEvent }
  | { kind: "connection"; status: Connection }
  | { kind: "user"; text: string };

const MAX_ENTRIES = 200;

function addEntry(state: UiState, entry: Omit<ChatEntry, "id">): UiState {
  const entries = [...state.entries, { ...entry, id: state.nextId }].slice(-MAX_ENTRIES);
  return { ...state, entries, nextId: state.nextId + 1 };
}

export function reducer(state: UiState, action: Action): UiState {
  if (action.kind === "connection") {
    const next = { ...state, connection: action.status };
    // A dropped connection ends any voice loop on the server side.
    return action.status === "open"
      ? next
      : { ...next, voiceActive: false, serverState: "idle", confirms: [], activeTool: null };
  }
  if (action.kind === "user") {
    return addEntry(state, {
      role: "user",
      text: action.text,
      mode: "text",
      private: state.localOnly || undefined,
    });
  }

  const event = action.event;
  switch (event.type) {
    case "hello":
      return {
        ...state,
        voiceReady: event.voice_ready,
        ttsEngine: event.tts_engine,
        maxText: event.max_text,
        localOnly: event.local_only ?? false,
      };
    case "local_only":
      return { ...state, localOnly: event.value };
    case "state":
      return { ...state, serverState: event.state };
    case "transcript":
      return addEntry(state, {
        role: "user",
        text: event.text,
        mode: "voice",
        private: state.localOnly || undefined,
      });
    case "answer":
      return {
        ...addEntry(state, {
          role: "jarvis",
          text: event.text,
          mode: event.mode,
          agentId: event.agent_id,
          task: event.task,
          latencyMs: event.latency_ms,
          attempts: event.attempts,
          tools: event.tools,
          private: state.localOnly || undefined,
        }),
        currentAgentId: event.agent_id,
        activeTool: null,
      };
    case "agents":
      return { ...state, agents: event.agents, costMode: event.cost_mode };
    case "voice":
      return {
        ...state,
        voiceActive: event.active,
        voiceReady: event.active ? true : state.voiceReady,
        voiceEngines: event.active && event.stt && event.tts ? `${event.stt} · ${event.tts}` : state.voiceEngines,
      };
    case "cleared":
      return { ...state, entries: [], currentAgentId: null };
    case "error":
      return addEntry(state, {
        role: "error",
        text: event.message,
        mode: "text",
        attempts: event.attempts,
      });
    case "history": {
      const entries: ChatEntry[] = event.messages.slice(-MAX_ENTRIES).map((m, i) => ({
        id: state.nextId + i,
        role: m.role === "user" ? "user" : "jarvis",
        text: m.text,
        mode: "text",
        agentId: m.agent_id ?? undefined,
        private: m.private || undefined,
      }));
      return {
        ...state,
        entries,
        nextId: state.nextId + entries.length,
        conversationId: event.conversation_id,
        conversationTitle: event.title,
        hasSummary: event.has_summary,
        currentAgentId: null,
      };
    }
    case "conversations":
      return { ...state, conversations: event.conversations };
    case "facts":
      return { ...state, memoryEnabled: event.enabled, facts: event.facts };
    case "tool":
      return { ...state, activeTool: event.phase === "start" ? event.title : null };
    case "confirm":
      return { ...state, confirms: [...state.confirms.filter((c) => c.id !== event.id), event] };
    case "confirm_closed":
      return { ...state, confirms: state.confirms.filter((c) => c.id !== event.id) };
    case "wake":
      return { ...state, serverState: "listening" };
    case "stop_audio":
      return state;
  }
}

export const STATE_LABELS: Record<JarvisState | "offline", string> = {
  idle: "Pronto",
  loading: "Carregando a voz…",
  sleeping: "Dormindo — diga “Hey Jarvis”",
  listening: "Ouvindo…",
  thinking: "Pensando…",
  speaking: "Falando…",
  offline: "Sem conexão com o servidor",
};
