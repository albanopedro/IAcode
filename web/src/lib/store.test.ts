import { describe, expect, it } from "vitest";
import type { AnswerEvent, ServerEvent } from "./protocol";
import { parseServerEvent } from "./protocol";
import { type UiState, initialState, reducer } from "./store";

function apply(state: UiState, ...events: ServerEvent[]): UiState {
  return events.reduce((s, event) => reducer(s, { kind: "server", event }), state);
}

const answer = (text: string, agent = "opencode:space-bunny-free"): AnswerEvent => ({
  type: "answer",
  mode: "voice",
  text,
  agent_id: agent,
  model: "m",
  task: "chat",
  latency_ms: 1800,
  cost: 0,
  attempts: [{ agent_id: agent, ok: true, error: null, latency_ms: 1800 }],
  ranking: [],
});

describe("reducer", () => {
  it("builds the conversation from voice turns and tracks the current agent", () => {
    const state = apply(
      initialState,
      { type: "state", state: "listening" },
      { type: "transcript", text: "explique Docker", stt_ms: 1300 },
      { type: "state", state: "thinking" },
      answer("Docker é uma plataforma."),
    );
    expect(state.serverState).toBe("thinking");
    expect(state.entries.map((e) => [e.role, e.mode, e.text])).toEqual([
      ["user", "voice", "explique Docker"],
      ["jarvis", "voice", "Docker é uma plataforma."],
    ]);
    expect(state.currentAgentId).toBe("opencode:space-bunny-free");
    expect(new Set(state.entries.map((e) => e.id)).size).toBe(2);
  });

  it("adds typed messages and errors, and clears", () => {
    let state = reducer(initialState, { kind: "user", text: "oi" });
    state = apply(state, { type: "error", message: "nenhum agente gratuito conseguiu responder" });
    expect(state.entries.map((e) => e.role)).toEqual(["user", "error"]);
    state = apply(state, { type: "cleared" });
    expect(state.entries).toEqual([]);
    expect(state.currentAgentId).toBeNull();
  });

  it("follows the voice loop and resets when the connection drops", () => {
    let state = apply(initialState, { type: "voice", active: true, stt: "whisper", tts: "say" });
    expect(state.voiceActive).toBe(true);
    expect(state.voiceEngines).toBe("whisper · say");
    state = reducer({ ...state, serverState: "speaking" }, { kind: "connection", status: "closed" });
    expect(state.voiceActive).toBe(false);
    expect(state.serverState).toBe("idle");
  });

  it("keeps the agent list and cost mode", () => {
    const state = apply(initialState, { type: "agents", cost_mode: "FREE_ONLY", agents: [] });
    expect(state.costMode).toBe("FREE_ONLY");
  });
});

describe("memory events", () => {
  it("replaces the conversation with a reopened history", () => {
    let state = reducer(initialState, { kind: "user", text: "algo da conversa anterior" });
    state = apply(state, {
      type: "history",
      conversation_id: "c1",
      title: "explique Docker",
      has_summary: true,
      messages: [
        { role: "user", text: "explique Docker", agent_id: null, created_at: 1 },
        { role: "assistant", text: "Docker é…", agent_id: "opencode:x", created_at: 2 },
      ],
    });
    expect(state.entries.map((e) => [e.role, e.text])).toEqual([
      ["user", "explique Docker"],
      ["jarvis", "Docker é…"],
    ]);
    expect(state.conversationId).toBe("c1");
    expect(state.hasSummary).toBe(true);
    expect(new Set(state.entries.map((e) => e.id)).size).toBe(2);
    expect(state.entries[0].id).toBeGreaterThan(1); // ids never repeat after a reload
  });

  it("keeps conversations and facts", () => {
    const state = apply(
      initialState,
      { type: "conversations", conversations: [{ id: "c1", title: "t", updated_at: 1, messages: 2 }] },
      { type: "facts", enabled: true, facts: [{ id: 1, text: "uso macOS", created_at: 1 }] },
    );
    expect(state.conversations).toHaveLength(1);
    expect(state.memoryEnabled).toBe(true);
    expect(state.facts[0].text).toBe("uso macOS");
  });
});

describe("parseServerEvent", () => {
  it("rejects garbage", () => {
    expect(parseServerEvent("nope")).toBeNull();
    expect(parseServerEvent('{"no":"type"}')).toBeNull();
    expect(parseServerEvent('{"type":"cleared"}')).toEqual({ type: "cleared" });
  });
});
