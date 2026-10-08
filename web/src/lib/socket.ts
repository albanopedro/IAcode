// WebSocket to the JARVIS server with automatic reconnection.
import { type ClientMessage, type ServerEvent, parseServerEvent } from "./protocol";

export interface SocketHandlers {
  onEvent: (event: ServerEvent) => void;
  onAudio: (wav: ArrayBuffer) => void;
  onStatus: (status: "connecting" | "open" | "closed") => void;
}

export class JarvisSocket {
  private ws: WebSocket | null = null;
  private retry = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private closedByUser = false;

  constructor(
    private readonly url: string,
    private readonly handlers: SocketHandlers,
  ) {}

  connect(): void {
    this.closedByUser = false;
    this.handlers.onStatus("connecting");
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.retry = 0;
      this.handlers.onStatus("open");
    };
    ws.onmessage = (message) => {
      if (message.data instanceof ArrayBuffer) {
        this.handlers.onAudio(message.data);
        return;
      }
      const event = parseServerEvent(String(message.data));
      if (event) this.handlers.onEvent(event);
    };
    ws.onclose = () => {
      this.ws = null;
      this.handlers.onStatus("closed");
      if (!this.closedByUser) this.scheduleReconnect();
    };
    this.ws = ws;
  }

  private scheduleReconnect(): void {
    const delay = Math.min(1000 * 2 ** this.retry, 15_000);
    this.retry += 1;
    this.timer = setTimeout(() => this.connect(), delay);
  }

  get isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  send(message: ClientMessage): boolean {
    if (!this.isOpen) return false;
    this.ws!.send(JSON.stringify(message));
    return true;
  }

  sendAudio(frame: ArrayBuffer): void {
    if (this.isOpen) this.ws!.send(frame);
  }

  close(): void {
    this.closedByUser = true;
    if (this.timer) clearTimeout(this.timer);
    this.ws?.close();
  }
}

export function defaultSocketUrl(location: Location = window.location): string {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws`;
}
