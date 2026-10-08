// Microphone capture: AudioWorklet → 16 kHz PCM16 frames of 30 ms → WebSocket.
import { FrameAssembler, downsample, rms } from "./audio";

// The worklet only copies the raw samples to the main thread; resampling happens there.
const WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("jarvis-tap", Tap);
`;

export class Microphone {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private readonly assembler = new FrameAssembler();
  /** Half-duplex: while JARVIS talks, audio is captured for the level meter but not sent. */
  muted = false;
  level = 0;

  constructor(private readonly onFrame: (frame: ArrayBuffer) => void) {}

  get active(): boolean {
    return this.context !== null;
  }

  async start(): Promise<void> {
    if (this.context) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    const context = new AudioContext();
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    try {
      await context.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    const source = context.createMediaStreamSource(this.stream);
    const node = new AudioWorkletNode(context, "jarvis-tap");
    node.port.onmessage = (message: MessageEvent<Float32Array>) => {
      const chunk = downsample(message.data, context.sampleRate);
      this.level = rms(message.data);
      if (this.muted) {
        this.assembler.reset();
        return;
      }
      for (const frame of this.assembler.push(chunk)) this.onFrame(frame.buffer as ArrayBuffer);
    };
    source.connect(node);
    this.context = context;
    this.node = node;
  }

  async stop(): Promise<void> {
    this.node?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    await this.context?.close();
    this.context = null;
    this.stream = null;
    this.node = null;
    this.level = 0;
    this.assembler.reset();
  }
}
