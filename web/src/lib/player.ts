// Plays JARVIS's voice: WAV sentences arrive one by one and play back to back.

export class VoicePlayer {
  private context: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private queue: ArrayBuffer[] = [];
  private current: AudioBufferSourceNode | null = null;
  private playing = false;
  private readonly samples = new Float32Array(1024);

  constructor(private readonly onBusyChange: (busy: boolean) => void) {}

  get busy(): boolean {
    return this.playing || this.queue.length > 0;
  }

  private ensureContext(): AudioContext {
    if (!this.context) {
      this.context = new AudioContext();
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analyser.connect(this.context.destination);
    }
    return this.context;
  }

  /** Call from a click handler: browsers only allow audio after a user gesture. */
  async unlock(): Promise<void> {
    await this.ensureContext().resume();
  }

  enqueue(wav: ArrayBuffer): void {
    const wasBusy = this.busy;
    this.queue.push(wav);
    if (!wasBusy) this.onBusyChange(true);
    if (!this.playing) void this.playNext();
  }

  private async playNext(): Promise<void> {
    const next = this.queue.shift();
    if (!next) {
      this.playing = false;
      this.onBusyChange(false);
      return;
    }
    this.playing = true;
    const context = this.ensureContext();
    try {
      const buffer = await context.decodeAudioData(next);
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.analyser!);
      source.onended = () => {
        if (this.current === source) this.current = null;
        void this.playNext();
      };
      this.current = source;
      source.start();
    } catch {
      void this.playNext(); // skip a sentence that could not be decoded
    }
  }

  stop(): void {
    this.queue = [];
    const current = this.current;
    this.current = null;
    if (current) {
      current.onended = null;
      current.stop();
    }
    if (this.playing) {
      this.playing = false;
      this.onBusyChange(false);
    }
  }

  /** Current output loudness (0..1), for the orb. */
  get level(): number {
    if (!this.analyser || !this.playing) return 0;
    this.analyser.getFloatTimeDomainData(this.samples);
    let sum = 0;
    for (const s of this.samples) sum += s * s;
    return Math.sqrt(sum / this.samples.length);
  }
}
