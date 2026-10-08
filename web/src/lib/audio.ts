// Pure audio helpers: the browser records at 44.1/48 kHz, the server wants 16 kHz PCM16.

export const TARGET_RATE = 16_000;
export const FRAME_SAMPLES = 480; // 30 ms at 16 kHz, the server's VAD frame

/** Downsample by averaging the input samples that fall into each output sample. */
export function downsample(input: Float32Array, inRate: number, outRate = TARGET_RATE): Float32Array {
  if (outRate === inRate) return input.slice();
  if (outRate > inRate) throw new Error("upsampling is not supported");
  const ratio = inRate / outRate;
  const length = Math.floor(input.length / ratio);
  const output = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), input.length);
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    output[i] = end > start ? sum / (end - start) : 0;
  }
  return output;
}

export function floatToPcm16(input: Float32Array): Int16Array {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
  }
  return output;
}

export function rms(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

/**
 * Collects resampled audio and hands out fixed 30 ms PCM16 frames.
 * Leftover samples wait for the next chunk, so no audio is lost between chunks.
 */
export class FrameAssembler {
  private buffer = new Float32Array(0);

  push(samples: Float32Array): Int16Array[] {
    const merged = new Float32Array(this.buffer.length + samples.length);
    merged.set(this.buffer);
    merged.set(samples, this.buffer.length);
    const frames: Int16Array[] = [];
    let offset = 0;
    while (merged.length - offset >= FRAME_SAMPLES) {
      frames.push(floatToPcm16(merged.subarray(offset, offset + FRAME_SAMPLES)));
      offset += FRAME_SAMPLES;
    }
    this.buffer = merged.slice(offset);
    return frames;
  }

  reset(): void {
    this.buffer = new Float32Array(0);
  }
}
