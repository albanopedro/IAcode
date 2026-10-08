import { describe, expect, it } from "vitest";
import { FRAME_SAMPLES, FrameAssembler, downsample, floatToPcm16, rms } from "./audio";

describe("downsample", () => {
  it("averages 48 kHz down to 16 kHz", () => {
    const input = Float32Array.from([0, 0.3, 0.6, 1, 1, 1]);
    expect(Array.from(downsample(input, 48_000))).toEqual([
      expect.closeTo(0.3, 5),
      expect.closeTo(1, 5),
    ]);
  });

  it("keeps the duration for 44.1 kHz", () => {
    const input = new Float32Array(44_100);
    expect(downsample(input, 44_100).length).toBe(16_000);
  });

  it("refuses to upsample", () => {
    expect(() => downsample(new Float32Array(10), 8_000)).toThrow();
  });
});

describe("floatToPcm16", () => {
  it("clips and scales", () => {
    expect(Array.from(floatToPcm16(Float32Array.from([-2, -1, 0, 0.5, 1, 3])))).toEqual([
      -32768, -32768, 0, 16384, 32767, 32767,
    ]);
  });
});

describe("rms", () => {
  it("measures loudness", () => {
    expect(rms(new Float32Array(0))).toBe(0);
    expect(rms(Float32Array.from([0.5, -0.5]))).toBeCloseTo(0.5);
  });
});

describe("FrameAssembler", () => {
  it("emits fixed 30 ms frames and keeps the remainder", () => {
    const assembler = new FrameAssembler();
    expect(assembler.push(new Float32Array(300))).toHaveLength(0);
    const frames = assembler.push(new Float32Array(700)); // 1000 samples in total
    expect(frames).toHaveLength(2);
    expect(frames.every((f) => f.length === FRAME_SAMPLES)).toBe(true);
    expect(assembler.push(new Float32Array(440))).toHaveLength(1); // 40 + 440 = 480
  });

  it("can be reset (audio captured while muted is dropped)", () => {
    const assembler = new FrameAssembler();
    assembler.push(new Float32Array(400));
    assembler.reset();
    expect(assembler.push(new Float32Array(400))).toHaveLength(0);
  });
});
