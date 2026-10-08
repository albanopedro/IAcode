import { useEffect, useRef } from "react";
import type { JarvisState } from "../lib/protocol";

type OrbState = JarvisState | "offline";

// Hue, speed and ring count per state: listening reacts to your voice,
// speaking to JARVIS's voice, thinking spins.
const LOOK: Record<OrbState, { hue: number; sat: number; speed: number; glow: number }> = {
  idle: { hue: 192, sat: 80, speed: 0.15, glow: 0.35 },
  loading: { hue: 260, sat: 60, speed: 0.5, glow: 0.4 },
  listening: { hue: 186, sat: 95, speed: 0.3, glow: 0.6 },
  thinking: { hue: 38, sat: 95, speed: 1.4, glow: 0.55 },
  speaking: { hue: 205, sat: 100, speed: 0.6, glow: 0.75 },
  offline: { hue: 0, sat: 0, speed: 0.05, glow: 0.15 },
};

interface Props {
  state: OrbState;
  /** Returns the current audio loudness, 0..1 (mic while listening, voice while speaking). */
  level: () => number;
}

export function Orb({ state, level }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let frame = 0;
    let angle = 0;
    let smooth = 0;

    const draw = () => {
      const size = canvas.clientWidth;
      const ratio = window.devicePixelRatio || 1;
      if (canvas.width !== size * ratio) {
        canvas.width = size * ratio;
        canvas.height = size * ratio;
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      const look = LOOK[stateRef.current];
      const target = Math.min(level() * 6, 1);
      smooth += (target - smooth) * 0.25;
      angle += reduced ? 0 : look.speed * 0.02;

      const c = size / 2;
      const base = size * 0.22;
      const radius = base * (1 + smooth * 0.35);
      ctx.clearRect(0, 0, size, size);

      // Glow
      const glow = ctx.createRadialGradient(c, c, radius * 0.2, c, c, size / 2);
      glow.addColorStop(0, `hsla(${look.hue}, ${look.sat}%, 60%, ${look.glow})`);
      glow.addColorStop(1, `hsla(${look.hue}, ${look.sat}%, 50%, 0)`);
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, size, size);

      // Core
      const core = ctx.createRadialGradient(c - radius * 0.3, c - radius * 0.3, 2, c, c, radius);
      core.addColorStop(0, `hsla(${look.hue}, ${look.sat}%, 92%, 1)`);
      core.addColorStop(0.5, `hsla(${look.hue}, ${look.sat}%, 55%, 0.9)`);
      core.addColorStop(1, `hsla(${look.hue}, ${look.sat}%, 25%, 0.6)`);
      ctx.beginPath();
      ctx.arc(c, c, radius, 0, Math.PI * 2);
      ctx.fillStyle = core;
      ctx.fill();

      // Rotating arcs (the "reactor" rings)
      ctx.lineCap = "round";
      for (let i = 0; i < 3; i++) {
        const r = base * (1.45 + i * 0.32) + smooth * 10 * (i + 1);
        const start = angle * (i % 2 ? -1 : 1) * (1 + i * 0.4) + i;
        ctx.strokeStyle = `hsla(${look.hue}, ${look.sat}%, ${70 - i * 10}%, ${0.7 - i * 0.18})`;
        ctx.lineWidth = 2.5 - i * 0.5;
        for (let k = 0; k < 3; k++) {
          ctx.beginPath();
          ctx.arc(c, c, r, start + (k * Math.PI * 2) / 3, start + (k * Math.PI * 2) / 3 + 1.2);
          ctx.stroke();
        }
      }
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [level]);

  return <canvas ref={canvasRef} className={`orb orb-${state}`} role="img" aria-label={`JARVIS: ${state}`} />;
}
