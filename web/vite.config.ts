/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The JARVIS server (python -m jarvis serve) listens on 127.0.0.1:8300.
const API = "http://127.0.0.1:8300";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5300,
    strictPort: true,
    proxy: {
      "/api": { target: API },
      "/ws": { target: API, ws: true },
    },
  },
  test: { environment: "node" },
});
