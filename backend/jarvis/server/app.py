"""FastAPI app: REST for status, WebSocket for the live conversation.

Security for a server that lives on your machine:
- it only listens on 127.0.0.1 (see ``jarvis serve``);
- browsers send an ``Origin`` header: only the JARVIS pages themselves are accepted,
  so another website open in your browser cannot drive JARVIS through localhost;
- no secret ever goes to the browser (agent status never includes keys);
- message sizes are capped.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from jarvis.config import PROJECT_ROOT
from jarvis.server.connection import Connection, agents_event
from jarvis.server.runtime import JarvisRuntime

DEFAULT_PORT = 8300
DEV_WEB_PORT = 5300
WEB_DIST = PROJECT_ROOT / "web" / "dist"


def default_origins(port: int = DEFAULT_PORT) -> set[str]:
    return {
        f"http://{host}:{p}" for host in ("127.0.0.1", "localhost") for p in (port, DEV_WEB_PORT)
    }


def create_app(
    runtime: JarvisRuntime | None = None,
    *,
    allowed_origins: Iterable[str] | None = None,
    web_dist: Path | None = WEB_DIST,
) -> FastAPI:
    origins = set(allowed_origins or default_origins())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime or JarvisRuntime.from_settings()
        await app.state.runtime.start()
        try:
            yield
        finally:
            await app.state.runtime.stop()

    app = FastAPI(title="JARVIS", lifespan=lifespan, docs_url=None, redoc_url=None)

    def origin_ok(origin: str | None) -> bool:
        # Non-browser clients (CLI, tests) send no Origin; browsers always do.
        return origin is None or origin in origins

    @app.middleware("http")
    async def check_origin(request: Request, call_next):
        if not origin_ok(request.headers.get("origin")):
            return JSONResponse({"detail": "origin not allowed"}, status_code=403)
        return await call_next(request)

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True}

    @app.get("/api/status")
    async def status(request: Request) -> dict:
        rt: JarvisRuntime = request.app.state.runtime
        event = agents_event(rt)
        return {
            "cost_mode": event["cost_mode"],
            "agents": event["agents"],
            "voice": {"ready": rt.voice_ready, "tts_engine": rt.settings.voice.tts_engine},
        }

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        if not origin_ok(websocket.headers.get("origin")):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await Connection(websocket, websocket.app.state.runtime).run()

    if web_dist is not None and (web_dist / "index.html").is_file():
        # The built interface (npm run build). In development, Vite serves it instead.
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

    return app
