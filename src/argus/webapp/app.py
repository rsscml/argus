"""FastAPI application: `argus-web` starts it, `create_app()` builds it.

Pages:  /            research desk (briefs, deep research, recent runs)
        /admin       sources, domains, candidate queue, settings
        /maintenance health, schedules, runs & replay, snapshots, data actions

Auth: optional. Set a web password on the settings page (or ARGUS_WEB_PASSWORD
in the environment) and every /api call requires sign-in; the pages show a
sign-in overlay. Sessions are cookie-based and invalidated on server restart.
This is a single-operator LAN gate, not internet-grade auth — put a real
reverse proxy in front for anything exposed.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from argus.webapp.api import router as api_router
from argus.webapp.jobs import JobManager
from argus.webapp.scheduler import SchedulerService
from argus.webapp.state import AppState

STATIC_DIR = Path(__file__).parent / "static"
_SESSION_NONCE = secrets.token_hex(16)  # restart invalidates sessions
_OPEN_PATHS = {"/api/login", "/api/authstate"}


class LoginBody(BaseModel):
    # module-level: with postponed annotations (PEP 563) FastAPI resolves
    # endpoint annotations against module globals, so a class local to
    # create_app() would silently degrade to an untyped query parameter.
    password: str


def _password() -> str:
    return os.environ.get("ARGUS_WEB_PASSWORD", "")


def _session_token(password: str) -> str:
    return hashlib.sha256(f"{_SESSION_NONCE}:{password}".encode()).hexdigest()


def _authorized(request: Request) -> bool:
    password = _password()
    if not password:
        return True
    cookie = request.cookies.get("argus_session", "")
    if cookie and hmac.compare_digest(cookie, _session_token(password)):
        return True
    bearer = request.headers.get("authorization", "")
    if bearer.startswith("Bearer ") and hmac.compare_digest(
            bearer.removeprefix("Bearer "), password):
        return True
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    state: AppState = app.state.argus
    state.scheduler.start()
    yield
    state.scheduler.shutdown()
    state.jobs.shutdown()


def create_app() -> FastAPI:
    state = AppState()
    state.jobs = JobManager()
    state.scheduler = SchedulerService(state)

    app = FastAPI(title="Argus", docs_url="/api/docs",
                  openapi_url="/api/openapi.json", lifespan=lifespan)
    app.state.argus = state

    @app.middleware("http")
    async def auth_gate(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api") and path not in _OPEN_PATHS:
            if not _authorized(request):
                return JSONResponse({"detail": "sign in required"},
                                    status_code=401)
        return await call_next(request)

    @app.post("/api/login")
    def login(body: LoginBody):
        password = _password()
        if not password:
            return {"ok": True, "required": False}
        if not hmac.compare_digest(body.password, password):
            time.sleep(0.4)  # blunt online brute force on the LAN gate
            return JSONResponse({"detail": "wrong password"}, status_code=401)
        response = JSONResponse({"ok": True, "required": True})
        response.set_cookie(
            "argus_session", _session_token(password),
            httponly=True, samesite="lax", max_age=12 * 3600,
        )
        return response

    @app.get("/api/authstate")
    def authstate(request: Request):
        return {"required": bool(_password()), "ok": _authorized(request)}

    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/admin", include_in_schema=False)
    def admin():
        return FileResponse(STATIC_DIR / "admin.html")

    @app.get("/maintenance", include_in_schema=False)
    def maintenance():
        return FileResponse(STATIC_DIR / "maintenance.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()


def main() -> None:
    host = os.environ.get("ARGUS_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("ARGUS_WEB_PORT", "8765"))
    uvicorn.run("argus.webapp.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
