"""Nybble web app: a gallery of MCP connectors plus a finetune entry point.

Run with:  uvicorn app.main:app --reload
"""

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.mcps import MCPS, MCPS_BY_ID

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Nybble")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# In-memory state. Resets when the server restarts; swap for a database later.
connections: dict[str, dict[str, str]] = {}
finetune_jobs: dict[str, dict] = {}


class ConnectRequest(BaseModel):
    fields: dict[str, str] = {}


class FinetuneRequest(BaseModel):
    sources: list[str]


def public_mcp(mcp: dict) -> dict:
    return {**mcp, "connected": mcp["id"] in connections}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"mcps": [public_mcp(m) for m in MCPS]}
    )


@app.get("/api/mcps")
def list_mcps():
    return [public_mcp(m) for m in MCPS]


@app.post("/api/mcps/{mcp_id}/connect")
def connect_mcp(mcp_id: str, body: ConnectRequest):
    mcp = MCPS_BY_ID.get(mcp_id)
    if mcp is None:
        raise HTTPException(404, "Unknown MCP")

    missing = [
        f["name"]
        for f in mcp.get("fields", [])
        if f.get("required", True) and not body.fields.get(f["name"], "").strip()
    ]
    if missing:
        raise HTTPException(422, f"Missing: {', '.join(missing)}")

    # Credentials are kept server-side only and never echoed back to the browser.
    connections[mcp_id] = body.fields
    return public_mcp(mcp)


@app.post("/api/mcps/{mcp_id}/disconnect")
def disconnect_mcp(mcp_id: str):
    mcp = MCPS_BY_ID.get(mcp_id)
    if mcp is None:
        raise HTTPException(404, "Unknown MCP")
    connections.pop(mcp_id, None)
    return public_mcp(mcp)


@app.post("/api/finetune")
def start_finetune(body: FinetuneRequest):
    if not body.sources:
        raise HTTPException(422, "Pick at least one connected source")
    not_connected = [s for s in body.sources if s not in connections]
    if not_connected:
        raise HTTPException(422, f"Not connected: {', '.join(not_connected)}")

    # Placeholder: records the job. Wire this to the real training pipeline.
    job = {"id": uuid4().hex[:8], "sources": body.sources, "status": "queued"}
    finetune_jobs[job["id"]] = job
    return job
