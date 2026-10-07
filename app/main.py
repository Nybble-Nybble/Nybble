"""Nybble web app: a gallery of MCP connectors plus a finetune entry point.

Run with:  uvicorn app.main:app --reload
"""

from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app import data_view
from app.mcps import MCPS, MCPS_BY_ID
from app.sources import SOURCES, SourceError, check, extract

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Nybble")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.middleware("http")
async def revalidate_static(request: Request, call_next):
    """Make browsers re-check CSS/JS on every load so edits show up without a hard refresh."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

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


def needs_input(mcp: dict) -> bool:
    return any(f.get("required", True) for f in mcp.get("fields", []))


@app.post("/api/mcps/connect-all")
def connect_all():
    """Connect every MCP that needs no user input (OAuth, or only optional fields).

    MCPs with required fields (API keys, paths) are skipped and reported back.
    Local message sources are only connected if their database can actually be read.
    """
    skipped, failed = [], []
    for mcp in MCPS:
        if mcp["id"] in connections:
            continue
        if needs_input(mcp):
            skipped.append(mcp["name"])
        elif mcp["id"] in SOURCES:
            try:
                connections[mcp["id"]] = {"db_path": check(mcp["id"])["db_path"]}
            except SourceError as err:
                failed.append({"name": mcp["name"], "error": str(err)})
        else:
            connections[mcp["id"]] = {}
    return {"mcps": [public_mcp(m) for m in MCPS], "skipped": skipped, "failed": failed}


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

    if mcp_id in SOURCES:
        # Prove the database is readable now, so problems show up in the connect popup, not at finetune time.
        try:
            info = check(mcp_id, body.fields.get("db_path", ""))
        except SourceError as err:
            raise HTTPException(422, str(err))
        connections[mcp_id] = {"db_path": info["db_path"]}
        return {**public_mcp(mcp), "detail": f"{info['messages']:,} messages found"}

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


def run_extractors(job: dict) -> None:
    """Turn each local message source into a sessions file under data/, then mark the job ready to train."""
    for source_id in job["sources"]:
        if source_id not in SOURCES:
            continue
        try:
            job["results"][source_id] = extract(source_id, connections.get(source_id, {}).get("db_path", ""))
        except Exception as err:  # one broken source shouldn't sink the others
            job["results"][source_id] = {"error": str(err)}
    job["status"] = "queued"


@app.post("/api/finetune")
def start_finetune(body: FinetuneRequest, background: BackgroundTasks):
    if not body.sources:
        raise HTTPException(422, "Pick at least one connected source")
    not_connected = [s for s in body.sources if s not in connections]
    if not_connected:
        raise HTTPException(422, f"Not connected: {', '.join(not_connected)}")

    # Message sources are extracted first; training itself is still a placeholder.
    extracting = any(s in SOURCES for s in body.sources)
    job = {
        "id": uuid4().hex[:8],
        "sources": body.sources,
        "status": "extracting" if extracting else "queued",
        "results": {},
    }
    finetune_jobs[job["id"]] = job
    if extracting:
        background.add_task(run_extractors, job)
    return job


@app.get("/api/finetune/{job_id}")
def finetune_status(job_id: str):
    job = finetune_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


# ---- Your data: browse what the extractors wrote (local only) ----


@app.get("/data", response_class=HTMLResponse)
def data_page(request: Request):
    return templates.TemplateResponse(request, "data.html", {})


@app.get("/api/data")
def data_summary():
    return {"chats": data_view.summary()}


@app.get("/api/data/search")
def data_search(q: str = ""):
    return {"hits": data_view.search(q)}


@app.get("/api/data/{source_id}/{chat_id}")
def data_chat(source_id: str, chat_id: int):
    c = data_view.chat(source_id, chat_id)
    if c is None:
        raise HTTPException(404, "Unknown chat")
    return c
