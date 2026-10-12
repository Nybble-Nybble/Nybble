"""Nybble web app: a gallery of MCP connectors plus a finetune entry point.

Run with:  python -m app.main
(or uvicorn app.main:app --reload --host 127.0.0.1). It must only ever listen on this machine.
"""

import ipaddress
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app import data_view
from app.mcps import MCPS, MCPS_BY_ID
from app.sources import SOURCES, SourceError, check, delete, extract, has_data, origin

BASE_DIR = Path(__file__).parent

log = logging.getLogger("nybble")

# The server reads your messages, so it only answers this machine:
#  - it binds to 127.0.0.1 (bottom of this file), and refuses any client that isn't this machine anyway,
#    in case it's ever started with --host 0.0.0.0;
#  - the Host check stops DNS rebinding, where a website points its own domain at 127.0.0.1 to get its
#    scripts past the browser's same-origin rule;
#  - POSTs carrying another website's Origin are refused (see local_only).
LOCAL_HOSTS = ["127.0.0.1", "localhost"]

app = FastAPI(title="Nybble")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOCAL_HOSTS)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.middleware("http")
async def revalidate_static(request: Request, call_next):
    """Make browsers re-check CSS/JS on every load so edits show up without a hard refresh."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

def is_loopback(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (getattr(addr, "ipv4_mapped", None) or addr).is_loopback  # ::ffff:127.0.0.1 counts too


@app.middleware("http")
async def local_only(request: Request, call_next):
    """Answer only this machine, and refuse POSTs sent by other websites (a plain form can POST here without CORS)."""
    if request.client and not is_loopback(request.client.host):
        return JSONResponse({"detail": "Nybble only accepts connections from this Mac"}, status_code=403)
    page = request.headers.get("origin")
    if request.method != "GET" and page and urlsplit(page).hostname not in LOCAL_HOSTS:
        return JSONResponse({"detail": "Requests from other websites aren't allowed"}, status_code=403)
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, err: RequestValidationError):
    """A malformed request (wrong type in the URL or body): one readable line instead of pydantic's error list."""
    return JSONResponse({"detail": "That request wasn't in the form Nybble expects."}, status_code=422)


@app.exception_handler(Exception)
async def unexpected_error(request: Request, err: Exception):
    """Anything not turned into a clear message: log the details here, show the browser something readable."""
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Something went wrong on the Nybble server. The details are in its terminal."},
                        status_code=500)

# Flip to True once training is actually implemented. While False the Finetune button is disabled
# and /api/finetune refuses jobs, so nobody queues a job that would never run.
FINETUNE_ENABLED = False

# In-memory state. Resets when the server restarts; swap for a database later.
connections: dict[str, dict[str, str]] = {}
finetune_jobs: dict[str, dict] = {}
# One lock per message source, held for the whole of a Connect, Update data or Disconnect. A second
# extraction while one runs is refused (see extracting). A Disconnect waits for it to finish and then
# deletes, instead of the extraction landing afterwards and bringing the data back.
source_locks = {s: threading.Lock() for s in SOURCES}


class Busy(SourceError):
    """The source is already being extracted."""


@contextmanager
def extracting(mcp_id: str):
    lock = source_locks[mcp_id]
    if not lock.acquire(blocking=False):
        raise Busy(f"{MCPS_BY_ID[mcp_id]['name']} is already being extracted. Try again once that finishes.")
    try:
        yield
    finally:
        lock.release()


def as_http(err: SourceError) -> HTTPException:
    return HTTPException(409 if isinstance(err, Busy) else 422, str(err))


class ConnectRequest(BaseModel):
    fields: dict[str, str] = {}


class FinetuneRequest(BaseModel):
    sources: list[str]


def public_mcp(mcp: dict) -> dict:
    # Only connectors backed by a real extractor can be connected; the rest are placeholders for now.
    return {**mcp, "available": mcp["id"] in SOURCES, "connected": is_connected(mcp["id"])}


def is_connected(mcp_id: str) -> bool:
    """Connected this session, or a message source whose extracted data is still in data/.

    The data file counts because connections are in memory: after a restart the data is still there,
    and the card must still offer Disconnect so it can be deleted.
    """
    return mcp_id in connections or (mcp_id in SOURCES and has_data(mcp_id))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html",
        {"mcps": [public_mcp(m) for m in MCPS], "finetune_enabled": FINETUNE_ENABLED},
    )


def needs_input(mcp: dict) -> bool:
    return any(f.get("required", True) for f in mcp.get("fields", []))


@app.post("/api/mcps/connect-all")
def connect_all():
    """Connect every available MCP that needs no user input.

    MCPs that aren't available yet are left alone. Ones with required fields are skipped and reported back.
    Local message sources are only connected if their database can be read, and are extracted right away.
    """
    skipped, failed = [], []
    for mcp in MCPS:
        if mcp["id"] not in SOURCES or is_connected(mcp["id"]):
            continue
        if needs_input(mcp):
            skipped.append(mcp["name"])
        else:
            try:
                with extracting(mcp["id"]):
                    db_path = check(mcp["id"])["db_path"]
                    run_extract(mcp["id"], db_path)
                    connections[mcp["id"]] = {"db_path": db_path}
            except SourceError as err:
                failed.append({"id": mcp["id"], "name": mcp["name"], "error": str(err)})
    return {"mcps": [public_mcp(m) for m in MCPS], "skipped": skipped, "failed": failed}


@app.post("/api/mcps/{mcp_id}/connect")
def connect_mcp(mcp_id: str, body: ConnectRequest):
    mcp = MCPS_BY_ID.get(mcp_id)
    if mcp is None:
        raise HTTPException(404, "Unknown MCP")
    if mcp_id not in SOURCES:
        raise HTTPException(422, f"{mcp['name']} isn't available yet")

    missing = [
        f["label"]
        for f in mcp.get("fields", [])
        if f.get("required", True) and not body.fields.get(f["name"], "").strip()
    ]
    if missing:
        raise HTTPException(422, f"Missing: {', '.join(missing)}")

    # Connecting a message source extracts it straight away (about a second for 180k messages),
    # so problems show up in the connect popup and the data is ready on the Your data page.
    try:
        with extracting(mcp_id):
            db_path = check(mcp_id, body.fields.get("db_path", ""))["db_path"]
            result = run_extract(mcp_id, db_path)
            connections[mcp_id] = {"db_path": db_path}
    except SourceError as err:
        raise as_http(err)
    return {**public_mcp(mcp), "detail": extracted(result)}


def run_extract(mcp_id: str, db_path: str) -> dict:
    """Extract a source, turning any failure into a SourceError. The real error goes to the server log."""
    try:
        return extract(mcp_id, db_path)
    except SourceError:
        raise
    except Exception:  # the database opened but the extractor choked on it
        log.exception("Extracting %s from %s failed", mcp_id, db_path)
        raise SourceError(
            f"Nybble opened the {MCPS_BY_ID[mcp_id]['name']} database but couldn't read its messages. "
            "It may be from a version of the app Nybble doesn't support yet."
        )


def extracted(result: dict) -> str:
    return f"{result['messages']:,} messages in {result['sessions']:,} sessions saved"


@app.post("/api/mcps/{mcp_id}/extract")
def extract_mcp(mcp_id: str):
    """Re-run a connected source's extractor to pick up messages that arrived since it was connected."""
    mcp = MCPS_BY_ID.get(mcp_id)
    if mcp is None:
        raise HTTPException(404, "Unknown MCP")
    if mcp_id not in SOURCES:
        raise HTTPException(422, f"{mcp['name']} isn't available yet")
    try:
        with extracting(mcp_id):
            # Checked under the lock, so a Disconnect that just finished is seen.
            if not is_connected(mcp_id):
                raise SourceError(f"{mcp['name']} isn't connected")
            # After a restart the in-memory connection is gone, but the data records which database it
            # came from, so a custom path is kept. Only data from before that was recorded falls back
            # to the default location, and the reply says so.
            db_path = connections.get(mcp_id, {}).get("db_path") or origin(mcp_id)
            result = run_extract(mcp_id, db_path or "")
    except SourceError as err:
        raise as_http(err)
    detail = extracted(result)
    if not db_path:
        detail += " (read from the default location)"
    return {**public_mcp(mcp), "detail": detail}


@app.post("/api/mcps/{mcp_id}/disconnect")
def disconnect_mcp(mcp_id: str):
    mcp = MCPS_BY_ID.get(mcp_id)
    if mcp is None:
        raise HTTPException(404, "Unknown MCP")
    if mcp_id not in SOURCES:
        connections.pop(mcp_id, None)
        return public_mcp(mcp)
    # Waits for any extraction of this source still running, so its data can't reappear afterwards.
    with source_locks[mcp_id]:
        connections.pop(mcp_id, None)
        # Disconnecting throws away what was extracted, even if the connection was already lost to a restart.
        delete(mcp_id)
    return public_mcp(mcp)


@app.post("/api/finetune")
def start_finetune(body: FinetuneRequest):
    if not FINETUNE_ENABLED:
        raise HTTPException(501, "Finetuning isn't available yet")
    if not body.sources:
        raise HTTPException(422, "Pick at least one connected source")
    not_connected = [s for s in body.sources if not is_connected(s)]
    if not_connected:
        names = [MCPS_BY_ID[s]["name"] if s in MCPS_BY_ID else s for s in not_connected]
        raise HTTPException(422, f"Not connected: {', '.join(names)}")

    # Sources were already extracted when they were connected; training itself is still a placeholder.
    job = {"id": uuid4().hex[:8], "sources": body.sources, "status": "queued"}
    finetune_jobs[job["id"]] = job
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
def data_chat(source_id: str, chat_id: int,
              start: Optional[int] = Query(None, ge=0), end: Optional[int] = Query(None, ge=0)):
    """A window of a chat's sessions, [start, end); by default the newest ones. See data_view.chat."""
    c = data_view.chat(source_id, chat_id, start, end)
    if c is None:
        raise HTTPException(404, "Unknown chat")
    return c


if __name__ == "__main__":
    import uvicorn

    # Explicitly this machine only, never 0.0.0.0.
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
