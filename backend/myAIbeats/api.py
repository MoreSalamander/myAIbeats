"""FastAPI app for my-AI-beats — streams the pipeline as NDJSON.

Same worker-thread + queue → StreamingResponse pattern as the rest of
the my-AI suite. Same shared event vocabulary, now driving a flashy
energy-arc music UI.
"""
from __future__ import annotations

import json
import queue
import threading
from dataclasses import replace
from pathlib import Path
from typing import Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from myAIbeats.events import EventEmitter
from myAIbeats.spec import SpecError, load_spec, parse_spec

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
SPECS_DIR    = Path(__file__).resolve().parent.parent.parent / "specs"
OUT_DIR      = Path(__file__).resolve().parent.parent.parent / "out"

app = FastAPI(title="my-AI-beats",
              description="A MoreSalamander StudioLabs production. SongSpec in, song out.")

# Single-flight guard: MusicGen runs on the one MPS device. Two concurrent
# generations contend and DEADLOCK the device (0% CPU hang, unrecoverable
# without a kill). Only one render may be in flight at a time.
_GEN_LOCK = threading.Lock()


class GenerateRequest(BaseModel):
    spec_name: str
    limit: Optional[int] = None
    vocals: bool = False
    quality: str = "final"     # "draft" (small, ~3× faster) | "final" (medium)


QUALITY_MODELS = {
    "draft": "facebook/musicgen-stereo-small",
    "final": "facebook/musicgen-stereo-medium",
}


# ---- streaming generate -------------------------------------------------

def _ndjson_generate(req: GenerateRequest) -> Iterator[str]:
    # Reject a second render while one is already running — prevents the
    # MPS device deadlock that comes from two concurrent MusicGen runs.
    if not _GEN_LOCK.acquire(blocking=False):
        yield json.dumps({"event": "error", "stage": "server",
                          "message": "A render is already in progress. "
                          "Wait for it to finish before starting another — "
                          "two MusicGen runs would deadlock the MPS device."}) + "\n"
        return

    try:
        yield from _generate_locked(req)
    finally:
        _GEN_LOCK.release()


def _generate_locked(req: GenerateRequest) -> Iterator[str]:
    spec_path = SPECS_DIR / f"{req.spec_name}.json"
    if not spec_path.exists():
        yield json.dumps({"event": "error", "stage": "server",
                          "message": f"spec not found: {req.spec_name}"}) + "\n"
        return
    try:
        spec = load_spec(spec_path)
    except SpecError as e:
        yield json.dumps({"event": "error", "stage": "spec_load",
                          "message": str(e)}) + "\n"
        return

    # override vocals enable from the UI toggle
    if req.vocals != spec.vocals.enabled:
        spec = replace(spec, vocals=replace(spec.vocals, enabled=req.vocals))

    out_dir = OUT_DIR / req.spec_name
    out_dir.mkdir(parents=True, exist_ok=True)
    song_path = str(out_dir / f"{req.spec_name}.wav")

    q: queue.Queue[dict | None] = queue.Queue()
    em = EventEmitter(out=None, sink=q.put)

    def _worker():
        try:
            from myAIbeats.local import LocalRenderer, MusicGenSynth
            from myAIbeats.pipeline import run
            model_id = QUALITY_MODELS.get(req.quality, QUALITY_MODELS["final"])
            em.emit("step_start", "model_load", quality=req.quality, model=model_id)
            r = LocalRenderer(out_dir=out_dir, synth=MusicGenSynth(model_id=model_id))
            em.emit("step_complete", "model_load", quality=req.quality)
            run(spec, r, em, out_path=song_path, limit=req.limit)
        except Exception as exc:
            em.error("server", message=f"{type(exc).__name__}: {exc}")
        finally:
            q.put(None)

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        ev = q.get()
        if ev is None:
            break
        yield json.dumps(ev, ensure_ascii=False) + "\n"


@app.post("/api/generate")
def generate(req: GenerateRequest) -> StreamingResponse:
    return StreamingResponse(_ndjson_generate(req), media_type="application/x-ndjson")


# ---- spec endpoints -----------------------------------------------------

@app.get("/api/specs")
def list_specs() -> list[dict]:
    out = []
    for p in sorted(SPECS_DIR.glob("*.json")):
        try:
            spec = load_spec(p)
            total_s = sum(s.duration_at_tempo(spec.song.tempo) for s in spec.sections)
            out.append({
                "name": p.stem,
                "title": spec.song.title,
                "genre": spec.song.genre,
                "tempo": spec.song.tempo,
                "key": spec.song.key,
                "mood": spec.song.mood,
                "sections": len(spec.sections),
                "duration_s": round(total_s, 1),
            })
        except Exception:
            pass
    return out


@app.get("/api/specs/{name}")
def get_spec(name: str) -> dict:
    p = SPECS_DIR / f"{name}.json"
    if not p.exists():
        raise HTTPException(404, "spec not found")
    spec = load_spec(p)
    # Return a UI-friendly view including per-section energy + duration
    return {
        "song": {
            "title": spec.song.title, "genre": spec.song.genre,
            "tempo": spec.song.tempo, "key": spec.song.key,
            "mood": spec.song.mood, "reference_feel": spec.song.reference_feel,
        },
        "sections": [
            {
                "id": s.id, "type": s.type, "bars": s.bars,
                "energy": s.energy, "prompt": s.prompt,
                "duration_s": round(s.duration_at_tempo(spec.song.tempo), 1),
                "continuation": s.continuation,
            }
            for s in spec.sections
        ],
    }


# ---- output serving -----------------------------------------------------

@app.get("/api/output/{spec_name}")
def list_outputs(spec_name: str) -> dict:
    d = OUT_DIR / spec_name
    if not d.exists():
        return {"files": []}
    files = sorted(str(p.name) for p in d.iterdir() if p.suffix == ".wav")
    return {"files": files}


@app.get("/api/output/{spec_name}/{filename}")
def get_output(spec_name: str, filename: str) -> FileResponse:
    path = OUT_DIR / spec_name / filename
    if not path.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(str(path), media_type="audio/wav")


# ---- frontend -----------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = FRONTEND_DIR / "index.html"
    if not html.exists():
        return HTMLResponse("<h1>my-AI-beats</h1><p>frontend not found</p>")
    return HTMLResponse(html.read_text(encoding="utf-8"))
