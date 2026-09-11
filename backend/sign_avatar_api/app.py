"""Local browser application foundation; no model inference is exposed yet."""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
SIGNS = json.loads((ROOT / "shared" / "signs.json").read_text())
app = FastAPI(title="Sign Avatar", version="0.2.0")


@app.get("/api/health")
def health():
    return {
        "service": "sign-avatar",
        "version": "0.2.0",
        "language": "NSL",
        "capabilities": {
            "sign_catalogue": True,
            "recognition": False,
            "dataset_import": False,
            "training": False,
            "sign_playback": False,
        },
    }


@app.get("/api/signs")
def list_signs(q: str = Query(default="", max_length=100), category: str | None = None):
    if category not in (None, "letter", "digit", "word"):
        raise HTTPException(status_code=422, detail="Unknown label category")
    results = [
        sign for sign in SIGNS
        if (category is None or sign["category"] == category)
        and (q.lower() in sign["label"].lower() or q.lower() in sign["id"])
    ]
    return {"items": results, "total": len(results)}


@app.get("/api/signs/{sign_id}")
def get_sign(sign_id: str):
    for sign in SIGNS:
        if sign["id"] == sign_id:
            return sign
    raise HTTPException(status_code=404, detail="Sign not found")


# API routes remain registered first; unknown /api routes must not fall through.
@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
def unknown_api(path: str):
    raise HTTPException(status_code=404, detail="API route not available")


# The built UI is served from the same origin as the API in local release mode.
if (ROOT / "web" / "dist" / "index.html").is_file():
    app.mount("/", StaticFiles(directory=ROOT / "web" / "dist", html=True), name="web")
