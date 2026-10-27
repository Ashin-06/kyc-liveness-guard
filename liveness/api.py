"""HTTP service: issue challenges, verify recorded sessions.

    python -m liveness serve            # settings: LIVENESS_* environment variables
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import db, ops
from .audio import AudioAnalyzer, load_audio
from .challenge import ChallengeError, ChallengeService
from .pipeline import evaluate
from .rppg import analyze_traces, extract_video

MAX_UPLOAD = 40 * 1024 * 1024
ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("liveness.api")

settings = ops.load_settings()
ops.setup_logging(settings.log_level)
store = db.connect(settings.database_url)
challenges = ChallengeService(settings.secret["CHALLENGE_KEY"], engine=store,
                              previous_keys=settings.previous["CHALLENGE_KEY"])
analyzer = AudioAnalyzer(os.environ.get("LIVENESS_MODEL", ROOT / "models" / "voice_model.joblib"),
                         os.environ.get("LIVENESS_MODEL_16K", ROOT / "models" / "voice_model_asvspoof.joblib"))
app = FastAPI(title="KYC Liveness Guard", version="1.0.0")
ops.install(app, settings, {"database": lambda: store.connect().close(),
                            "voice_model_16k": lambda: analyzer.narrowband is not None})
log.info("voice engines", extra={"fields": {"wideband": "model+rules" if analyzer.model else "rules",
                                            "16k": "ASVspoof model+rules" if analyzer.narrowband else "same as wideband"}})


async def _read(f: UploadFile) -> bytes:
    data = await f.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "upload too large")
    return data


def _video_from_bytes(data: bytes, suffix: str, duration: float | None):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return extract_video(path, duration)
    except ValueError as e:
        raise HTTPException(422, str(e).replace(path, "upload")) from e  # no server paths in replies
    finally:
        os.unlink(path)


def _audio_from_bytes(data: bytes):
    try:
        return load_audio(data)
    except Exception:  # soundfile raises its own error types
        raise HTTPException(422, "unreadable audio") from None


def _suffix(name: str | None, default: str) -> str:
    """The upload's extension, used only to name the temporary file OpenCV reads."""
    s = Path(name or "").suffix.lower()
    return s if s in {".webm", ".mp4", ".mov", ".avi", ".mkv"} else default


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "capture.html")


@app.post("/v1/challenge")
def new_challenge(ident: dict = Depends(ops.require("verify"))):
    token, payload = challenges.issue()
    return {"token": token, "duration": payload["duration"], "cues": payload["cues"], "expires_at": payload["exp"]}


@app.post("/v1/verify")
async def verify(
    token: str = Form(..., max_length=4096),
    audio: UploadFile = File(...),
    video: UploadFile | None = File(None),
    duration: float | None = Form(None, gt=0, le=120),
    ident: dict = Depends(ops.require("verify")),
):
    try:
        payload = challenges.redeem(token)
    except ChallengeError as e:
        ops.DECISIONS.labels(decision="challenge_rejected").inc()
        raise HTTPException(401, f"challenge rejected: {e}") from e
    a = _audio_from_bytes(await _read(audio))
    v = None
    if video is not None:
        v = _video_from_bytes(await _read(video), _suffix(video.filename, ".webm"), duration)
    result = evaluate(a, v, payload, analyzer).as_dict()
    ops.DECISIONS.labels(decision=result["decision"]).inc()
    log.info("verified", extra={"fields": {"user": ident["user"], "challenge": payload["id"],
                                           "decision": result["decision"], "risk": result["risk"]}})
    return result


@app.post("/v1/analyze/audio")
async def analyze_audio(audio: UploadFile = File(...), ident: dict = Depends(ops.require("analyze"))):
    x, sr = _audio_from_bytes(await _read(audio))
    rep = analyzer.analyze(x, sr)
    return {"risk": round(rep.risk, 3), "engine": rep.source, "reasons": rep.reasons,
            "features": {k: v for k, v in rep.features.items() if not k.startswith("mfcc")}}


@app.post("/v1/analyze/video")
async def analyze_video(video: UploadFile = File(...), duration: float | None = Form(None, gt=0, le=120),
                        ident: dict = Depends(ops.require("analyze"))):
    v = _video_from_bytes(await _read(video), _suffix(video.filename, ".mp4"), duration)
    pr = analyze_traces(v.regions, v.fps, v.background)
    return {"risk": round(pr.risk, 3), "bpm": round(pr.bpm, 1), "snr_db": round(pr.snr_db, 2),
            "region_bpm": pr.region_bpm, "face_ratio": v.face_ratio, "fps": v.fps, "reasons": pr.reasons}
