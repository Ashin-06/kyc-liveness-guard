"""HTTP service: issue challenges, verify recorded sessions.

    uvicorn liveness.api:app --port 8001
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .audio import AudioAnalyzer, load_audio
from .challenge import ChallengeError, ChallengeService
from .pipeline import evaluate
from .rppg import analyze_traces, extract_video

MAX_UPLOAD = 40 * 1024 * 1024
ROOT = Path(__file__).resolve().parent.parent

secret = os.environ.get("LIVENESS_SECRET")
challenges = ChallengeService(bytes.fromhex(secret) if secret else None)
analyzer = AudioAnalyzer(os.environ.get("LIVENESS_MODEL", ROOT / "models" / "voice_model.joblib"),
                         os.environ.get("LIVENESS_MODEL_16K", ROOT / "models" / "voice_model_asvspoof.joblib"))
app = FastAPI(title="KYC Liveness Guard", version="1.0")


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
        raise HTTPException(422, str(e)) from e
    finally:
        os.unlink(path)


def _audio_from_bytes(data: bytes):
    try:
        return load_audio(data)
    except Exception as e:  # soundfile raises its own error types
        raise HTTPException(422, f"unreadable audio: {e}") from e


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "capture.html")


@app.get("/healthz")
def health():
    return {"ok": True, "voice_engine": "model+rules" if analyzer.model else "rules",
            "voice_engine_16k": "ASVspoof model+rules" if analyzer.narrowband else "same as above"}


@app.post("/v1/challenge")
def new_challenge():
    token, payload = challenges.issue()
    return {"token": token, "duration": payload["duration"], "cues": payload["cues"], "expires_at": payload["exp"]}


@app.post("/v1/verify")
async def verify(
    token: str = Form(...),
    audio: UploadFile = File(...),
    video: UploadFile | None = File(None),
    duration: float | None = Form(None),
):
    try:
        payload = challenges.redeem(token)
    except ChallengeError as e:
        raise HTTPException(401, f"challenge rejected: {e}") from e
    a = _audio_from_bytes(await _read(audio))
    v = None
    if video is not None:
        v = _video_from_bytes(await _read(video), Path(video.filename or "v.webm").suffix or ".webm", duration)
    return evaluate(a, v, payload, analyzer).as_dict()


@app.post("/v1/analyze/audio")
async def analyze_audio(audio: UploadFile = File(...)):
    x, sr = _audio_from_bytes(await _read(audio))
    rep = analyzer.analyze(x, sr)
    return {"risk": round(rep.risk, 3), "engine": rep.source, "reasons": rep.reasons,
            "features": {k: v for k, v in rep.features.items() if not k.startswith("mfcc")}}


@app.post("/v1/analyze/video")
async def analyze_video(video: UploadFile = File(...), duration: float | None = Form(None)):
    v = _video_from_bytes(await _read(video), Path(video.filename or "v.mp4").suffix or ".mp4", duration)
    pr = analyze_traces(v.regions, v.fps)
    return {"risk": round(pr.risk, 3), "bpm": round(pr.bpm, 1), "snr_db": round(pr.snr_db, 2),
            "region_bpm": pr.region_bpm, "face_ratio": v.face_ratio, "fps": v.fps, "reasons": pr.reasons}
