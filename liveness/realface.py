"""Pulse check on real faces.

MCD-rPPG (Egorov et al., ACM MM 2025; CC BY 4.0): 3-minute webcam recordings of 600 people at
rest and after exercise, with a finger PPG synchronised to every frame.
https://huggingface.co/datasets/kyegorov/mcd_rppg. Used here for pulse estimation only, which
is what the dataset is published for; nothing identifies anyone and no frames are kept.

From each frontal-webcam recording, two 15-second windows (about the length of a KYC capture)
are scored as live, with the heart rate from the PPG as ground truth. Three photo attacks are
built from a frame of the same recording, as a camera pointed at a print would see it:
  photo          the still frame with sensor noise and slow exposure drift
  photo_flicker  plus a lamp flickering at 72 bpm (equal in all colours)
  photo_green    plus a green-tinted flicker at 72 bpm: an attacker who knows the method

The off-face rhythm check (BG_CORR_MAX in rppg.py) was set on the first half of the subjects;
results are reported on the other half.

GRID (Cooke et al., 2006; CC BY 4.0, https://zenodo.org/records/3625687): each clip's mouth
movement is scored against its own audio, against another sentence by the same speaker (a dub
or separate audio over a real face), and against its own audio 0.5 s late (a slow voice changer).

    python -m liveness realface --mcd mcd_rppg/ [--subjects 20] [--cache traces.pkl] [--grid grid/]
"""
from __future__ import annotations

import csv
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import rppg
from .lipsync import SYNC_MIN_CORR, envelope_per_frame, sync_score
from .pipeline import PASS_BELOW, REJECT_ABOVE
from .rppg import analyze_traces, bandpass, extract_frames, extract_video, pulse_snr

CLIP_S = 15
STARTS_S = (20, 100)
ATTACKS = ("photo", "photo_flicker", "photo_green")
FLICKER_HZ = 1.2


def _status(risk: float) -> str:
    return "pass" if risk < PASS_BELOW else ("warn" if risk < REJECT_ABOVE else "fail")


def _attack_frames(base: np.ndarray, n: int, fps: float, kind: str, rng: np.random.Generator):
    # a bank of noise frames drawn at random, so the noise has no period of its own
    bank = np.clip(np.round(rng.normal(0, 2.0, size=(16, *base.shape))), -127, 127).astype(np.int8)
    gain = 1.0
    for i in range(n):
        gain = float(np.clip(gain + rng.normal(0, 5e-4), 0.97, 1.03))
        f = base.astype(np.float32) * gain + bank[rng.integers(len(bank))]
        wave = np.sin(2 * np.pi * FLICKER_HZ * i / fps)
        if kind == "photo_flicker":
            f *= 1 + 0.01 * wave
        elif kind == "photo_green":
            f[:, :, 1] += 1.5 * wave  # BGR: green channel only
        yield np.clip(f, 0, 255).astype(np.uint8)


def _clip(kind: str, v, truth=None) -> dict:
    return {"kind": kind, "fps": v.fps, "regions": v.regions, "background": v.background, "face_ratio": v.face_ratio, "truth_bpm": truth}


def trace_recording(video: str, ppg_sync: str, seed: int) -> list[dict]:
    """Colour traces for the live windows and the photo attacks of one recording."""
    import cv2

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(CLIP_S * fps)
    for _ in range(int(STARTS_S[0] * fps)):
        cap.grab()
    ok, base = cap.read()
    cap.release()
    ppg = np.loadtxt(ppg_sync, usecols=0)
    clips = []
    for start in STARTS_S:
        s = int(start * fps)
        v = extract_video(video, max_frames=n, start_frame=s)
        seg = ppg[s : s + n]
        truth = pulse_snr(bandpass(seg - seg.mean(), v.fps), v.fps)[0] if len(seg) > fps * 5 else None
        clips.append(_clip("live", v, truth))
    if ok:
        rng = np.random.default_rng(seed)
        clips += [_clip(kind, extract_frames(_attack_frames(base, n, fps, kind, rng), fps)) for kind in ATTACKS]
    return clips


def _job(args):
    pid, *rest = args
    return [{"subject": pid, **c} for c in trace_recording(*rest)]


def _score(clips: list[dict], background: bool) -> list[dict]:
    return [{**c, "report": analyze_traces(c["regions"], c["fps"], c["background"] if background else None)}
            for c in clips if c["face_ratio"] >= 0.5]


def summarise(clips: list[dict], background: bool = True) -> dict:
    scored = _score(clips, background)
    live = [c for c in scored if c["kind"] == "live"]
    live_risk = [c["report"].risk for c in live]
    out = {"live_clips": len(live),
           "live_pulse_check": {s: round(float(np.mean([_status(r) == s for r in live_risk])), 3) for s in ("pass", "warn", "fail")}}
    for kind in ATTACKS:
        att = [c["report"].risk for c in scored if c["kind"] == kind]
        out[kind] = {"clips": len(att),
                     "pulse_check": {s: round(float(np.mean([_status(r) == s for r in att])), 3) for s in ("pass", "warn", "fail")},
                     "auc_live_vs_attack": round(float(roc_auc_score([0] * len(live_risk) + [1] * len(att), live_risk + att)), 3)}
    return out


def mcd(folder: str | Path, subjects: int = 20, cache: str | Path | None = None) -> dict:
    folder = Path(folder)
    if cache and Path(cache).exists():
        # a cache this command wrote itself on an earlier run
        clips = pickle.loads(Path(cache).read_bytes())  # nosec B301
    else:
        rows = [r for r in csv.DictReader(open(folder / "db.csv", encoding="utf-8")) if r["camera"] == "FullHDwebcam"]
        keep = sorted({r["patient_id"] for r in rows if (folder / r["video"]).exists()})[:subjects]
        jobs = [(r["patient_id"], str(folder / r["video"]), str(folder / r["ppg_sync"]), int(r["patient_id"]) * 2 + (r["step"] == "after"))
                for r in rows if r["patient_id"] in keep]
        with ProcessPoolExecutor(min(2, os.cpu_count() or 1)) as pool:  # each worker decodes video; memory, not CPU, is the limit
            clips = [c for rec in pool.map(_job, jobs) for c in rec]
        if cache:
            Path(cache).write_bytes(pickle.dumps(clips))

    ids = sorted({c["subject"] for c in clips})
    tune, test = set(ids[: len(ids) // 2]), set(ids[len(ids) // 2 :])
    live = _score([c for c in clips if c["kind"] == "live"], True)
    err = np.array([abs(c["report"].bpm - c["truth_bpm"]) for c in live if c["truth_bpm"]])
    tune_corr = [c["report"].background_corr for c in live if c["subject"] in tune]
    test_clips = [c for c in clips if c["subject"] in test]
    return {
        "dataset": "MCD-rPPG, frontal webcam (640x480, 30 fps), rest and after exercise",
        "subjects": len(ids), "clip_seconds": CLIP_S,
        "live_clips_without_face": sum(c["face_ratio"] < 0.5 for c in clips if c["kind"] == "live"),
        "heart_rate_vs_finger_ppg": {"clips": int(len(err)), "mae_bpm": round(float(err.mean()), 1),
                                     "median_abs_error_bpm": round(float(np.median(err)), 1),
                                     "within_5_bpm": round(float((err <= 5).mean()), 3), "within_10_bpm": round(float((err <= 10).mean()), 3)},
        "live_snr_db_median": round(float(np.median([c["report"].snr_db for c in live])), 2),
        "background_check": {"tuning_subjects": len(tune), "test_subjects": len(test),
                             "live_corr_on_tuning": {"median": round(float(np.median(tune_corr)), 3), "p95": round(float(np.quantile(tune_corr, 0.95)), 3)},
                             "threshold_in_code": rppg.BG_CORR_MAX},
        "test_subjects_without_background_check": summarise(test_clips, background=False),
        "test_subjects_with_background_check": summarise(test_clips, background=True),
    }


# ---------------------------------------------------------------- lip-sync on GRID

def mpg_audio(path: str | Path, sr: int = 16000) -> np.ndarray | None:
    import av

    with av.open(str(path)) as c:
        if not c.streams.audio:
            return None
        rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
        parts = [f.to_ndarray().ravel() for fr in c.decode(c.streams.audio[0]) for f in rs.resample(fr)]
    return np.concatenate(parts) if parts else None


def _grid_job(path: str) -> dict | None:
    x = mpg_audio(path)
    if x is None:
        return None
    v = extract_video(path)
    return {"path": path, "fps": v.fps, "mouth": v.mouth_motion, "face_ratio": v.face_ratio, "audio": x}


def grid(folder: str | Path, per_speaker: int = 100, seed: int = 0) -> dict:
    """GRID (Cooke et al., 2006; CC BY 4.0): 3-second sentences filmed face-on at 25 fps."""
    rng = np.random.default_rng(seed)
    paths = []
    videos = [p for p in Path(folder).rglob("*.mpg") if not p.name.startswith("._")]  # skip macOS metadata files
    for spk in sorted({p.parent for p in videos}):
        files = sorted(p for p in videos if p.parent == spk)
        paths += [str(files[i]) for i in sorted(rng.choice(len(files), min(per_speaker, len(files)), replace=False))]
    with ProcessPoolExecutor(min(2, os.cpu_count() or 1)) as pool:
        clips = [c for c in pool.map(_grid_job, paths) if c and c["face_ratio"] >= 0.5]

    sr = 16000
    corr = {"genuine": [], "swapped": [], "lagged_0.5s": []}
    for i, c in enumerate(clips):
        n = len(c["mouth"])
        other = clips[(i + 1) % len(clips)]
        if Path(other["path"]).parent != Path(c["path"]).parent:  # same speaker, different sentence
            other = clips[i - 1]
        lagged = np.concatenate([np.zeros(sr // 2, dtype=np.float32), c["audio"]])[: len(c["audio"])]
        for name, audio in (("genuine", c["audio"]), ("swapped", other["audio"]), ("lagged_0.5s", lagged)):
            corr[name].append(sync_score(envelope_per_frame(audio, sr, c["fps"], n), c["mouth"], c["fps"])[0])

    passes = {k: round(float(np.mean(np.array(v) >= SYNC_MIN_CORR)), 3) for k, v in corr.items()}
    return {
        "dataset": "GRID audio-visual corpus, speakers " + ", ".join(sorted({Path(c["path"]).parent.name for c in clips})),
        "clips": len(clips), "sync_threshold": SYNC_MIN_CORR,
        "median_corr": {k: round(float(np.median(v)), 3) for k, v in corr.items()},
        "passes_sync_check": passes,
        "auc_genuine_vs": {k: round(float(roc_auc_score([0] * len(corr["genuine"]) + [1] * len(v), [-x for x in corr["genuine"] + v])), 3)
                           for k, v in corr.items() if k != "genuine"},
    }


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcd", help="MCD-rPPG folder with db.csv, video/ and ppg_sync/")
    ap.add_argument("--grid", help="GRID folder with one sub-folder of .mpg files per speaker")
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--cache", help="keep the colour traces here so a re-run skips decoding")
    ap.add_argument("--out", default="results/real_face.json")
    a = ap.parse_args(argv)
    res = {}
    if a.mcd:
        res["pulse"] = mcd(a.mcd, a.subjects, a.cache)
    if a.grid:
        res["lipsync"] = grid(a.grid)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
