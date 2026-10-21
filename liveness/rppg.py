"""Remote photoplethysmography (rPPG) liveness.

A live face shows a faint colour pulse from blood volume changes, strongest in the
green channel and in the same rhythm on forehead and both cheeks. A printed photo has
no pulse. A face-swap renders skin texture per frame and tends to destroy or scramble
the pulse between regions. A screen replay keeps a weak pulse but adds flicker.

Pulse extraction uses the POS method (Wang et al., 2017).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

HR_BAND = (0.7, 3.5)  # 42 - 210 bpm
REGIONS = ("forehead", "cheek_l", "cheek_r")

# Calibration knobs. Tune on your own camera pipeline.
SNR_LIVE_DB = 1.0       # POS SNR on a 10 s clip from a laptop webcam usually lands in 2..8 dB
BPM_AGREEMENT = 9.0     # max spread of per-region heart rates, bpm
# Which way OpenCV's profile cascade "looks" in the raw frame. The stock cascade is
# trained on faces turned toward image-left; flip if your build disagrees.
PROFILE_CASCADE_LOOKS = "left"


def pos(rgb: np.ndarray, fps: float) -> np.ndarray:
    """Plane-Orthogonal-to-Skin pulse signal from a (T, 3) mean-RGB trace."""
    rgb = np.asarray(rgb, dtype=float)
    t = len(rgb)
    win = max(int(1.6 * fps), 8)
    proj = np.array([[0, 1, -1], [-2, 1, 1]], dtype=float)
    h = np.zeros(t)
    for n in range(0, t - win + 1):
        c = rgb[n : n + win].T
        cn = c / (c.mean(axis=1, keepdims=True) + 1e-9)
        s = proj @ cn
        p = s[0] + (s[0].std() / (s[1].std() + 1e-9)) * s[1]
        h[n : n + win] += p - p.mean()
    return h


def bandpass(x: np.ndarray, fps: float, lo: float = HR_BAND[0], hi: float = HR_BAND[1]) -> np.ndarray:
    b, a = signal.butter(3, [lo / (fps / 2), min(hi / (fps / 2), 0.99)], btype="band")
    return signal.filtfilt(b, a, x, padlen=min(len(x) - 1, 3 * max(len(a), len(b))))


def spectrum(x: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    nfft = 1 << int(np.ceil(np.log2(len(x) * 8)))
    f = np.fft.rfftfreq(nfft, 1 / fps)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=nfft)) ** 2
    return f, p


def pulse_snr(x: np.ndarray, fps: float) -> tuple[float, float]:
    """(heart rate bpm, SNR dB) with the de Haan definition: fundamental +-0.1 Hz and
    first harmonic +-0.2 Hz against the rest of the heart-rate band."""
    f, p = spectrum(x, fps)
    band = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    f0 = f[band][np.argmax(p[band])]
    sig = (np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.2)
    noise = band & ~sig
    snr = 10 * np.log10(p[sig & (f <= 2 * HR_BAND[1])].sum() / (p[noise].sum() + 1e-12) + 1e-12)
    return float(f0 * 60), float(snr)


@dataclass
class PulseReport:
    bpm: float
    snr_db: float
    region_bpm: dict
    region_snr: dict
    bpm_spread: float
    risk: float
    reasons: list[str] = field(default_factory=list)
    pulse: np.ndarray | None = None


def analyze_traces(traces: dict[str, np.ndarray], fps: float) -> PulseReport:
    pulses, rbpm, rsnr = {}, {}, {}
    for name in REGIONS:
        if name not in traces:
            continue
        pulses[name] = bandpass(pos(traces[name], fps), fps)
        rbpm[name], rsnr[name] = pulse_snr(pulses[name], fps)
    combined = np.sum([p / (p.std() + 1e-9) for p in pulses.values()], axis=0)
    bpm, snr = pulse_snr(combined, fps)
    spread = float(max(rbpm.values()) - min(rbpm.values())) if len(rbpm) > 1 else 0.0

    reasons, score = [], -2.0
    if snr < SNR_LIVE_DB:
        # map SNR deficit smoothly into risk
        score += min(4.0, 1.5 + (SNR_LIVE_DB - snr) * 0.35)
        reasons.append(f"no clear skin pulse (SNR {snr:.1f} dB)")
    if spread > BPM_AGREEMENT:
        score += 2.0
        reasons.append(f"forehead and cheeks disagree on heart rate ({spread:.0f} bpm spread)")
    if not (45 <= bpm <= 180):
        score += 1.0
        reasons.append(f"implausible heart rate {bpm:.0f} bpm")
    return PulseReport(bpm, snr, rbpm, rsnr, spread, float(1 / (1 + np.exp(-score))), reasons, combined)


# ---------------------------------------------------------------- video -> traces

@dataclass
class VideoTraces:
    fps: float
    regions: dict[str, np.ndarray]
    mouth_motion: np.ndarray
    face_ratio: float
    yaw: list[str]  # per frame: "front" | "left" | "right" | "none" (image coordinates)


def _roi(box, fx0, fy0, fx1, fy1):
    x, y, w, h = box
    return int(x + fx0 * w), int(y + fy0 * h), int(x + fx1 * w), int(y + fy1 * h)


def extract_video(path: str | Path, duration_hint: float | None = None, max_frames: int = 900) -> VideoTraces:
    import cv2

    frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    rois = {k: [] for k in REGIONS}
    mouth, yaw = [], []
    box, prev_mouth, found = None, None, 0
    for _ in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.5, fy=0.5)
        faces = frontal.detectMultiScale(small, 1.15, 5, minSize=(40, 40))
        direction = "none"
        if len(faces):
            nb = 2 * np.array(max(faces, key=lambda b: b[2] * b[3]), dtype=float)
            box = nb if box is None else 0.6 * box + 0.4 * nb  # smooth jitter
            direction, found = "front", found + 1
        else:
            other = "right" if PROFILE_CASCADE_LOOKS == "left" else "left"
            if len(profile.detectMultiScale(small, 1.15, 4, minSize=(40, 40))):
                direction = PROFILE_CASCADE_LOOKS
            elif len(profile.detectMultiScale(cv2.flip(small, 1), 1.15, 4, minSize=(40, 40))):
                direction = other
        yaw.append(direction)

        if box is None:
            for k in REGIONS:
                rois[k].append([np.nan] * 3)
            mouth.append(np.nan)
            continue
        rgb = frame[:, :, ::-1]
        spec = {"forehead": (0.30, 0.08, 0.70, 0.25), "cheek_l": (0.15, 0.50, 0.35, 0.70), "cheek_r": (0.65, 0.50, 0.85, 0.70)}
        for k, s in spec.items():
            x0, y0, x1, y1 = _roi(box, *s)
            patch = rgb[max(y0, 0) : y1, max(x0, 0) : x1]
            rois[k].append(patch.reshape(-1, 3).mean(axis=0) if patch.size else [np.nan] * 3)
        x0, y0, x1, y1 = _roi(box, 0.30, 0.70, 0.70, 0.95)
        m = gray[max(y0, 0) : y1, max(x0, 0) : x1]
        m = cv2.resize(m, (32, 16)).astype(float) if m.size else None
        mouth.append(np.nan if m is None or prev_mouth is None else float(np.mean(np.abs(m - prev_mouth))))
        prev_mouth = m
    cap.release()

    n = len(yaw)
    if n < 30:
        raise ValueError("video too short for liveness analysis (need at least 30 frames)")
    # MediaRecorder webm files often carry no usable fps; trust the client duration instead
    if duration_hint and (not fps or fps < 5 or fps > 120 or abs(n / fps - duration_hint) > 1.0):
        fps = n / duration_hint
    fps = fps if fps and 5 <= fps <= 120 else 30.0

    def fill(a):
        a = np.array(a, dtype=float)
        if a.ndim == 1:
            a = a[:, None]
        for c in range(a.shape[1]):
            col, ok = a[:, c], ~np.isnan(a[:, c])
            a[:, c] = np.interp(np.arange(len(col)), np.flatnonzero(ok), col[ok]) if ok.any() else 0.0
        return a

    return VideoTraces(
        fps=float(fps),
        regions={k: fill(v) for k, v in rois.items()},
        mouth_motion=fill(mouth)[:, 0],
        face_ratio=found / n,
        yaw=yaw,
    )
