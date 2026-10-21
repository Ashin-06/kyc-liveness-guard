"""Audio-visual synchrony: mouth motion should follow the speech envelope within ~100 ms.
Dubbed deepfakes, voice-converted streams over a real face and replays with separate
audio all break this coupling."""
from __future__ import annotations

import numpy as np

SYNC_MIN_CORR = 0.25  # calibration knob
MAX_LAG_S = 0.30


def envelope_per_frame(x: np.ndarray, sr: int, fps: float, n_frames: int) -> np.ndarray:
    hop = sr / fps
    out = np.zeros(n_frames)
    for i in range(n_frames):
        seg = x[int(i * hop) : int((i + 1) * hop)]
        out[i] = np.sqrt(np.mean(seg**2)) if len(seg) else 0.0
    return out


def _z(a: np.ndarray) -> np.ndarray:
    a = np.convolve(a, np.ones(3) / 3, "same")
    return (a - a.mean()) / (a.std() + 1e-9)


def sync_score(envelope: np.ndarray, mouth: np.ndarray, fps: float) -> tuple[float, float]:
    """Best Pearson correlation over lags in +-MAX_LAG_S, and the lag in seconds
    (positive = mouth lags audio)."""
    n = min(len(envelope), len(mouth))
    e, m = _z(envelope[:n]), _z(mouth[:n])
    best, best_lag = -1.0, 0
    for lag in range(-int(MAX_LAG_S * fps), int(MAX_LAG_S * fps) + 1):
        a, b = (e[: n - lag], m[lag:]) if lag >= 0 else (e[-lag:], m[: n + lag])
        if len(a) < 10:
            continue
        c = float(np.mean(a * b))
        if c > best:
            best, best_lag = c, lag
    return best, best_lag / fps


def lipsync_risk(corr: float) -> tuple[float, list[str]]:
    if corr >= SYNC_MIN_CORR:
        return float(1 / (1 + np.exp(4 * (corr - SYNC_MIN_CORR) + 1.5))), []
    return float(1 / (1 + np.exp(-(SYNC_MIN_CORR - corr) * 8 - 0.5))), [
        f"mouth movement does not follow the voice (sync r={corr:.2f})"
    ]
