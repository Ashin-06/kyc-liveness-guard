"""Train the voice spoofing classifier.

With --asvspoof it reads an ASVspoof 2019/2021 LA style protocol file
(`SPEAKER FILE_ID - SYSTEM LABEL`) and the matching .flac directory. Without it a
synthetic corpus is generated so the pipeline can be exercised end to end.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from . import synth
from .audio import extract_features, feature_vector, load_audio


def eer(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def synthetic_corpus(n: int, seed: int = 0):
    X, y = [], []
    rng = np.random.default_rng(seed)
    for i in range(n):
        s = seed * 100_000 + i
        dur = rng.uniform(2.5, 5.0)
        kind = rng.integers(4)
        if kind == 0 or kind == 1:
            x, label = synth.voice(48000, dur, s, noise_db=rng.uniform(-66, -48)), 0
        elif kind == 2:
            x, label = synth.tts(48000, dur, s, native_sr=int(rng.choice([16000, 22050, 24000])), add_noise=bool(rng.integers(2))), 1
        else:
            x, label = synth.replay(48000, dur, s), 1
        X.append(feature_vector(extract_features(x, 48000)))
        y.append(label)
    return np.array(X), np.array(y)


def asvspoof_corpus(protocol: Path, audio_dir: Path, limit: int | None):
    X, y = [], []
    lines = protocol.read_text().splitlines()[:limit]
    for line in lines:
        parts = line.split()
        file_id, label = parts[1], parts[-1]
        path = next((audio_dir / f"{file_id}{ext}" for ext in (".flac", ".wav") if (audio_dir / f"{file_id}{ext}").exists()), None)
        if path is None:
            continue
        x, sr = load_audio(path)
        X.append(feature_vector(extract_features(x, sr)))
        y.append(0 if label == "bonafide" else 1)
    return np.array(X), np.array(y)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asvspoof", nargs=2, metavar=("PROTOCOL", "AUDIO_DIR"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--synthetic", type=int, default=600, help="synthetic samples when no dataset is given")
    ap.add_argument("--out", default="models/voice_model.joblib")
    a = ap.parse_args(argv)

    if a.asvspoof:
        X, y = asvspoof_corpus(Path(a.asvspoof[0]), Path(a.asvspoof[1]), a.limit)
    else:
        X, y = synthetic_corpus(a.synthetic)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced").fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    print(f"samples={len(y)}  spoof share={y.mean():.2f}  AUC={roc_auc_score(yte, p):.4f}  EER={eer(yte, p):.2%}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
