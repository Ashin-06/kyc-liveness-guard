"""Train and test the voice-spoofing model on ASVspoof 2019 LA (real speech, real attacks).

ASVspoof 2019 logical access (Wang et al., 2020; ODC-By licence): bona fide VCTK speech and
speech from 19 text-to-speech and voice-conversion systems. The training split contains
attacks A01-A06 only; the evaluation split contains A07-A19, most of them never seen in
training, which is the setting a bank faces when a new cloning tool appears.

Read from Hugging Face parquet mirrors (audio stored as FLAC bytes):
    train  Bisher/ASVspoof_2019_LA                          data/train-*.parquet  (system_id "-" = bona fide)
    eval   SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA     data/test-*.parquet   (label 0 = bona fide)

    python -m liveness realdata --train train.parquet --eval test-00000.parquet test-00001.parquet

ASVspoof audio is 16 kHz, so the resulting model suits 16 kHz channels (phone and app
voice). The shipped model, used by the 48 kHz browser capture, is left unchanged.
"""
from __future__ import annotations

import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .audio import extract_features, feature_vector
from .train import eer

MODELS = Path(__file__).resolve().parent.parent / "models"


def features_from_flac(blob: bytes) -> np.ndarray | None:
    try:
        x, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
        return feature_vector(extract_features(x.mean(axis=1), int(sr)))
    except Exception:  # an undecodable clip is skipped, not allowed to stop the run
        return None


def load_features(paths: list[str | Path], pool: ProcessPoolExecutor) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels (1 = spoof) from either mirror's layout, streamed in batches so the
    audio never has to sit in memory all at once."""
    X, y = [], []
    for path in paths:
        pf = pq.ParquetFile(path)
        label_col = "system_id" if "system_id" in pf.schema_arrow.names else "label"
        for batch in pf.iter_batches(batch_size=1024, columns=["audio", label_col]):
            blobs = batch.column(0).field("bytes").to_pylist()
            labels = batch.column(1).to_pylist()
            ys = [lab != "-" for lab in labels] if label_col == "system_id" else [lab == 1 for lab in labels]
            for f, lab in zip(pool.map(features_from_flac, blobs, chunksize=16), ys):
                if f is not None:
                    X.append(f)
                    y.append(lab)
    return np.array(X), np.array(y, dtype=int)


def run(train: list[str], evaluate: list[str], out_model: Path | None = MODELS / "voice_model_asvspoof.joblib",
        shipped: Path = MODELS / "voice_model.joblib", log=print) -> dict:
    with ProcessPoolExecutor(min(4, os.cpu_count() or 1)) as pool:
        log("extracting features from the training audio")
        Xtr, ytr = load_features(train, pool)
        log(f"  {len(ytr)} clips, {int((ytr == 0).sum())} bona fide; now the evaluation audio")
        Xte, yte = load_features(evaluate, pool)

    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, class_weight="balanced", random_state=0).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    res = {"dataset": "ASVspoof 2019 LA: train split (attacks A01-A06) -> eval split (attacks A07-A19)",
           "train_clips": int(len(ytr)), "train_bonafide": int((ytr == 0).sum()),
           "eval_clips": int(len(yte)), "eval_bonafide": int((yte == 0).sum()),
           "trained_on_asvspoof": {"eer": round(eer(yte, p), 4), "auc": round(float(roc_auc_score(yte, p)), 4)},
           "published_baselines_eer": {"LFCC-GMM": 0.0809, "CQCC-GMM": 0.0957},
           "note": ("Evaluation uses 2 of the 9 eval shards (about a fifth of the eval set); the published "
                    "baselines are on the full set. Same 37 hand-crafted features as the live system, gradient boosting.")}
    if shipped.exists():
        ps = joblib.load(shipped).predict_proba(Xte)[:, 1]
        res["shipped_synthetic_model"] = {"eer": round(eer(yte, ps), 4), "auc": round(float(roc_auc_score(yte, ps)), 4)}
    if out_model:
        joblib.dump(clf, out_model)
        res["saved_model"] = str(out_model.name)
    return res


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--out", default="results/real_data.json")
    a = ap.parse_args(argv)
    res = run(a.train, a.eval)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
