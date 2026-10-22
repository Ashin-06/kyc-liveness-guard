"""Command line entry point.

    python -m liveness redteam --trials 30
    python -m liveness audio sample.wav
    python -m liveness video selfie.mp4 [--audio sample.wav]
    python -m liveness train [--asvspoof PROTOCOL AUDIO_DIR]
    python -m liveness realdata --train train.parquet --eval test-0000*.parquet   # ASVspoof 2019 LA parquet
    python -m liveness serve --port 8001
"""
from __future__ import annotations

import argparse
import json
import sys

from .audio import AudioAnalyzer, load_audio
from .pipeline import evaluate
from .rppg import extract_video


def main(argv=None):
    ap = argparse.ArgumentParser(prog="liveness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("redteam", help="run synthetic attack benchmark")
    r.add_argument("--trials", type=int, default=25)
    r.add_argument("--json", help="write full results here")
    a = sub.add_parser("audio", help="score a voice recording")
    a.add_argument("file")
    a.add_argument("--model", default="models/voice_model.joblib")
    v = sub.add_parser("video", help="score a face video (optionally with its audio)")
    v.add_argument("file")
    v.add_argument("--audio")
    v.add_argument("--duration", type=float)
    sub.add_parser("train", help="train the voice classifier", add_help=False)
    sub.add_parser("realdata", help="train/test on ASVspoof 2019 LA parquet files", add_help=False)
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--port", type=int, default=8001)
    s.add_argument("--host", default="127.0.0.1")

    args, rest = ap.parse_known_args(argv)
    if args.cmd == "redteam":
        from .redteam import run, table

        res = run(args.trials)
        print(table(res))
        if args.json:
            with open(args.json, "w") as f:
                json.dump(res, f, indent=2)
    elif args.cmd == "audio":
        rep = AudioAnalyzer(args.model).analyze(*load_audio(args.file))
        print(f"risk {rep.risk:.2f} ({rep.source})")
        for reason in rep.reasons:
            print(" -", reason)
    elif args.cmd == "video":
        d = evaluate(load_audio(args.audio) if args.audio else None, extract_video(args.file, args.duration))
        print(json.dumps(d.as_dict(), indent=2, default=float))
    elif args.cmd == "train":
        from .train import main as train

        train(rest)
    elif args.cmd == "realdata":
        from .realdata import main as realdata

        realdata(rest)
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("liveness.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
