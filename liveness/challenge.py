"""Timed challenge-response.

The server picks a random schedule of cues ("say WORD", "turn head LEFT") at random
times on the recording timeline. A pre-recorded clip or an offline voice clone cannot
know the schedule, so speech onsets and head turns must land right after each cue.
This needs no speech recognition: timing alone is verified server side.

Tokens are HMAC-signed and single use. Redeemed token ids are kept in the database
until they expire, so a replay is refused after a restart and on every replica.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import asdict, dataclass

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from . import db

WORDS = (
    "amber anchor apple arrow basket bridge butter candle canyon carpet castle cherry cobalt "
    "copper cotton crystal desert dragon falcon forest garden ginger glacier harbor helmet "
    "island jasmine kettle lantern lemon marble meadow mirror monsoon nectar ocean orchid "
    "paddle pepper pillow planet pocket rabbit river saddle salmon shadow silver spider "
    "summit tiger timber tunnel velvet violet walnut window winter yellow zebra"
).split()

TOKEN_TTL_S = 180
SPEAK_WINDOW = (0.15, 2.2)  # seconds after cue in which speech must start
TURN_WINDOW = (0.2, 2.8)
# The client shows a mirrored preview but records the raw camera frame. A user turning
# to *their* left therefore appears facing image-right in the recording.
USER_TO_IMAGE = {"left": "right", "right": "left"}


@dataclass
class Cue:
    t: float
    kind: str  # "speak" | "turn"
    value: str


class ChallengeError(Exception):
    pass


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class ChallengeService:
    def __init__(self, secret: bytes | None = None, ttl: int = TOKEN_TTL_S, engine: Engine | None = None,
                 previous_keys: tuple[bytes, ...] = ()):
        self.secret = secret or secrets.token_bytes(32)
        self.keys = [self.secret, *previous_keys]  # tokens issued before a key rotation still redeem
        self.ttl = ttl
        self.db = engine or db.connect("sqlite://")

    def issue(self, n_speak: int = 3, n_turn: int = 2, now: float | None = None) -> tuple[str, dict]:
        rng = secrets.SystemRandom()
        kinds = ["speak"] * n_speak + ["turn"] * n_turn
        rng.shuffle(kinds)
        t, cues = 1.5, []
        for k in kinds:
            value = rng.choice(WORDS) if k == "speak" else rng.choice(["left", "right"])
            cues.append(Cue(round(t, 2), k, value))
            t += rng.uniform(2.6, 3.6)  # gaps random so timing cannot be guessed
        now = now or time.time()
        payload = {
            "id": secrets.token_hex(12),
            "iat": int(now),
            "exp": int(now + self.ttl),
            "duration": round(t + 0.5, 2),
            "cues": [asdict(c) for c in cues],
        }
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        sig = hmac.new(self.secret, body, hashlib.sha256).digest()
        return f"{_b64(body)}.{_b64(sig)}", payload

    def redeem(self, token: str, now: float | None = None) -> dict:
        """Verify signature, expiry and single use. Consumes the token."""
        now = now or time.time()
        try:
            body_b64, sig_b64 = token.split(".")
            body = _unb64(body_b64)
            sig = _unb64(sig_b64)
        except ValueError as e:
            raise ChallengeError("malformed token") from e
        if not any(hmac.compare_digest(sig, hmac.new(k, body, hashlib.sha256).digest()) for k in self.keys):
            raise ChallengeError("bad signature")
        payload = json.loads(body)
        if now > payload["exp"]:
            raise ChallengeError("token expired")
        try:
            with self.db.begin() as c:
                c.execute(text("DELETE FROM used_challenges WHERE expires_at < :now"), {"now": int(now)})
                c.execute(text("INSERT INTO used_challenges (id, expires_at) VALUES (:id, :exp)"),
                          {"id": str(payload["id"]), "exp": int(payload["exp"])})
        except IntegrityError:  # the id is already there: redeemed before, here or on another replica
            raise ChallengeError("token already used (replay)") from None
        return payload


def check_responses(payload: dict, onsets: list[float], yaw: list[str] | None, fps: float | None) -> dict:
    """Match speech onsets and head-turn frames against the cue schedule."""
    cues = [Cue(**c) for c in payload["cues"]]
    results, used_onsets = [], set()
    for c in cues:
        if c.kind == "speak":
            hit = next((o for o in onsets if c.t + SPEAK_WINDOW[0] <= o <= c.t + SPEAK_WINDOW[1] and o not in used_onsets), None)
            if hit is not None:
                used_onsets.add(hit)
            results.append({**asdict(c), "passed": hit is not None, "observed": None if hit is None else round(hit - c.t, 2)})
        else:
            ok = None
            if yaw and fps:
                want = USER_TO_IMAGE[c.value]
                lo, hi = int((c.t + TURN_WINDOW[0]) * fps), int((c.t + TURN_WINDOW[1]) * fps)
                window = yaw[lo:hi]
                ok = sum(1 for y in window if y == want) >= 3 or (
                    # profile cascades miss many real turns; a clear frontal dropout counts as weak evidence
                    sum(1 for y in window if y == "front") < 0.3 * max(len(window), 1)
                    and sum(1 for y in window if y == USER_TO_IMAGE["left" if c.value == "right" else "right"]) < 2
                )
            results.append({**asdict(c), "passed": ok})

    speak = [r for r in results if r["kind"] == "speak"]
    turn = [r for r in results if r["kind"] == "turn" and r["passed"] is not None]
    # speech that starts nowhere near any cue suggests a continuous playback
    stray = [o for o in onsets if o not in used_onsets]
    speak_ok = all(r["passed"] for r in speak)
    turn_ok = (sum(r["passed"] for r in turn) >= (len(turn) + 1) // 2) if turn else None
    return {
        "cues": results,
        "speech_timing_ok": speak_ok,
        "head_turn_ok": turn_ok,
        "stray_onsets": len(stray),
        "passed": speak_ok and turn_ok is not False and len(stray) <= len(speak) + 2,
    }
