# Threat model

What the liveness service protects, from whom, and what is still open. Written for the 1.0.0
code; it has not been reviewed by anyone independent.

## What it protects

| Asset | Where it lives |
|---|---|
| The onboarding decision (is a live, present person behind this session?) | returned to the caller per `/v1/verify`; logged with risk and challenge id |
| Face video and voice recordings (biometric data) | in transit and in memory only; video is written to a temporary file for decoding and deleted straight after |
| Challenge signing key | environment / secret store (`LIVENESS_CHALLENGE_KEY`) |
| Redeemed challenge ids | `used_challenges` table until the token expires |
| Service API keys | callers hold the key; the service holds only SHA-256 hashes |
| Voice models | `models/` in the image, read only |

## Trust boundaries

```
customer device ──► bank's onboarding app / back end ──(TLS at your ingress)──► liveness service
                                                                                    │
                                                                                    └──► database (redeemed challenges)
```

1. **Customer device → onboarding app.** Everything the device sends is attacker-controlled:
   the camera feed, the microphone, the timing. The service only sees what the app forwards.
2. **App → liveness service.** Callers are authenticated by API key; the key decides the role.
   The bundled capture page is a demo of the flow, not a customer-facing front end.
3. **Service → database.** Holds only challenge ids and expiry times, no biometrics.

## Threats and what stops them

| Threat | Mitigation | Residual risk |
|---|---|---|
| Voice clone or TTS | Spectral edge, noise floor, pitch jitter and reverberation rules; ASVspoof-trained model for 16 kHz audio | **15.0% EER on unseen ASVspoof attacks** (16 kHz); the 48 kHz path has rules only in the image and was never validated on real speech. Real-time converters at 48 kHz weaken it further. |
| Printed photo or screen replay | Pulse (rPPG) on forehead and cheeks, background-pulse check, head-turn cues | On held-out real webcams 5% of photos passed the pulse check and 35% were held; the head-turn cues carry most of the weight |
| Face swap with a cloned voice, following the cues live | Pulse, voice and lip sync together | Passes on part of the synthetic benchmark (see README); lip sync cannot tell which words were said |
| Pre-recorded session or injected virtual camera | Random cue schedule per token, speech and head turns must land after each cue | A live puppeteered deepfake can follow cues; stop camera injection at the device (attestation) |
| Replay of a captured token | HMAC-signed tokens, 3-minute expiry, single use enforced by the database primary key across restarts and replicas | A token is a bearer value for those 3 minutes; bind it to your own session on the app side |
| Forged token | HMAC-SHA256 with a 32-byte key; no default key in any mode | A leaked key lets anyone mint tokens until it is rotated |
| Stolen API key | Per-client keys, roles (`kyc_app` cannot run the analysis endpoints), rate limit per key, request log with user | Works until removed from the key list and the service restarts |
| Oversized uploads, decoder abuse | 50 MB body cap, 40 MB per file, rate limit; OpenCV/FFmpeg and libsndfile parse uploads | A decoder vulnerability in OpenCV's FFmpeg or libsndfile would run inside the service; keep the image patched (pip-audit in CI, rebuild often) |
| Biometric data kept longer than needed | Nothing is stored; the temp video file is deleted in a `finally` block | A crash between write and delete can leave a file in `/tmp` of the container; the container filesystem is discarded on restart |
| Errors leak data | Validation errors carry no input; unhandled errors return only a request ID; temp paths are removed from decoder errors | — |
| Unfair rejection of some users | Thresholds documented; very dark skin in poor light is a known false-reject source | The pulse thresholds come from one public webcam set; measure per channel and per population before use (DPDP and RBI fairness expectations) |
| Misconfigured production start | `LIVENESS_ENV=production` refuses to start without hashed keys, challenge key and database; no public bind without keys | Production mode can't check that keys are *good*, only present and well formed |

## Not covered

- TLS. Terminate it at your ingress; the service speaks plain HTTP.
- Device and app attestation (Play Integrity, App Attest), which is the real defence against
  injected camera streams.
- Matching the face to an ID document (face match), and deciding what to do with a STEP_UP.
- Storage of recordings for audit or dispute; if the bank needs it, that is a separate,
  access-controlled store with its own retention rules.
