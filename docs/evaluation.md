# Evaluation

## Attack benchmark (synthetic)

`python -m liveness redteam --trials 20`, one run with the locked dependencies:

```
scenario                    pass  step   rej   risk  metric
----------------------------------------------------------------
bona_fide                     20     0     0   0.09  BPCER 0.0%
photo_tts                      0     0    20   0.86  APCER 0.0%
screen_replay                  0     0    20   0.20  APCER 0.0%
faceswap_clone                 3     8     9   0.55  APCER 15.0%
clone_noise_masked             0     0    20   0.67  APCER 0.0%
virtual_cam_old_session        0     0    20   0.14  APCER 0.0%
```

APCER and BPCER follow ISO/IEC 30107-3: APCER is the share of attacks accepted, BPCER the share of genuine users not accepted. The challenge schedule is random on every run, so the counts move a little. Over three runs, `faceswap_clone` APCER was 15%, 20% and 15%; every other attack stayed at 0% and BPCER at 0%.

Two of the attackers adapt to the defence:
- `clone_noise_masked` adds fake room noise to hide digital silence. The spectral edge and pitch jitter still catch it.
- `faceswap_clone` follows the cues live, so timing alone can't catch it and the pulse and voice checks have to. With the pulse thresholds calibrated on real webcams (below), 3 or 4 of 20 get through. With the earlier thresholds none did, but those settings also rejected two thirds of real people.

Voices in this benchmark come from a source-filter model and faces are RGB traces, so treat it as a regression suite rather than a field accuracy figure.

## Real speech: ASVspoof 2019

[ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336) (ODC-By) has real speech and real spoofing attacks. Training used the 25,380-clip train split, whose spoofs come from attacks A01–A06. Testing used 15,831 clips (2 of the 9 eval shards) whose spoofs come from attacks A07–A19, most of which never appear in training. That is the position a bank is in when a new cloning tool comes out.

`python -m liveness realdata` writes `results/real_data.json`.

| Voice model | Equal error rate | AUC |
|---|---|---|
| Model trained on the synthetic voices | 50.5% (chance) | 0.50 |
| Same 37 features, trained on ASVspoof | 15.0% | 0.93 |
| Published baselines, full eval set: LFCC-GMM / CQCC-GMM | 8.1% / 9.6% | |

The model trained on synthetic voices is no better than chance on real deepfakes. Trained on real speech, the same features catch most unseen attacks but still trail the published spectral baselines. The real-speech model is used for audio at 16 kHz or below (phone and app channels). The 48 kHz browser path keeps the original model and should be calibrated on recordings from the bank's own channel.

## Real faces: MCD-rPPG

[MCD-rPPG](https://huggingface.co/datasets/kyegorov/mcd_rppg) (Egorov et al., ACM MM 2025; CC BY 4.0) has frontal webcam recordings (640×480, 30 fps) of people at rest and after exercise, with a finger pulse sensor synchronised to every frame. It was used here only for pulse estimation, which is what it is published for.

From 20 people, two 15-second windows per recording were scored as live, about the length of a KYC capture. Three photo attacks were built from a frame of each recording: the still frame with sensor noise, the same under a lamp flickering at 72 bpm, and under a green-tinted flicker aimed at the pulse method.

`python -m liveness realface` writes `results/real_face.json`.

Heart rate matches the finger sensor closely: the median error over 80 clips is 0.9 bpm, and 76% of clips are within 5 bpm. The mean error of 6.8 bpm comes from a few clips where the estimate locked onto a harmonic.

The original thresholds rejected 67.5% of real faces, because real webcam video gives a weaker pulse signal than they assumed. The thresholds were reset on 10 people so that at most 10% of their clips fail, and a new check rejects a pulse that also appears in the background of the frame (a lamp or screen lights the room, blood only colours skin). Results on the other 10 people:

| Held-out people | Pass | Review | Reject |
|---|---|---|---|
| Live faces, old thresholds | 12.5% | 20% | 67.5% |
| Live faces, new thresholds | 57.5% | 27.5% | 15% |
| Photo | 5% | 35% | 60% |
| Photo under a white lamp flickering at 72 bpm | 30% | 5% | 65% |
| Photo under a green-tinted flicker, without the background check | 100% | 0% | 0% |
| Photo under a green-tinted flicker, with the background check | 0% | 0% | 100% |

On a 15-second webcam clip the pulse is weak evidence. It stops most photos but also sends a quarter of real people to review, so it can't decide on its own; against a printed photo, the head-turn cues carry most of the weight. The live reject rate on the held-out people (15%) is above the 10% targeted on the tuning people.

## Lip sync: GRID

[GRID](https://zenodo.org/records/3625687) (Cooke et al., 2006; CC BY 4.0): 200 three-second sentences from two speakers. Each clip's mouth movement was scored against its own audio, against another sentence by the same speaker, and against its own audio 0.5 s late.

| Audio | Passes the sync check | AUC against genuine |
|---|---|---|
| Its own | 100% | |
| Another sentence, same speaker | 100% | 0.54 |
| Its own, 0.5 s late | 76.5% | 0.86 |

The check never rejected a real speaker, but it can't tell which words were said: it compares mouth movement with loudness, and every GRID sentence has the same rhythm. It catches gross mismatches only, such as speech over a still mouth or audio far out of step. Matching the actual words would need a lip-reading model.
