# OOD Detection

Out-of-distribution detection wrapped around the SO-101 inference loop. When the front camera sees something the policy wasn't trained on — novel lighting, an unseen object, a different background — we log it. The policy still executes; this is observation only.

## How it works

```
camera frame  ──►  DINOv2 ViT-S/14  ──►  L2-normalised vector
                                              │
                              ┌───────────────┘
                              ▼
              (PCA → 32 dims, then Mahalanobis distance)
                              │
                              ▼
              score ≥ τ ? → log "[OOD] frame=N score=… threshold=…"
                              │
                              ▼
                       policy.predict()      (always executes)
```

The detector is fit once on training-distribution frames. At runtime each frame is encoded, scored, and compared against the threshold τ (the 95th percentile of in-distribution scores by default).

## Files

| Path | Purpose |
|---|---|
| `src/lerobot_ood/detector.py` | `OODDetector` — Mahalanobis density model with PCA, fit/score/save/load |
| `src/lerobot_ood/encoder.py`  | `DinoV2Encoder` — frozen DINOv2 ViT-S/14, image → 384-dim vector |
| `src/lerobot_ood/obs.py`      | `extract_camera_frame()` — pluck a camera frame from a lerobot obs dict |
| `scripts/fit_ood_detector.py` | One-shot fit from a `LeRobotDataset` |
| `scripts/run_policy_with_ood.py` | Custom inference loop with per-frame OOD scoring |
| `scripts/run_policy_with_ood.sh` | Bash launcher matching the style of the other scripts |

## Setup

```bash
uv sync                       # installs lerobot[feetech] + torch + sklearn into .venv
```

The first `fit` / `run` call will download the DINOv2 ViT-S/14 weights (~85 MB) into `~/.cache/torch/hub/`.

## Step 1: fit the detector

```bash
uv run python scripts/fit_ood_detector.py \
  --dataset_repo_id ofcourseistillloveyou/so-101-feed-me \
  --camera_name front \
  --output_path models/ood_detector.npz
```

Defaults:

- `--max_frames 2000` — random subsample of training frames (set `0` to use all)
- `--pca_components 32` — PCA pre-projection before Mahalanobis
- `--threshold_percentile 95.0` — at this threshold ~5% of clean in-dist frames will be flagged
- `--encoder dinov2_vits14` — also accepts `dinov2_vitb14` / `dinov2_vitl14`
- `--device` — auto-selects CUDA → MPS → CPU

Output is a single `.npz` containing the fitted mean, inverse covariance, PCA components, and threshold.

## Step 2: run inference with OOD logging

```bash
./scripts/run_policy_with_ood.sh
```

Same env-var contract as `scripts/run_policy_on_robot.sh` plus four OOD knobs:

| Env var | Default | Purpose |
|---|---|---|
| `OOD_DETECTOR_PATH` | `models/ood_detector.npz` | Path to the fitted detector |
| `OOD_CAMERA` | `front` | Which camera feeds the OOD score |
| `OOD_ENCODER` | `dinov2_vits14` | Must match the encoder used at fit time |
| `OOD_LOG_IN_DIST_EVERY_N` | `0` | If >0, also print in-dist scores every N frames (debugging) |

For first runs, keep the episode short and watch the log:

```bash
DURATION=15 OOD_LOG_IN_DIST_EVERY_N=30 ./scripts/run_policy_with_ood.sh
```

You'll see two kinds of lines:

```
[in-dist] frame=30  score=12.412  threshold=21.080
[OOD]     frame=87  score=34.219  threshold=21.080  (1/87 so far, 1.1%)
```

At the end of the run:

```
Run complete: 450 frames, 12 flagged OOD (2.7%), mean score=14.831, threshold=21.080
```

## Interpreting the output

- **OOD rate ~5% on the trained scene** is the expected baseline (matches the 95th-percentile threshold).
- **OOD rate spikes** when you change something — new lighting, new objects, a different table cover. That's the signal we're building toward.
- **OOD rate ~0%** in a new scene means the threshold is too loose (raise `--threshold_percentile`) or the encoder isn't discriminating (try `dinov2_vitb14`).
- **OOD rate ~100%** in the trained scene means the detector wasn't fit on representative data — refit with more frames or check the camera index hasn't changed.

The encoder used at fit time **must match** the one used at runtime. If you refit with `dinov2_vitb14`, also export `OOD_ENCODER=dinov2_vitb14` when running.

## Tuning the threshold

Two options:

1. **Percentile-based (default).** Refit with `--threshold_percentile 99` for fewer false positives. This is appropriate when you don't have labelled OOD examples.
2. **Labelled OOD set.** Not implemented in the current code — would require collecting a few rollouts where you deliberately introduced OOD conditions (new lighting, new object) and adding an AUROC-tuned threshold path to `OODDetector`.

## Latency budget

At 30 FPS the control loop has 33 ms per tick. Approximate costs on the OOD path:

| Step | Cost |
|---|---|
| `extract_camera_frame` | <1 ms |
| `DinoV2Encoder.__call__` (ViT-S, MPS) | ~3 ms |
| `OODDetector.score` (PCA + Mahalanobis) | <1 ms |
| **Total OOD overhead** | **~4 ms** |

If you see `loop running slow` warnings:

- The encoder is the long pole — keep `--ood_encoder dinov2_vits14` (smallest) and ensure `--device mps` (or `cuda`).
- Skipping OOD scoring on alternate frames is an easy 50% saving; would go in `run_policy_with_ood.py` around the `encoder(frame)` call.

## Limitations today

- **Log-only.** The policy never halts on OOD — this is the observation phase by design. Promoting to a halt mode is the next step (below).
- **Single-camera gate.** Only the `front` frame is scored. `side` is captured for the policy but ignored by the detector. Multi-camera scoring (max/mean/OR) is a small extension.
- **Per-frame, no debounce.** A single noisy frame triggers a log. At 30 FPS this produces chatty output during transient flashes. A K-of-last-N debounce is the obvious next refinement.
- **Encoder ≠ policy backbone.** DINOv2 has strong general visual features but isn't aligned with ACT's ResNet18 vision encoder. Swapping to ACT's own backbone would make OOD scores more directly reflect what the policy sees.

## Next steps

In rough order of value:

1. **Trajectory debounce** — flag only when K of the last N frames are OOD. Cheap, big false-positive reduction.
2. **Takeover logging** — write `(frame, embedding, score, timestamp)` to disk when OOD persists, for offline cluster analysis.
3. **Halt mode** — add `--ood_on=halt` that holds the current pose instead of sending the policy action. SO-101 has enough holding torque that "hold current pose" is the safe stop, not "send no action."
4. **ACT-backbone encoder** — extract ResNet18 from the trained policy checkpoint, swap into `encoder.py` as an alternative.
5. **Failure-mode clustering** — once you have a few dozen logged OOD episodes, cluster them and send representative frames to a VLM for natural-language failure descriptions.

## Reference

The design vocabulary (Mahalanobis scoring, L2-normalised hyperspherical embeddings, contrastive VLM interrogation) comes from the VLA OOD detection design doc at `~/projects/robood/vla_ood_design_doc.md`. This repo vendors the detector core (no `robood` dependency) so the runtime safety story stays self-contained.
