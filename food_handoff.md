# Food Handoff Runtime

This flow is triggered by speech. ChatGPT Realtime listens for a request,
selects `strawberry`, `oreo`, or `marshmallow` through the `run_handoff` tool,
then the lower-level runner immediately starts the matching policy, detects
whether the food reaches the hand, and speaks success or OOD audio through
ElevenLabs.

Mock mode and robot mode follow the same sequence. `--test-mode` replaces
LeRobot, hardware, policy execution, OOD scoring, and success detection with
deterministic mocks. Removing `--test-mode` runs the same flow against the
SO-101.

Robot mode repeats until interrupted by default. Test mode runs one cycle by
default, so local commands finish; use `--max-cycles N` to test multiple cycles.
Between cycles the runner pauses for `RESET_PAUSE_S=7` seconds by default so the
scene can be reset before the next voice-triggered cycle.

## Setup

Review the checked-in runtime configs:

- `config/food_policies.json` with the three Hugging Face policy repo ids.
- `config/food_handoff_vision.json` with camera ROI and color thresholds. The
  current config uses camera index 1 / `side` for placement success because
  that scene view shows the food-in-hand state most clearly.
- `.env` with `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, and
  `ELEVENLABS_MODEL_ID` for text-to-speech.

There is no separate STT model setting. Speech-to-text uses ElevenLabs
`scribe_v2` internally because the STT endpoint accepts Scribe models rather
than TTS models such as `eleven_v3`.

The canonical target enum is `marshmallow`; the speech classifier also accepts
the spoken/transcribed misspelling `marshmellow`.

## Mocked Local Run

Use `--test-mode` for the simple local path. This does not connect to SO-101 and
does not require LeRobot, policy checkpoints, or OOD detector files. Speech
input and output are enabled by default:

```bash
./scripts/run_food_handoff.sh \
  --test-mode \
  --test-audio recordings/voice_requests/strawberry_request.wav
```

Run multiple mocked cycles with the reset pause visible in logs, keeping speech
input and output enabled:

```bash
./scripts/run_food_handoff.sh \
  --test-mode \
  --test-audio recordings/voice_requests/oreo_request.wav \
  --max-cycles 3 --reset-pause-s 7
```

Debug-only overrides, when you explicitly do not want audio:

```bash
./scripts/run_food_handoff.sh --test-mode --no-voice --test-audio recordings/voice_requests/oreo_request.wav
./scripts/run_food_handoff.sh --test-mode --no-voice --no-stt --target strawberry
```

## Robot Run

Run the production Realtime flow against SO-101:

```bash
./scripts/run_realtime_food_handoff.sh
```

Speech input and output are enabled by default. Use `--max-cycles N` to stop
after N lower-level handoffs when debugging the lower-level runner, or leave it
unset for an unlimited Realtime robot loop:

```bash
./scripts/run_food_handoff.sh --max-cycles 1
./scripts/run_food_handoff.sh --reset-pause-s 10
```

Debug-only overrides are available as `--no-voice` and
`--no-stt --target strawberry|oreo|marshmallow`.

## Flow

1. `LISTEN`: ChatGPT Realtime listens to the microphone and transcribes the
   user request.
2. `ORCHESTRATE`: ChatGPT calls `run_handoff(target)` for a clear Strawberry,
   Oreo, or Marshmallow request, calls `unsupported_item_requested(item)` for
   out-of-set foods, or asks a short clarification for ambiguity.
3. `LOAD_OR_SELECT_POLICY`: The lower-level runner loads the policy repo id from
   `food_policies.json`.
4. `FETCHING_TTS`: ElevenLabs speaks the fetching phrase for the selected food.
5. `RUN_POLICY`: The selected LeRobot policy starts immediately.
6. `SUCCESS`: GPT-5.4-nano checks side-camera frames in the background. Success
   requires the correct target food in the user's hand plus visible evidence that
   the robot gripper is near the hand and actively placing or just releasing that
   food. A stalled robot, wrong item, item only on the tray, item only in the
   gripper, or user grab without robot placement must be rejected. Local
   ROI/color detection is only a candidate trigger/log signal, not an accepted
   success source.
7. `OOD`: Speak an OOD phrase on scene OOD or unclassified requests.
8. `RESET_PAUSE`: Wait 5-10 seconds, default 7, so the scene can be reset.
9. Return to `LISTEN` unless the max cycle count was reached or the process was
   interrupted.

## Logs

Expected log markers:

```text
[CYCLE] start cycle=1
[REQUEST] transcript='Please bring me the Oreo' target=oreo
[REQUEST] target=oreo policy=org/oreo-policy task='Pick up the Oreo...'
[TTS] queue wait=True phrase='...Oreo...'
[POLICY] starting target=oreo fps=30 duration=30s
[OOD] frame=87 score=34.219 threshold=21.080
[OPENAI_SUCCESS] frame=... success=True confidence=...
[TASK_SUCCESS] target=oreo frame=342 source=openai opencv_score=...
Run complete: target=oreo success=True success_frame=342 ...
[CYCLE] complete cycle=1 outcome=success
[CYCLE] reset pause 7.0s; reset the scene before the next cycle
```

## Smoke Tests

TTS:

```bash
python3 eleven_labs_test.py
```

STT, using a previously recorded request clip:

```bash
python3 eleven_labs_stt_test.py request.wav
```

Extract and replay non-robot fixtures from a Hugging Face LeRobot dataset:

```bash
uv run python scripts/extract_handoff_fixtures.py \
  --download \
  --repo-id ofcourseistillloveyou/so101_recording_strawberry_num40_20260516_161110

uv run python scripts/replay_handoff_mock.py
```

## Mocking Without The Robot

To accurately mock this flow without connecting to the SO-101, collect a small
fixture pack:

- LeRobot-style observation samples for each state: empty scene, each target on
  the pickup surface, each target delivered, and obvious OOD scenes.
- Matching `front` and `side` camera frames from the real camera positions. The
  OpenCV ROI detectors are camera-geometry dependent, so synthetic images are
  much less useful than frames from the real setup.
- A few short request audio clips for each target plus transcripts from
  ElevenLabs STT, including the misspelling-prone Marshmallow/Marshmellow case.
- The three trained policy repo IDs and either their fitted OOD detector files
  or representative training-distribution frames so detectors can be fit.
- Success labels for fixture frames: target, whether the item was delivered,
  and whether the scene should be OOD.

Training data is useful if it contains the real camera observations from the
three tasks. It lets us replay observations through the classifier, OOD scorer,
and ROI success detector without moving hardware, and it provides realistic
thresholds for success debounce tuning.
