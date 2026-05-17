# Food Handoff Runtime

This flow waits for a hand, records a short voice request, classifies the request
as `strawberry`, `oreo`, or `marshmallow`, runs the matching policy, detects
whether the food reaches the hand, and speaks success or OOD audio through
ElevenLabs.

Mock mode and robot mode follow the same sequence. `--test-mode` replaces
LeRobot, hardware, policy execution, OOD scoring, and success detection with
deterministic mocks. Removing `--test-mode` runs the same flow against the
SO-101.

Robot mode repeats until interrupted by default. Test mode runs one cycle by
default, so local commands finish; use `--max-cycles N` to test multiple cycles.
Between cycles the runner pauses for `RESET_PAUSE_S=7` seconds by default so the
existing hand can move out of frame before the next hand trigger.

## Setup

Review the checked-in runtime configs:

- `config/food_policies.json` with the three Hugging Face policy repo ids.
- `config/food_handoff_vision.json` with camera ROI and color thresholds. The
  current config uses camera index 1 / `side` for both hand entry and placement
  success, because that scene view shows the food-in-hand state more clearly.
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

Run the same flow against SO-101 by removing `--test-mode`:

```bash
./scripts/run_food_handoff.sh
```

Speech input and output are enabled by default. Use `--max-cycles N` to stop
after N handoffs, or leave it unset for an unlimited robot loop:

```bash
./scripts/run_food_handoff.sh --max-cycles 1
./scripts/run_food_handoff.sh --reset-pause-s 10
```

Debug-only overrides are available as `--no-voice` and
`--no-stt --target strawberry|oreo|marshmallow`.

## Flow

1. `WAIT_FOR_HAND`: Open the configured camera index and debounce hand entry in
   the hand ROI.
2. `RECORD_REQUEST`: Record a short mono WAV clip from the default microphone.
3. `TRANSCRIBE_AND_CLASSIFY`: Send the clip to ElevenLabs STT and classify the
   transcript with local keyword rules.
4. `LOAD_OR_SELECT_POLICY`: Load the policy repo id from `food_policies.json`.
5. `RUN_POLICY`: Run the same LeRobot control loop as the OOD wrapper.
6. `SUCCESS`: Detect target-colored pixels in the configured hand ROI for a
   stable debounce window, then speak success.
7. `OOD`: Speak an OOD phrase on scene OOD or unclassified requests.
8. `RESET_PAUSE`: Wait 5-10 seconds, default 7, so the hand can leave the frame.
9. Return to `WAIT_FOR_HAND` unless the max cycle count was reached.

## Logs

Expected log markers:

```text
[CYCLE] start cycle=1
[HAND] present
[REQUEST] transcript='Please bring me the Oreo' target=oreo
[REQUEST] target=oreo policy=org/oreo-policy task='Pick up the Oreo...'
[POLICY] starting target=oreo fps=30 duration=30s
[OOD] frame=87 score=34.219 threshold=21.080
[TASK_SUCCESS] target=oreo frame=342 confidence=0.091 target_blob_fraction=0.091
Run complete: target=oreo success=True success_frame=342 ...
[CYCLE] complete cycle=1 outcome=success
[CYCLE] reset pause 7.0s; move the hand out of frame before the next cycle
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

- LeRobot-style observation samples for each state: empty scene, hand entering,
  hand present, each target on the pickup surface, each target in the hand, and
  obvious OOD scenes.
- Matching `front` and `side` camera frames from the real camera positions. The
  OpenCV ROI detectors are camera-geometry dependent, so synthetic images are
  much less useful than frames from the real setup.
- A few short request audio clips for each target plus transcripts from
  ElevenLabs STT, including the misspelling-prone Marshmallow/Marshmellow case.
- The three trained policy repo IDs and either their fitted OOD detector files
  or representative training-distribution frames so detectors can be fit.
- Success labels for fixture frames: target, whether the hand is present,
  whether the item is in the hand, and whether the scene should be OOD.

Training data is useful if it contains the real camera observations from the
three tasks. It lets us replay observations through the classifier, OOD scorer,
and ROI success detector without moving hardware, and it provides realistic
thresholds for hand/success debounce tuning.
