# SO-101 Voice-Gated Food Handoff

This branch contains the SO-101 helpers plus a voice-gated food handoff flow for
Strawberry, Oreo, and Marshmallow policies.

The handoff loop is the same in mock mode and robot mode:

1. Wait for a hand in the scene camera frame.
2. Record a short user request.
3. Transcribe with ElevenLabs STT and classify the target policy.
4. Run the selected food handoff policy.
5. Score OOD from camera frames while the policy runs.
6. Detect placement success from the scene camera.
7. Optionally confirm success with OpenAI vision.
8. Speak the outcome with ElevenLabs TTS.
9. Pause so the hand can leave the frame, then repeat.

The normal robot path uses local OpenCV detectors for hand entry and placement
success. OpenAI vision is optional and is used only as a post-policy success
confirmation; it is not in the tight robot action loop.

## Production Run

The default production run is a continuous, voice-driven, multi-policy robot
loop:

```bash
./scripts/run_food_handoff.sh
```

Each cycle waits for a hand, records a voice request, classifies the requested
food, loads that food's configured policy, runs the handoff, speaks the outcome,
pauses for hand reset, and then returns to listening. Do not pass `--target` in
production; `--target` is only a debug override and forces the same policy every
cycle.

Enable OpenAI success confirmation for production by setting one environment
variable:

```bash
OPENAI_SUCCESS_ENABLED=true ./scripts/run_food_handoff.sh
```

## Full Loop

The runtime is `scripts/run_food_handoff.sh`, which launches
`scripts/run_food_handoff.py`.

- **Hand gate:** `config/food_handoff_vision.json` opens OpenCV camera index `1`
  and watches the configured `hand.roi`. It builds a short empty-scene baseline,
  then triggers when enough pixels in that ROI change for enough consecutive
  frames.
- **Speech request:** the runner records a short microphone clip, sends it to
  ElevenLabs STT, and classifies the transcript as `strawberry`, `oreo`, or
  `marshmallow`. Debug runs can bypass this with `--no-stt --target strawberry`.
- **Policy selection:** `config/food_policies.json` maps the classified food to
  a Hugging Face policy repo, task prompt, and per-target OOD detector.
- **Policy execution:** the selected LeRobot policy runs on SO-101 for
  `DURATION` seconds or until success is detected.
- **OOD scoring:** the ACT backbone encoder scores the configured `OOD_CAMERA`
  against the target detector `.npz`. OOD events are logged and can trigger
  ElevenLabs voice alerts, but they do not stop the policy.
- **Success detection:** the local ROI/color detector watches named camera
  `side`, which maps to `CAMERA_SIDE_INDEX=1`. This is the scene camera in the
  current setup.
- **OpenAI confirmation:** when `OPENAI_SUCCESS_ENABLED=true`, an OpenCV success
  candidate is sent to `gpt-5.4-nano` using the side-camera frame. The loop only
  accepts success if the model says the target food is in the user's hand with
  at least `OPENAI_SUCCESS_MIN_CONFIDENCE` from `.env` (default `0.70`). If the
  API request fails, the runner logs a warning and falls back to the OpenCV
  success result.
- **Reset/repeat:** robot mode loops until interrupted by default. Test mode
  runs one cycle by default. Use `--max-cycles N` to bound either mode, or
  `--max-cycles 0` for an unlimited loop.

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` with:

```bash
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
ELEVENLABS_MODEL_ID=eleven_v3
```

There is no separate STT model setting. `ELEVENLABS_MODEL_ID` configures
text-to-speech output. Speech-to-text uses ElevenLabs `scribe_v2` internally,
because the STT endpoint accepts Scribe models rather than TTS models such as
`eleven_v3`.

Optional OpenAI success confirmation uses the same `.env` file:

```bash
OAI_KEY=...
OPENAI_VISION_MODEL=gpt-5.4-nano
OPENAI_SUCCESS_MIN_CONFIDENCE=0.70
```

`OPENAI_API_KEY` is also accepted if you prefer that name.

For robot execution, this branch expects a latest LeRobot checkout at
`vendor/lerobot-main`:

```bash
./scripts/update_lerobot_latest.sh
```

## Test Views

Use these four views for normal testing. In all cases, speech input and
ElevenLabs speech output are enabled by default. Unless you pass the debug-only
`--target` override, every cycle listens to the user's voice request and can
select any configured food policy.

### 1. Test One Voice Request, No Robot

Use `--test-mode` with a recorded request clip. This does not connect to SO-101,
does not load LeRobot rollout code, and does not require policy checkpoints or
OOD detector files. It still exercises ElevenLabs STT, target classification,
policy selection, mocked success/OOD handling, and ElevenLabs TTS.

Strawberry:

```bash
./scripts/run_food_handoff.sh \
  --test-mode \
  --test-audio recordings/voice_requests/strawberry_request.wav \
  --max-cycles 1
```

Oreo:

```bash
./scripts/run_food_handoff.sh \
  --test-mode \
  --test-audio recordings/voice_requests/oreo_request.wav \
  --max-cycles 1
```

Marshmallow:

```bash
./scripts/run_food_handoff.sh \
  --test-mode \
  --test-audio recordings/voice_requests/marshmallow_request.wav \
  --max-cycles 1
```

### 2. Test One Voice Request, With Robot

Preconfiguration:

- Confirm `vendor/lerobot-main` exists. If not, run
  `./scripts/update_lerobot_latest.sh`.
- Confirm `config/food_policies.json` has policy repo ids and
  `ood_detector_path` values for every food you may request.
- Confirm the relevant OOD detectors exist locally, for example
  `models/ood_detector_strawberry.npz`.
- Confirm `config/food_handoff_vision.json` uses the scene camera:
  `hand.camera_index=1` and success camera `side`.
- Keep one hand near power/USB for the first robot run.

Run one robot cycle and say any configured food, for example "strawberry",
during the request recording window:

```bash
./scripts/run_food_handoff.sh \
  --max-cycles 1 \
  --reset-pause-s 5
```

Optional OpenAI success confirmation:

```bash
OPENAI_SUCCESS_ENABLED=true \
./scripts/run_food_handoff.sh \
  --max-cycles 1 \
  --reset-pause-s 5
```

### 3. Test A Set Of Voice Requests, No Robot

Test mode takes one recorded request clip at a time. To test a set of possible
voice requests, run one command per recorded clip. This example verifies that
Strawberry and Oreo each select their own configured policy:

```bash
./scripts/run_food_handoff.sh \
  --test-mode \
  --test-audio recordings/voice_requests/strawberry_request.wav \
  --max-cycles 1 \
  --reset-pause-s 5

./scripts/run_food_handoff.sh \
  --test-mode \
  --test-audio recordings/voice_requests/oreo_request.wav \
  --max-cycles 1 \
  --reset-pause-s 5
```

To stress repeated reset behavior for one recorded request clip, increase
`--max-cycles`:

```bash
./scripts/run_food_handoff.sh \
  --test-mode \
  --test-audio recordings/voice_requests/strawberry_request.wav \
  --max-cycles 3 \
  --reset-pause-s 5
```

### 4. Test A Set Of Voice Requests, With Robot

Preconfiguration is the same as the single-cycle robot test: every food the user
might request must have a valid policy repo id and local OOD detector in
`config/food_policies.json`.

Run two robot cycles and request Strawberry during the first cycle, then
Marshmallow during the second cycle:

```bash
./scripts/run_food_handoff.sh \
  --max-cycles 2 \
  --reset-pause-s 5
```

With OpenAI success confirmation:

```bash
OPENAI_SUCCESS_ENABLED=true \
./scripts/run_food_handoff.sh \
  --max-cycles 2 \
  --reset-pause-s 5
```

The production robot loop is the same command without `--max-cycles`; stop it
with Ctrl-C:

```bash
./scripts/run_food_handoff.sh
```

Expected successful logs include:

```text
[CYCLE] start cycle=1
[HAND] present
[REQUEST] transcript='Please put the strawberry in my hand' target=strawberry
[REQUEST] target=strawberry policy=...
[POLICY] starting target=strawberry ...
[TASK_SUCCESS] target=strawberry frame=... confidence=...
[CYCLE] complete cycle=1 outcome=success
```

Debug-only override: `--no-stt --target strawberry` bypasses microphone/STT and
forces one target for every cycle. The normal path uses speech input and output,
and each cycle chooses its policy from the user's request.

Useful robot-mode environment knobs:

```bash
CAMERA_SIDE_INDEX=1              # scene camera used for hand and success
CAMERA_FRONT_INDEX=0             # on-robot/front camera, used by OOD by default
OOD_CAMERA=front
OPENAI_SUCCESS_ENABLED=false     # set true to confirm OpenCV success candidates
OPENAI_SUCCESS_CONFIG_PATH=.env
OPENAI_SUCCESS_EVERY_N=15        # minimum frames between OpenAI confirmation calls
DURATION=30
HAND_WAIT_TIMEOUT_S=0
```

## Smoke Tests

Run unit tests:

```bash
uv run python -m unittest discover -s tests
```

Check TTS and laptop speaker output:

```bash
python3 eleven_labs_test.py
```

Check a recorded request clip through STT and target classification:

```bash
python3 eleven_labs_stt_test.py recordings/voice_requests/strawberry_request.wav
```

Replay transcript classification, STT fixtures, and vision success fixtures:

```bash
uv run python scripts/replay_handoff_mock.py --use-stt
```

To extract fixture frames from the Hugging Face Strawberry dataset:

```bash
uv run python scripts/extract_handoff_fixtures.py \
  --download \
  --repo-id ofcourseistillloveyou/so101_recording_strawberry_num40_20260516_161110
```

## OOD Policy Wrapper

Run the current OOD wrapper with ElevenLabs OOD voice alerts:

```bash
./scripts/run_policy_with_ood.sh
```

See `ood_detection.md` and `food_handoff.md` for deeper operational details.
