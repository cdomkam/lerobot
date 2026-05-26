# SO-101 Voice-Gated Food Handoff

This branch contains the SO-101 helpers plus a voice-gated food handoff flow for
Strawberry, Oreo, and Marshmallow policies.

The production handoff loop is ChatGPT Realtime driven:

1. ChatGPT Realtime listens to the microphone, transcribes the user request,
   and decides when to call the `run_handoff` tool.
2. The deterministic handoff control loop speaks the fetching line, immediately
   runs the selected policy, and checks success.
3. GPT-5.4-nano vision checks side-camera success on background threads while
   the policy runs.
4. All spoken robot responses use the existing ElevenLabs TTS flow and phrase
   inventories.

When OpenAI success confirmation is enabled, side-camera success checks run
during the policy loop so a correct delivery can be accepted even if the local
color/ROI success detector misses it.

## Production Run

The default production run is a continuous ChatGPT Realtime orchestrated,
multi-policy robot loop:

```bash
./scripts/run_realtime_food_handoff.sh
```

ChatGPT Realtime listens and calls `run_handoff(target)` when the user requests
Strawberry, Oreo, or Marshmallow. The tool runs one complete deterministic
handoff cycle, including ElevenLabs fetching/success speech, policy execution,
side-camera success checks, and reset.

OpenAI success confirmation is enabled by default and is required for automatic
robot success.

## Core Run Paths

- **Robot, production:** `./scripts/run_realtime_food_handoff.sh`
  listens with ChatGPT Realtime, uses the `run_handoff` tool, runs the robot
  handoff loop, checks success with GPT-5.4-nano, and speaks through ElevenLabs.
- **Robot, lower-level debug:** `./scripts/run_food_handoff.sh --no-stt --target strawberry --max-cycles 1`
  bypasses Realtime and forces one food for one robot cycle.
- **No robot, one recorded request:** `./scripts/run_food_handoff.sh --test-mode --test-audio recordings/voice_requests/strawberry_request.wav --max-cycles 1`
  exercises request classification, policy selection, mocked policy/success, and
  ElevenLabs output without connecting to SO-101.
- **Realtime tool dry run:** `python3 scripts/gpt_realtime/run_clanker.py --dry-run-tool strawberry`
  validates Realtime tool config and target normalization without opening the
  websocket or touching the robot.

## Full Loop

The runtime is `scripts/run_realtime_food_handoff.sh`, which launches the
Realtime orchestrator in `scripts/gpt_realtime/run_clanker.py`. The Realtime
tool calls `scripts/run_food_handoff.sh --no-stt --target <food> --max-cycles 1`
for the actual robot cycle.

- **Speech request:** ChatGPT Realtime transcribes the microphone stream and
  chooses the target through a `run_handoff` tool call. Realtime model output is
  text-only and is not spoken; robot speech is owned by the local ElevenLabs
  control/tool paths.
- **Unsupported requests:** if the user asks for anything outside Strawberry,
  Oreo, or Marshmallow, Realtime calls `unsupported_item_requested` and the local
  control path speaks a randomized unavailable-item phrase through ElevenLabs.
- **Policy trigger:** once Realtime calls `run_handoff(target)`, the lower-level
  runner speaks the fetching phrase and starts the selected policy immediately.
- **Policy selection:** `config/food_policies.json` maps the classified food to
  a Hugging Face policy repo, task prompt, and per-target OOD detector.
- **Policy execution:** the selected LeRobot policy sends actions to SO-101 for
  `DURATION` seconds or until success is detected. After `DURATION`, the robot
  stops sending actions and the loop keeps checking side-camera success for
  `OPENAI_SUCCESS_GRACE_S` seconds before returning timeout.
- **Neutral reset:** after robot-motion terminal states such as success, visual
  OOD failure, timeout, or policy failure, the runner attempts to move SO-101 to
  the local neutral pose from `config/robot_neutral.json` before the next voice
  request can trigger a policy. Unsupported or unclassified text requests skip
  physical reset because no robot action ran.
- **OOD scoring:** enabled by default. The ACT backbone encoder scores the
  configured `OOD_CAMERA` against the selected target's detector `.npz` from
  `config/food_policies.json`. OOD events are logged and can trigger ElevenLabs
  voice alerts. After `OOD_FAILURE_AFTER_N` OOD detections, default `3`, the
  cycle returns `ood_failure`, speaks the OOD phrase, stops the current policy,
  records the failed action, and runs the normal reset pause before the next
  voice request.
- **Success detection:** the side camera, named `side`, maps to
  `CAMERA_SIDE_INDEX=0` and is the scene camera in the current setup. When
  OpenAI success is enabled, the loop sends an ordered short sequence of recent
  side-camera frames to `gpt-5.4-nano` every `OPENAI_SUCCESS_EVERY_N` frames and
  immediately on any local ROI/color success candidate. Success is accepted only
  when the sequence shows the robot gripper near the hand placing or releasing
  the correct target food, and the newest frame shows that food in the user's
  hand, not still solely held by the gripper or sitting on the tray/table. It is
  not success if the robot is stalled, absent, holding the wrong item, holding
  the item away from the hand, or if the user grabs the item without robot
  placement. Confidence must be at least `OPENAI_SUCCESS_MIN_CONFIDENCE` from
  `.env` (default `0.70`). Local OpenCV ROI/color detection is only a candidate
  trigger/log signal; it is not accepted as success when OpenAI is disabled or
  when OpenAI confirmation fails.
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

`ELEVENLABS_MODEL_ID` configures text-to-speech output. Production speech input
uses ChatGPT Realtime rather than ElevenLabs STT.

OpenAI success confirmation uses the same `.env` file:

```bash
OAI_KEY=...
OPENAI_REALTIME_MODEL=gpt-realtime-2
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

Use these four views for normal testing. Realtime robot runs listen through
ChatGPT Realtime and speak through ElevenLabs. Lower-level mock runs use recorded
clips and the deterministic handoff runner directly.

### 1. Test One Voice Request, No Robot

Use lower-level `--test-mode` with a recorded request clip. This does not
connect to SO-101, does not load LeRobot rollout code, and does not require
policy checkpoints or OOD detector files. It still exercises recorded speech
classification, policy selection, mocked success/OOD handling, and ElevenLabs
TTS.

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
- Confirm `config/food_handoff_vision.json` uses the scene camera for success:
  `success.camera_name=side`.
- Create local ignored `config/robot_neutral.json` with the operator-approved
  neutral pose for this robot:
  ```json
  {
    "action": {
      "shoulder_pan.pos": 0.0,
      "shoulder_lift.pos": 0.0,
      "elbow_flex.pos": 0.0,
      "wrist_flex.pos": 0.0,
      "wrist_roll.pos": 0.0,
      "gripper.pos": 50.0
    }
  }
  ```
- Keep power/USB within reach for the first robot run.

Run the Realtime robot loop and ask for any configured food, for example
"strawberry". Stop after the cycle with Enter or Ctrl-C:

```bash
./scripts/run_realtime_food_handoff.sh
```

For robot success testing, keep OpenAI success confirmation enabled.

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

Run the Realtime robot loop and request Strawberry during one tool call, then
Marshmallow during the next:

```bash
./scripts/run_realtime_food_handoff.sh
```

For robot success testing, keep OpenAI success confirmation enabled.

The production robot loop is the same command without `--max-cycles`; stop it
with Ctrl-C:

```bash
./scripts/run_realtime_food_handoff.sh
```

Expected successful logs include:

```text
[user] Please put the strawberry in my hand
[handoff] Started strawberry: ...
[CYCLE] start cycle=1
[REQUEST] target=strawberry policy=...
[TTS] queue wait=True phrase='...Strawberry...'
[POLICY] starting target=strawberry ...
[OPENAI_SUCCESS] frame=... sequence_frames=5 success=True ... robot_placing=True ...
[TASK_SUCCESS] target=strawberry frame=... source=openai opencv_score=...
[TTS] queue wait=True phrase='...Strawberry...'
[CYCLE] complete cycle=1 outcome=success
[handoff] Finished strawberry exit_code=0 ...
```

Lower-level debug override: `./scripts/run_food_handoff.sh --no-stt --target
strawberry --max-cycles 1` bypasses Realtime and forces one target. The normal
path uses Realtime speech input/tool orchestration and ElevenLabs speech output.

Useful robot-mode environment knobs:

```bash
CAMERA_SIDE_INDEX=0              # scene camera used for success
CAMERA_FRONT_INDEX=1             # on-robot/front camera, used by OOD by default
OOD_ENABLED=true                 # default; requires per-target detector files
OOD_CAMERA=front
OOD_FAILURE_AFTER_N=3            # OOD detections before failing/resetting cycle
OOD_TTS_EVERY_N=3                # non-terminal OOD voice alert cadence
OPENAI_SUCCESS_ENABLED=true      # default; required for automatic robot success
OPENAI_SUCCESS_CONFIG_PATH=.env
OPENAI_SUCCESS_EVERY_N=15        # minimum frames between OpenAI confirmation calls
OPENAI_SUCCESS_SEQUENCE_FRAMES=5 # number of side-camera frames sent per check
OPENAI_SUCCESS_SEQUENCE_STRIDE=5 # frame spacing inside the success sequence
OPENAI_SUCCESS_GRACE_S=15        # post-action window for pending/final success checks
ROBOT_NEUTRAL_RESET_ENABLED=true
ROBOT_NEUTRAL_CONFIG=config/robot_neutral.json
ROBOT_NEUTRAL_RESET_DURATION_S=3.0
ROBOT_NEUTRAL_RESET_FPS=30
DURATION=30
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
