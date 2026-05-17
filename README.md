# SO-101 Voice-Gated Food Handoff

This branch contains the SO-101 helpers plus a voice-gated food handoff flow for
Strawberry, Oreo, and Marshmallow policies.

The handoff flow is the same in mock mode and robot mode:

1. Wait for a hand in the camera frame.
2. Record a short user request.
3. Transcribe with ElevenLabs STT and classify the target policy.
4. Run the selected food handoff policy.
5. Detect success or OOD.
6. Speak the outcome with ElevenLabs TTS.
7. Pause so the hand can leave the frame, then repeat.

Robot mode repeats until interrupted by default. Test mode runs one cycle by
default, so local commands finish; use `--max-cycles N` to test multiple cycles
or `--max-cycles 0` for an unlimited loop.

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

For robot execution, this branch expects a latest LeRobot checkout at
`vendor/lerobot-main`:

```bash
./scripts/update_lerobot_latest.sh
```

## Mocked Local Test

Use `--test-mode` to run the handoff flow without connecting to SO-101, without
loading LeRobot rollout code, and without requiring policy checkpoints or OOD
detector files. It still exercises ElevenLabs STT, target classification, policy
selection, success/OOD handling, and ElevenLabs TTS by default.

Run one mocked cycle from a recorded voice request:

```bash
./scripts/run_food_handoff.sh \
  --test-mode \
  --test-audio recordings/voice_requests/strawberry_request.wav
```

Other recorded request clips:

```bash
./scripts/run_food_handoff.sh \
  --test-mode \
  --test-audio recordings/voice_requests/oreo_request.wav

./scripts/run_food_handoff.sh \
  --test-mode \
  --test-audio recordings/voice_requests/marshmallow_request.wav
```

Run multiple mocked cycles with speech input and output enabled. The reset pause
gives the previous hand time to leave the frame:

```bash
./scripts/run_food_handoff.sh \
  --test-mode \
  --test-audio recordings/voice_requests/oreo_request.wav \
  --max-cycles 3 --reset-pause-s 7
```

Debug-only overrides, when you explicitly do not want audio:

```bash
./scripts/run_food_handoff.sh --test-mode --no-voice --test-audio recordings/voice_requests/oreo_request.wav
./scripts/run_food_handoff.sh --test-mode --no-voice --no-stt --target marshmallow
```

Expected successful mock logs include:

```text
[CYCLE] start cycle=1
[HAND] mocked present
[REQUEST] test_audio=... transcript='Please put the strawberry in my hand' target=strawberry
[REQUEST] target=strawberry policy=...
[POLICY] mocked starting target=strawberry ...
[MOCK_ACTION] frame=1 target=strawberry
[TASK_SUCCESS] target=strawberry frame=3 confidence=1.000 mocked_success=true
Run complete: target=strawberry success=True ... test_mode=true
[CYCLE] complete cycle=1 outcome=success
```

## Robot Handoff

Review the checked-in runtime configs:

- `config/food_policies.json` with the Strawberry, Oreo, and Marshmallow policy
  repo ids and any per-target OOD detector paths.
- `config/food_handoff_vision.json` with the scene-camera hand ROI, success ROI,
  and color thresholds for the camera setup.
- `.env` with ElevenLabs credentials.

Run the same flow against the robot by removing `--test-mode`:

```bash
./scripts/run_food_handoff.sh
```

That command uses speech input and speech output, loops until Ctrl-C, and waits
`RESET_PAUSE_S=7` seconds between cycles so the previous hand can move out of
frame.

Run one real handoff:

```bash
./scripts/run_food_handoff.sh --max-cycles 1
```

Adjust the hand reset pause:

```bash
./scripts/run_food_handoff.sh --reset-pause-s 10
```

Debug-only overrides are available as `--no-voice` and `--no-stt --target ...`,
but the normal path uses speech input and output.

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

Run the current OOD wrapper without voice:

```bash
DURATION=15 OOD_LOG_IN_DIST_EVERY_N=30 ./scripts/run_policy_with_ood.sh --no-voice
```

Run with ElevenLabs OOD voice alerts:

```bash
./scripts/run_policy_with_ood.sh
```

See `ood_detection.md` and `food_handoff.md` for deeper operational details.
