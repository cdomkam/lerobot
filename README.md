# SO-101 LeRobot Helpers

Operational scripts for SO-101 data collection, policy rollout, OOD detection,
and voice-gated food handoff.

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

Speech-to-text uses ElevenLabs `scribe_v2` internally. `ELEVENLABS_MODEL_ID`
is only for text-to-speech.

For robot execution, this branch expects a latest LeRobot checkout at
`vendor/lerobot-main`:

```bash
./scripts/update_lerobot_latest.sh
```

## Local Tests

Run unit tests:

```bash
PYTHONPATH=src python3 -m unittest tests/test_tts.py tests/test_food_handoff.py
uv run python -m unittest discover -s tests
```

Check the ElevenLabs TTS path and laptop speaker output:

```bash
python3 eleven_labs_test.py
```

Check one recorded request clip through ElevenLabs STT and target
classification:

```bash
python3 eleven_labs_stt_test.py recordings/voice_requests/strawberry_request.wav
python3 eleven_labs_stt_test.py recordings/voice_requests/oreo_request.wav
```

## Mocked Handoff Flow

Use `--test-mode` to test the handoff flow without connecting to the SO-101,
without loading LeRobot rollout code, and without requiring policy checkpoints
or OOD detector files. Test mode mocks hand entry, policy actions, OOD, and
success while still exercising ElevenLabs STT and target/policy selection.

```bash
./scripts/run_food_handoff.sh \
  --test-mode --no-voice \
  --test-audio recordings/voice_requests/strawberry_request.wav

./scripts/run_food_handoff.sh \
  --test-mode --no-voice \
  --test-audio recordings/voice_requests/oreo_request.wav
```

With success voice enabled:

```bash
./scripts/run_food_handoff.sh \
  --test-mode \
  --test-audio recordings/voice_requests/strawberry_request.wav
```

Bypass STT but keep the mocked handoff loop:

```bash
./scripts/run_food_handoff.sh --test-mode --no-voice --no-stt --target strawberry
```

The expected successful path logs:

```text
[HAND] mocked present
[REQUEST] transcript='Please put the strawberry in my hand' target=strawberry
[POLICY] mocked starting target=strawberry ...
[MOCK_ACTION] frame=1 target=strawberry
[TASK_SUCCESS] target=strawberry frame=3 confidence=1.000 mocked_success=true
Run complete: target=strawberry success=True ... test_mode=true
```

## Dataset Fixture Replay

Download and extract non-robot vision fixtures from a Hugging Face LeRobot
dataset:

```bash
uv run python scripts/extract_handoff_fixtures.py \
  --download \
  --repo-id ofcourseistillloveyou/so101_recording_strawberry_num40_20260516_161110
```

Replay local transcript classification and ROI success detection:

```bash
uv run python scripts/replay_handoff_mock.py
```

To include ElevenLabs STT over local audio clips:

```bash
uv run python scripts/replay_handoff_mock.py --use-stt
```

## Robot Handoff Flow

Create local runtime configs:

```bash
cp config/food_policies.example.json config/food_policies.json
cp config/food_handoff_vision.example.json config/food_handoff_vision.json
```

Then edit:

- `config/food_policies.json` with the Strawberry, Oreo, and Marshmallow policy
  repo ids.
- `config/food_handoff_vision.json` with the actual hand ROI, success ROI, and
  color thresholds for the camera setup.
- `.env` with ElevenLabs credentials.

Run with voice and STT enabled:

```bash
./scripts/run_food_handoff.sh
```

Run without speaker output:

```bash
./scripts/run_food_handoff.sh --no-voice
```

Run a selected target without STT:

```bash
./scripts/run_food_handoff.sh --no-voice --no-stt --target strawberry
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
