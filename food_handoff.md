# Food Handoff Runtime

This flow waits for a hand, records a short voice request, classifies the request as
`strawberry`, `oreo`, or `marshmallow`, runs the matching policy, detects whether
the food reaches the hand, and speaks success or OOD audio through ElevenLabs.

## Setup

Create local runtime configs:

```bash
cp config/food_policies.example.json config/food_policies.json
cp config/food_handoff_vision.example.json config/food_handoff_vision.json
```

Then edit:

- `config/food_policies.json` with the three Hugging Face policy repo ids.
- `config/food_handoff_vision.json` with camera ROI and color thresholds. The
  example uses the front camera for hand entry and the side camera for placement
  success, because the side view shows the food-in-hand state more clearly.
- `.env` with `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`,
  and `ELEVENLABS_MODEL_ID` for text-to-speech. Speech-to-text uses ElevenLabs
  `scribe_v2` internally because the STT endpoint only accepts Scribe models.

The canonical target enum is `marshmallow`; the speech classifier also accepts
the spoken/transcribed misspelling `marshmellow`.

## Run

```bash
./scripts/run_food_handoff.sh
```

For local policy-path testing without voice or speech-to-text:

```bash
./scripts/run_food_handoff.sh --no-voice --no-stt --target strawberry
```

Voice can be disabled independently with `--no-voice`. Speech-to-text can be
disabled only when `--target strawberry|oreo|marshmallow` is supplied.

For end-to-end local testing without LeRobot, robot hardware, policy checkpoints,
or OOD detector files, use `--test-mode`. This mocks hand entry, policy actions,
OOD, and success while still exercising ElevenLabs STT and target/policy
selection:

```bash
./scripts/run_food_handoff.sh \
  --test-mode --no-voice \
  --test-audio recordings/voice_requests/strawberry_request.wav

./scripts/run_food_handoff.sh \
  --test-mode --no-voice \
  --test-audio recordings/voice_requests/oreo_request.wav
```

To bypass STT in the same mocked loop:

```bash
./scripts/run_food_handoff.sh --test-mode --no-voice --no-stt --target strawberry
```

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

## Logs

Expected log markers:

```text
[HAND] present
[REQUEST] transcript='Please bring me the Oreo' target=oreo
[REQUEST] target=oreo policy=org/oreo-policy task='Pick up the Oreo...'
[POLICY] starting target=oreo fps=30 duration=30s
[OOD] frame=87 score=34.219 threshold=21.080
[TASK_SUCCESS] target=oreo frame=342 confidence=0.091 target_blob_fraction=0.091
Run complete: target=oreo success=True success_frame=342 ...
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
