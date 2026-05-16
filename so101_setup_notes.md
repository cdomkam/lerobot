# SO-101 Local Setup Notes

## Arm Ports

- Leader arm: `/dev/cu.usbmodem5C4C1255491`
- Follower arm: `/dev/cu.usbmodem5C4C1268491`

## Camera Indexes

Use these OpenCV / AVFoundation camera indexes for the current setup:

- Camera index `0`: camera to use
- Camera index `1`: camera to use
- Camera index `2`: ignore for LeRobot

Fresh labeled captures were saved in:

- `captures/20260515_204753/camera_0.jpg`
- `captures/20260515_204753/camera_1.jpg`
- `captures/20260515_204753/camera_2.jpg`

## Useful Commands

Follower calibration:

```bash
uvx --from 'lerobot[feetech]' lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/cu.usbmodem5C4C1268491 \
  --robot.id=so101_follower
```

Leader calibration:

```bash
uvx --from 'lerobot[feetech]' lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/cu.usbmodem5C4C1255491 \
  --teleop.id=so101_leader
```

Teleoperation:

```bash
./scripts/teleoperate.sh
```

Record data:

```bash
./scripts/record_data.sh
```

By default, each recording run gets a timestamped dataset id like `cdomkam/so101_recording_YYYYMMDD_HHMMSS`.

Useful recording overrides:

```bash
TASK="Pick up the cube and drop it in the bin." NUM_EPISODES=10 ./scripts/record_data.sh
```

Resume an existing dataset:

```bash
DATASET_REPO_ID=cdomkam/so101_recording RESUME=true ./scripts/record_data.sh
```

Upload the latest feed-me dataset to Hugging Face:

```bash
./scripts/upload_dataset_to_hf.sh
```

The upload script defaults to:

- Source dataset: `cdomkam/so101_recording_20260515_212609`
- Target repo: `ofcourseistillloveyou/so-101-feed-me`

Train on Modal:

```bash
modal secret create huggingface HF_TOKEN=hf_xxx
./scripts/train_on_modal.sh
```

The Modal training script defaults to:

- Dataset repo: `ofcourseistillloveyou/so-101-feed-me`
- Policy repo: `ofcourseistillloveyou/so-101-feed-me-act`
- Policy type: `act`
- GPU: `A10G`
- Steps: `1000`
- Batch size: `8`

Useful training overrides:

```bash
STEPS=3000 BATCH_SIZE=16 POLICY_REPO_ID=ofcourseistillloveyou/so-101-feed-me-act-v2 ./scripts/train_on_modal.sh
```

Download/cache a trained policy:

```bash
./scripts/download_policy.sh
```

Run the trained policy on the robot:

```bash
./scripts/run_policy_on_robot.sh
```

The policy runner defaults to:

- Policy repo: `ofcourseistillloveyou/act-so101-feed-me-vai-10ep-run1`
- Follower port: `/dev/cu.usbmodem5C4C1268491`
- Cameras: `front=0`, `side=1`
- Policy device: `mps`
- Rollout dataset id prefix: `cdomkam/eval_so101_policy_rollout_...`

For a first autonomous run, keep the episode short and be ready to stop:

```bash
NUM_EPISODES=1 EPISODE_TIME_S=5 ./scripts/run_policy_on_robot.sh
```

## DAgger Strategy

The existing helper scripts use the PyPI LeRobot CLI that has been working for calibration, recording, upload, and basic policy rollout.

For live human-in-the-loop DAgger, use the side-by-side latest LeRobot checkout:

- Checkout path: `vendor/lerobot-main`
- Version in checkout: `0.5.2`
- Update command: `./scripts/update_lerobot_latest.sh`

Run v0.5.2 HIL/DAgger with leader handoff:

```bash
NUM_EPISODES=5 ./scripts/hil_dagger_rollout_latest.sh
```

DAgger controls in v0.5.2:

- `space`: pause/resume policy
- `tab`: toggle human correction
- `enter`: upload/push on demand

By default this records correction windows only. To record autonomous segments too:

```bash
RECORD_AUTONOMOUS=true NUM_EPISODES=5 ./scripts/hil_dagger_rollout_latest.sh
```

The HIL/DAgger script defaults to `FPS=30` because the current OpenCV cameras reject `FPS=15`, and it uses dataset names like `cdomkam/rollout_hil_so101_feed_me_YYYYMMDD_HHMMSS` because v0.5.2 requires rollout datasets to start with `rollout_`.

The older local LeRobot path does not let the leader arm override a policy inside the same `lerobot-record` control loop: if `policy` is provided, policy actions are used. For that older CLI, use a round-based DAgger loop instead.

1. Run the current policy briefly and watch where it fails:

```bash
FPS=15 NUM_EPISODES=1 EPISODE_TIME_S=5 ./scripts/run_policy_on_robot.sh
```

2. Initialize a local aggregate dataset from the original demos:

```bash
./scripts/dagger_init_dataset.sh
```

3. Reset the scene to failure-like states and record expert correction episodes with the leader:

```bash
NUM_EPISODES=5 ./scripts/dagger_collect_corrections.sh
```

4. Upload the aggregate DAgger dataset:

```bash
./scripts/dagger_upload_dataset.sh
```

5. Fine-tune the next policy on Modal from the previous policy:

```bash
./scripts/dagger_train_on_modal.sh
```

Default DAgger ids:

- Base dataset: `ofcourseistillloveyou/so-101-feed-me`
- Local aggregate dataset: `cdomkam/so101_feed_me_dagger_r1`
- Uploaded aggregate dataset: `ofcourseistillloveyou/so-101-feed-me-dagger-r1`
- Previous policy: `ofcourseistillloveyou/act-so101-feed-me-vai-10ep-run1`
- Next policy: `ofcourseistillloveyou/act-so101-feed-me-dagger-r1`
