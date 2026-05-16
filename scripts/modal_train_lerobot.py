import os
import subprocess
import shutil
from pathlib import Path

import modal


DEFAULT_DATASET_REPO_ID = "ofcourseistillloveyou/so-101-feed-me"
DEFAULT_POLICY_REPO_ID = "ofcourseistillloveyou/so-101-feed-me-act"
DEFAULT_JOB_NAME = "so101-feed-me-act"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "hf_transfer",
        "huggingface_hub",
        "lerobot[feetech]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("so101-lerobot-training", image=image)
volume = modal.Volume.from_name("so101-lerobot-training", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 8,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/outputs": volume},
)
def train(
    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID,
    policy_repo_id: str = DEFAULT_POLICY_REPO_ID,
    pretrained_policy_repo_id: str = "",
    job_name: str = DEFAULT_JOB_NAME,
    policy_type: str = "act",
    steps: int = 1000,
    batch_size: int = 8,
    num_workers: int = 4,
    save_freq: int = 500,
    wandb_enable: bool = False,
    resume: bool = False,
):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing HF_TOKEN. Create it with: modal secret create huggingface HF_TOKEN=hf_xxx"
        )

    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)

    output_dir = Path("/outputs") / job_name
    if output_dir.exists() and not resume:
        print(f"Removing existing non-resume output directory: {output_dir}")
        shutil.rmtree(output_dir)

    cmd = [
        "lerobot-train",
        "--dataset.repo_id",
        dataset_repo_id,
        "--policy.type",
        policy_type,
        "--policy.device",
        "cuda",
        "--policy.push_to_hub",
        "true",
        "--policy.repo_id",
        policy_repo_id,
        "--output_dir",
        str(output_dir),
        "--job_name",
        job_name,
        "--steps",
        str(steps),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--save_freq",
        str(save_freq),
        "--wandb.enable",
        str(wandb_enable).lower(),
        "--resume",
        str(resume).lower(),
    ]
    if pretrained_policy_repo_id:
        cmd.extend(["--policy.pretrained_path", pretrained_policy_repo_id])

    print("Running training command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    volume.commit()


@app.local_entrypoint()
def main(
    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID,
    policy_repo_id: str = DEFAULT_POLICY_REPO_ID,
    pretrained_policy_repo_id: str = "",
    job_name: str = DEFAULT_JOB_NAME,
    policy_type: str = "act",
    steps: int = 1000,
    batch_size: int = 8,
    num_workers: int = 4,
    save_freq: int = 500,
    wandb_enable: bool = False,
    resume: bool = False,
):
    train.remote(
        dataset_repo_id=dataset_repo_id,
        policy_repo_id=policy_repo_id,
        pretrained_policy_repo_id=pretrained_policy_repo_id,
        job_name=job_name,
        policy_type=policy_type,
        steps=steps,
        batch_size=batch_size,
        num_workers=num_workers,
        save_freq=save_freq,
        wandb_enable=wandb_enable,
        resume=resume,
    )
