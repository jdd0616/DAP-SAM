"""Train the supervised dual-branch model on RETOUCH."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from sam_lora_image_encoder_prompt import LoRA_Sam
from segment_anything import sam_model_registry
from trainer_dual_branch import (
    trainer_dualbrunch_onlylabl_normalassisted,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, help="RETOUCH dataset root")
    p.add_argument("--checkpoint", required=True, help="Base SAM checkpoint")
    p.add_argument("--output", default="runs/dual_branch")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--warmup-iterations", type=int, default=4000)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--coe", type=float, default=1.5)
    p.add_argument("--coe2", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    cli = parse_args()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=True)

    model, embedding_size = sam_model_registry[
        "vit_b_dualmask_same_prompt_class_random_large"
    ](
        image_size=cli.image_size,
        num_classes=1,
        checkpoint=cli.checkpoint,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    model = LoRA_Sam(model, rank=4).to(device)

    args = argparse.Namespace(
        dataset="RETOUCH",
        root_path=cli.data_root,
        base_lr=cli.learning_rate,
        batch_size=cli.batch_size,
        max_epochs=cli.epochs,
        stop_epoch=cli.epochs,
        img_size=cli.image_size,
        num_classes=1,
        ratio=cli.ratio,
        promptmode="mask",
        seed=cli.seed,
        warm_iter=cli.warmup_iterations,
        warmup=True,
        warmup_period=250,
        AdamW=True,
        T=cli.temperature,
        coe=cli.coe,
        coe2=cli.coe2,
    )
    trainer_dualbrunch_onlylabl_normalassisted(
        args, model, str(output), True, embedding_size * 4, device
    )


if __name__ == "__main__":
    main()
