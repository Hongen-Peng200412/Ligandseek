from __future__ import annotations

from pathlib import Path

import torch

from ligandseek.models.stage1_model import build_release_model


VARIANT_CHECKPOINTS = {
    "unet_raw": "TOP_epoch_04_score_0.6803.ckpt",
    "unet_diff": "TOP_epoch_05_score_0.7024.ckpt",
    "unet_full": "TOP_epoch_05_score_0.7430.ckpt",
    "ligandseek_pv": "TOP_epoch_05_score_0.7472.ckpt",
}


def load_release_checkpoint(
    variant: str,
    checkpoint_dir: str | Path,
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Strictly load a released tensor-only checkpoint, given its file or directory."""
    model = build_release_model(variant)
    ckpt_path = Path(checkpoint_dir)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / VARIANT_CHECKPOINTS[variant]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = {
        key.removeprefix("backbone."): value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("backbone.")
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(map_location)
    model.eval()
    return model
