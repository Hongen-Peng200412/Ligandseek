"""Run four released models from experimental density and optional prepared receptor inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ligandseek.inference.postprocess import write_prediction_outputs
from ligandseek.inference.sliding_window import predict_volume
from ligandseek.inference.volume_io import load_mrc_volume
from ligandseek.models.checkpoint import load_release_checkpoint
from ligandseek.parse.receptor_features import load_receptor_atom_features, load_receptor_from_structure
from ligandseek.parse.receptor_geometry import build_hardmask_from_world_coordinates


DENSITY_CHANNELS = {
    "unet_raw": ["exp_clipnorm_nopost"],
    "unet_diff": ["exp_clipnorm_nopost", "diff_clipnorm_nopost"],
    "unet_full": ["all"],
    "ligandseek_pv": ["all"],
}


def run_prediction(
    model_name: str,
    map_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    receptor_path: str | Path | None,
    sim_path: str | Path | None,
    atom_feature_path: str | Path | None,
    receptor_mode: str,
    threshold: float | None,
    device: str,
    batch_size: int,
    seed: int,
):
    """Write probability, filtered mask, instances, candidate coordinates, and run/grid metadata.

    Input density and prepared simulation are independently Fourier-resampled to the paper grid.
    Receptor coordinates are XYZ in Angstrom; all output volumes use ZYX array order.
    raw supports map-only prediction; a supplied receptor additionally clears occupied voxels.
    Precomputed features can replace CIF feature extraction, but are never required for raw.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if model_name != "unet_raw" and sim_path is None:
        raise ValueError(f"{model_name} requires a prepared receptor-only --sim map.")
    if model_name in {"unet_full", "ligandseek_pv"} and receptor_path is None and atom_feature_path is None:
        raise ValueError(f"{model_name} requires --receptor or --atom-features.")

    print(f"Loading experimental density: {Path(map_path).name}", flush=True)
    exp, voxel_size, origin = load_mrc_volume(map_path)
    sim = None
    if sim_path is not None and model_name != "unet_raw":
        print(f"Loading prepared simulation: {Path(sim_path).name}", flush=True)
        sim, sim_spacing, sim_origin = load_mrc_volume(sim_path)
        if sim.shape != exp.shape or not np.allclose(sim_spacing, voxel_size, rtol=1e-5, atol=1e-6) or not np.allclose(sim_origin, origin, rtol=1e-5, atol=1e-4):
            raise ValueError("Experimental and simulated maps must share the resampled shape, spacing, and origin.")

    receptor = None
    if atom_feature_path is not None:
        receptor = load_receptor_atom_features(atom_feature_path)
    elif receptor_path is not None:
        print(f"Extracting receptor atoms: {Path(receptor_path).name}", flush=True)
        receptor = load_receptor_from_structure(receptor_path, compute_density=True, error_dir=output)
    hardmask = None if receptor is None else build_hardmask_from_world_coordinates(
        atom_coords_world=receptor.coords,
        box_origin_world=origin,
        voxel_size_world=voxel_size,
        box_shape_zyx=np.asarray(exp.shape, dtype=np.int64),
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Loading {model_name}; grid={tuple(exp.shape)} device={device}", flush=True)
    model = load_release_checkpoint(model_name, checkpoint_path, map_location=device)
    # raw/diff density input stays unchanged; receptor coordinates retain the paper output constraint.
    probability = predict_volume(
        model=model,
        exp_volume=exp,
        sim_volume=sim,
        density_channels=DENSITY_CHANNELS[model_name],
        device=device,
        window_size=80,
        stride=40,
        batch_size=batch_size,
        core_offset=5,
        gaussian_sigma_ratio=0.5,
        atom_coords_world=None if receptor is None else receptor.coords,
        atom_features=None if receptor is None else receptor.features,
        origin_xyz=origin,
        voxel_size_xyz=voxel_size,
        atom_buffer_radius_world=8.0,
    )
    result = write_prediction_outputs(probability, origin, voxel_size, hardmask, model_name, receptor_mode, output, threshold=threshold)
    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "seed": seed,
        "device": device,
        "batch_size": batch_size,
        "grid_shape_zyx": list(exp.shape),
        "grid_preprocessing": "paper_fourier_1A",
        "window_size": 80,
        "stride": 40,
        "core_offset": 5,
        "gaussian_sigma_ratio": 0.5,
        "atom_buffer_radius_angstrom": 8.0,
        "receptor_input": "map_only" if receptor is None else ("precomputed_features" if atom_feature_path is not None else "structure"),
        "receptor_atom_count": 0 if receptor is None else int(receptor.coords.shape[0]),
        "torch_version": str(torch.__version__),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved prediction: {output}", flush=True)
    return result
