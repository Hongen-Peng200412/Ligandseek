"""Predict a volume with paper window geometry, Gaussian stitching and receptor masking.

predict_volume returns a float32 (Z, Y, X) probability array on the input grid.
Atoms use world XYZ coordinates in angstroms; a map-only call has no receptor mask.
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import Iterable

import numpy as np

import torch

from ligandseek.datasets.density_channel_builder import (
    DensityChannelConfig,
    build_density_channels,
)


@dataclass(frozen=True)
class WindowSpec:

    position_zyx: tuple[int, int, int]

    shape_zyx: tuple[int, int, int]


DEFAULT_GAUSSIAN_SIGMA_RATIO = 0.5
DEFAULT_ATOM_BUFFER_RADIUS_WORLD = 8.0


def compute_window_starts(dim: int, window_size: int, stride: int) -> list[int]:

    if dim <= window_size:

        return [0]

    starts = list(range(0, dim - window_size + 1, stride))

    last = dim - window_size

    if starts[-1] != last:

        starts.append(last)

    return starts


def iter_window_specs(
    shape_zyx: tuple[int, int, int], window_size: int, stride: int
) -> Iterable[WindowSpec]:

    z_starts = compute_window_starts(int(shape_zyx[0]), window_size, stride)

    y_starts = compute_window_starts(int(shape_zyx[1]), window_size, stride)

    x_starts = compute_window_starts(int(shape_zyx[2]), window_size, stride)

    for z0 in z_starts:

        for y0 in y_starts:

            for x0 in x_starts:

                yield WindowSpec(
                    position_zyx=(z0, y0, x0),
                    shape_zyx=(
                        min(window_size, int(shape_zyx[0]) - z0),
                        min(window_size, int(shape_zyx[1]) - y0),
                        min(window_size, int(shape_zyx[2]) - x0),
                    ),
                )


def _pad_to_window(x: np.ndarray, window_size: int) -> np.ndarray:

    pad_width = [(0, max(0, window_size - int(size))) for size in x.shape]

    if any(width[1] > 0 for width in pad_width):

        return np.pad(x, pad_width, mode="constant")

    return x


def _empty_atom_batch(
    batch_size: int, window_size: int, channels: int, device: torch.device
) -> dict[str, torch.Tensor | list]:

    return {
        "voxel_grid": torch.empty(
            (batch_size, channels, window_size, window_size, window_size),
            dtype=torch.float32,
            device=device,
        ),
        "voxel_label": torch.zeros(
            (batch_size, window_size, window_size, window_size),
            dtype=torch.long,
            device=device,
        ),
        "hardmask": torch.zeros(
            (batch_size, window_size, window_size, window_size),
            dtype=torch.long,
            device=device,
        ),
        "box_origin_world": torch.zeros(
            (batch_size, 3), dtype=torch.float32, device=device
        ),
        "voxel_size_world": torch.ones(
            (batch_size, 3), dtype=torch.float32, device=device
        ),
        "box_shape_zyx": torch.full(
            (batch_size, 3), window_size, dtype=torch.long, device=device
        ),
        "atom_coord_world": torch.empty((0, 3), dtype=torch.float32, device=device),
        "atom_coord_local_voxel": torch.empty(
            (0, 3), dtype=torch.float32, device=device
        ),
        "atom_coord_centered_world": torch.empty(
            (0, 3), dtype=torch.float32, device=device
        ),
        "atom_feat": torch.empty((0, 49), dtype=torch.float32, device=device),
        "atom_label": torch.empty((0,), dtype=torch.long, device=device),
        "atom_is_in_core_box": torch.empty((0,), dtype=torch.bool, device=device),
        "atom_global_indices": torch.empty((0,), dtype=torch.long, device=device),
        "atom_batch_index": torch.empty((0,), dtype=torch.long, device=device),
        "atom_counts": torch.zeros((batch_size,), dtype=torch.long, device=device),
        "atom_offsets": torch.zeros((batch_size,), dtype=torch.long, device=device),
        "sample_name": ["window"] * batch_size,
        "pdb_id": ["map"] * batch_size,
        "class_name": ["inference"] * batch_size,
        "region_id": [0] * batch_size,
        "is_center_box": [False] * batch_size,
    }


def _build_gaussian_weight(
    box_shape_zyx: tuple[int, int, int], gaussian_sigma_ratio: float
) -> np.ndarray:

    """Build the Gaussian weights used by the paper inference pipeline."""

    if gaussian_sigma_ratio <= 0:

        raise ValueError("gaussian_sigma_ratio must be greater than zero.")

    depth, height, width = (int(v) for v in box_shape_zyx)

    zz, yy, xx = np.meshgrid(
        np.arange(depth, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    sigma = float(gaussian_sigma_ratio) * float(max(depth, height, width))

    distance_squared = (
        (zz - 0.5 * float(depth - 1)) ** 2
        + (yy - 0.5 * float(height - 1)) ** 2
        + (xx - 0.5 * float(width - 1)) ** 2
    )

    return np.exp(-0.5 * distance_squared / (sigma * sigma)).astype(np.float32)


def _merge_probability(
    value_sum: np.ndarray,
    weight_sum: np.ndarray,
    box_prob: np.ndarray,
    spec: WindowSpec,
    gaussian_weight: np.ndarray,
    core_offset: int = 0,
) -> None:
    # Crop only window faces inside the volume; preserve its outer boundary.
    box_slices = []
    full_slices = []
    for start, box_size, full_size in zip(
        spec.position_zyx, box_prob.shape, value_sum.shape
    ):
        local_end = min(int(box_size), int(full_size) - int(start))
        local_start = int(core_offset) if start > 0 else 0
        if start + box_size < full_size:
            local_end = min(local_end, int(box_size) - int(core_offset))
        if local_start >= local_end:
            raise ValueError("core_offset leaves no valid window merge region.")
        box_slices.append(slice(local_start, local_end))
        full_slices.append(slice(int(start) + local_start, int(start) + local_end))
    sl_box = tuple(box_slices)
    sl_full = tuple(full_slices)

    weight = gaussian_weight[sl_box]

    value_sum[sl_full] += box_prob[sl_box].astype(np.float32, copy=False) * weight

    weight_sum[sl_full] += weight


def _build_receptor_mask(
    atom_coords_world: np.ndarray | None,
    origin_xyz: np.ndarray | tuple[float, float, float] | None,
    voxel_size_xyz: np.ndarray | tuple[float, float, float] | None,
    spec: WindowSpec,
    window_size: int,
) -> np.ndarray | None:

    if atom_coords_world is None:

        return None

    coords_world = np.asarray(atom_coords_world, dtype=np.float32)

    if coords_world.size == 0:

        return np.zeros((window_size, window_size, window_size), dtype=bool)

    origin = (
        np.zeros(3, dtype=np.float32)
        if origin_xyz is None
        else np.asarray(origin_xyz, dtype=np.float32)
    )

    voxel_size = (
        np.ones(3, dtype=np.float32)
        if voxel_size_xyz is None
        else np.asarray(voxel_size_xyz, dtype=np.float32)
    )

    z0, y0, x0 = spec.position_zyx
    box_origin = origin + np.asarray([x0, y0, z0], dtype=np.float32) * voxel_size
    box_max = box_origin + np.asarray([window_size] * 3, dtype=np.float32) * voxel_size
    core = np.all(coords_world >= box_origin[None, :], axis=1)
    core &= np.all(coords_world < box_max[None, :], axis=1)
    local_xyz = (coords_world[core] - box_origin[None, :]) / voxel_size[None, :]

    ijk = np.floor(local_xyz).astype(np.int64, copy=False)

    valid = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < window_size)
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < window_size)
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < window_size)
    )

    mask = np.zeros((window_size, window_size, window_size), dtype=bool)

    if valid.any():

        xyz = ijk[valid]

        mask[xyz[:, 2], xyz[:, 1], xyz[:, 0]] = True

    return mask


def _build_receptor_occupancy_mask(
    atom_coords_world: np.ndarray | None,
    origin_xyz: np.ndarray | tuple[float, float, float] | None,
    voxel_size_xyz: np.ndarray | tuple[float, float, float] | None,
    shape_zyx: tuple[int, int, int],
) -> np.ndarray | None:

    if atom_coords_world is None:

        return None

    coords_world = np.asarray(atom_coords_world, dtype=np.float32)

    mask = np.zeros(shape_zyx, dtype=bool)

    if coords_world.size == 0:

        return mask

    origin = (
        np.zeros(3, dtype=np.float32)
        if origin_xyz is None
        else np.asarray(origin_xyz, dtype=np.float32)
    )

    voxel_size = (
        np.ones(3, dtype=np.float32)
        if voxel_size_xyz is None
        else np.asarray(voxel_size_xyz, dtype=np.float32)
    )

    coords_voxel_xyz = (coords_world - origin[None, :]) / voxel_size[None, :]

    voxel_xyz = np.floor(coords_voxel_xyz).astype(np.int64, copy=False)

    depth, height, width = (int(value) for value in shape_zyx)

    valid = (
        (voxel_xyz[:, 0] >= 0)
        & (voxel_xyz[:, 0] < width)
        & (voxel_xyz[:, 1] >= 0)
        & (voxel_xyz[:, 1] < height)
        & (voxel_xyz[:, 2] >= 0)
        & (voxel_xyz[:, 2] < depth)
    )

    if valid.any():

        selected = voxel_xyz[valid]

        mask[selected[:, 2], selected[:, 1], selected[:, 0]] = True

    return mask


def predict_volume(
    model: torch.nn.Module,
    exp_volume: np.ndarray,
    sim_volume: np.ndarray | None,
    density_channels: list[str],
    device: str | torch.device = "cpu",
    window_size: int = 80,
    stride: int = 40,
    batch_size: int = 1,
    gaussian_sigma_ratio: float = DEFAULT_GAUSSIAN_SIGMA_RATIO,
    atom_coords_world: np.ndarray | None = None,
    atom_features: np.ndarray | None = None,
    origin_xyz: np.ndarray | tuple[float, float, float] | None = None,
    voxel_size_xyz: np.ndarray | tuple[float, float, float] | None = None,
    atom_buffer_radius_world: float = DEFAULT_ATOM_BUFFER_RADIUS_WORLD,
    output_mask_atom_coords_world: np.ndarray | None = None,
    core_offset: int = 5,
) -> np.ndarray:

    device_obj = torch.device(device)

    model.to(device_obj)

    model.eval()

    exp = _pad_to_window(np.asarray(exp_volume, dtype=np.float32), window_size)

    sim = (
        None
        if sim_volume is None
        else _pad_to_window(np.asarray(sim_volume, dtype=np.float32), window_size)
    )

    shape_zyx = tuple(int(v) for v in exp.shape)

    receptor_occupancy_mask = _build_receptor_occupancy_mask(
        atom_coords_world=(
            atom_coords_world
            if output_mask_atom_coords_world is None
            else output_mask_atom_coords_world
        ),
        origin_xyz=origin_xyz,
        voxel_size_xyz=voxel_size_xyz,
        shape_zyx=shape_zyx,
    )

    specs = list(iter_window_specs(shape_zyx, window_size=window_size, stride=stride))

    value_sum = np.zeros(shape_zyx, dtype=np.float32)

    weight_sum = np.zeros(shape_zyx, dtype=np.float32)

    cfg = DensityChannelConfig(enabled_channels=list(density_channels))

    gaussian_weights: dict[tuple[int, int, int], np.ndarray] = {}

    for start in range(0, len(specs), batch_size):

        group = specs[start : start + batch_size]

        channel_blocks: list[np.ndarray] = []

        receptor_masks: list[np.ndarray | None] = []

        for spec in group:

            z0, y0, x0 = spec.position_zyx

            exp_box = exp[
                z0 : z0 + window_size, y0 : y0 + window_size, x0 : x0 + window_size
            ]

            sim_box = None

            if sim is not None:

                sim_box = sim[
                    z0 : z0 + window_size, y0 : y0 + window_size, x0 : x0 + window_size
                ]

            receptor_mask = _build_receptor_mask(
                atom_coords_world=atom_coords_world,
                origin_xyz=origin_xyz,
                voxel_size_xyz=voxel_size_xyz,
                spec=spec,
                window_size=window_size,
            )

            receptor_masks.append(receptor_mask)

            channel_blocks.append(
                build_density_channels(
                    exp_box, sim_box, cfg, receptor_mask=receptor_mask
                )
            )

        voxel_grid = torch.from_numpy(np.stack(channel_blocks, axis=0)).to(
            device=device_obj
        )

        batch = _empty_atom_batch(
            batch_size=len(group),
            window_size=window_size,
            channels=int(voxel_grid.shape[1]),
            device=device_obj,
        )

        batch["voxel_grid"] = voxel_grid

        _fill_window_metadata(
            batch=batch,
            specs=group,
            receptor_masks=receptor_masks,
            origin_xyz=origin_xyz,
            voxel_size_xyz=voxel_size_xyz,
            device=device_obj,
        )

        if atom_coords_world is not None and atom_features is not None:

            _fill_atom_batch(
                batch=batch,
                specs=group,
                atom_coords_world=atom_coords_world,
                atom_features=atom_features,
                origin_xyz=origin_xyz,
                voxel_size_xyz=voxel_size_xyz,
                atom_buffer_radius_world=atom_buffer_radius_world,
                device=device_obj,
            )

        with torch.no_grad():

            outputs = model(batch)

            logits = outputs["voxel_logits_ligand"]

            probs = (
                torch.sigmoid(logits[:, 0]).detach().cpu().numpy().astype(np.float32)
            )

        for spec, prob in zip(group, probs):

            probability_shape = tuple(int(v) for v in prob.shape)

            if probability_shape not in gaussian_weights:

                gaussian_weights[probability_shape] = _build_gaussian_weight(
                    probability_shape, gaussian_sigma_ratio
                )

            _merge_probability(
                value_sum=value_sum,
                weight_sum=weight_sum,
                box_prob=prob,
                spec=spec,
                gaussian_weight=gaussian_weights[probability_shape],
                core_offset=core_offset,
            )

    pred = np.divide(
        value_sum, weight_sum, out=np.zeros_like(value_sum), where=weight_sum > 0
    )

    pred = pred[
        : exp_volume.shape[0], : exp_volume.shape[1], : exp_volume.shape[2]
    ].astype(np.float32)

    if receptor_occupancy_mask is not None:

        receptor_occupancy_mask = receptor_occupancy_mask[
            : exp_volume.shape[0], : exp_volume.shape[1], : exp_volume.shape[2]
        ]

        pred[receptor_occupancy_mask] = 0.0

    return pred


def _fill_window_metadata(
    batch: dict,
    specs: list[WindowSpec],
    receptor_masks: list[np.ndarray | None],
    origin_xyz: np.ndarray | tuple[float, float, float] | None,
    voxel_size_xyz: np.ndarray | tuple[float, float, float] | None,
    device: torch.device,
) -> None:

    origin = (
        np.zeros(3, dtype=np.float32)
        if origin_xyz is None
        else np.asarray(origin_xyz, dtype=np.float32)
    )

    voxel_size = (
        np.ones(3, dtype=np.float32)
        if voxel_size_xyz is None
        else np.asarray(voxel_size_xyz, dtype=np.float32)
    )

    starts_zyx = np.asarray([spec.position_zyx for spec in specs], dtype=np.float32)

    box_origin = origin[None, :] + starts_zyx[:, [2, 1, 0]] * voxel_size[None, :]

    batch["box_origin_world"] = torch.from_numpy(box_origin.astype(np.float32)).to(
        device=device
    )

    batch["voxel_size_world"] = torch.from_numpy(
        np.repeat(voxel_size[None, :], len(specs), axis=0).astype(np.float32)
    ).to(device=device)

    batch["box_shape_zyx"] = torch.from_numpy(
        np.asarray([spec.shape_zyx for spec in specs], dtype=np.int64)
    ).to(device=device)

    hardmask = np.stack(
        [
            (
                np.zeros(tuple(int(v) for v in spec.shape_zyx), dtype=np.int64)
                if receptor_mask is None
                else receptor_mask.astype(np.int64, copy=False)
            )
            for spec, receptor_mask in zip(specs, receptor_masks)
        ],
        axis=0,
    )

    batch["hardmask"] = torch.from_numpy(hardmask).to(device=device)


def _fill_atom_batch(
    batch: dict,
    specs: list[WindowSpec],
    atom_coords_world: np.ndarray,
    atom_features: np.ndarray,
    origin_xyz: np.ndarray | tuple[float, float, float] | None,
    voxel_size_xyz: np.ndarray | tuple[float, float, float] | None,
    atom_buffer_radius_world: float,
    device: torch.device,
) -> None:

    coords_world = np.asarray(atom_coords_world, dtype=np.float32)

    features = np.asarray(atom_features, dtype=np.float32)

    if coords_world.ndim != 2 or coords_world.shape[1] != 3:

        raise ValueError("atom_coords_world must have shape (N, 3).")

    if (
        features.ndim != 2
        or features.shape[0] != coords_world.shape[0]
        or features.shape[1] != 49
    ):

        raise ValueError("atom_features must have shape (N, 49).")

    origin = (
        np.zeros(3, dtype=np.float32)
        if origin_xyz is None
        else np.asarray(origin_xyz, dtype=np.float32)
    )

    voxel_size = (
        np.ones(3, dtype=np.float32)
        if voxel_size_xyz is None
        else np.asarray(voxel_size_xyz, dtype=np.float32)
    )

    coord_world_parts: list[np.ndarray] = []

    coord_local_parts: list[np.ndarray] = []

    feature_parts: list[np.ndarray] = []

    batch_parts: list[np.ndarray] = []

    global_parts: list[np.ndarray] = []

    core_parts: list[np.ndarray] = []

    centered_parts: list[np.ndarray] = []

    counts: list[int] = []

    for batch_idx, spec in enumerate(specs):

        z0, y0, x0 = spec.position_zyx

        dz, dy, dx = spec.shape_zyx

        box_origin_world = origin + np.asarray([x0, y0, z0], dtype=np.float32) * voxel_size

        box_extent_world = np.asarray([dx, dy, dz], dtype=np.float32) * voxel_size

        box_max_world = box_origin_world + box_extent_world

        core_mask = np.all(coords_world >= box_origin_world[None, :], axis=1)

        core_mask &= np.all(coords_world < box_max_world[None, :], axis=1)

        selected_mask = np.all(
            coords_world
            >= (box_origin_world - float(atom_buffer_radius_world))[None, :],
            axis=1,
        )

        selected_mask &= np.all(
            coords_world
            < (box_max_world + float(atom_buffer_radius_world))[None, :],
            axis=1,
        )

        indices = np.where(selected_mask)[0]

        counts.append(int(indices.size))

        if indices.size == 0:

            continue

        coord_world_parts.append(coords_world[indices])

        coord_local_parts.append(
            ((coords_world[indices] - box_origin_world[None, :]) / voxel_size[None, :]).astype(np.float32, copy=False)
        )

        box_center_world = box_origin_world + 0.5 * np.asarray([dx, dy, dz], dtype=np.float32) * voxel_size
        centered_parts.append((coords_world[indices] - box_center_world[None, :]).astype(np.float32, copy=False))

        feature_parts.append(features[indices])

        batch_parts.append(np.full(indices.size, batch_idx, dtype=np.int64))

        global_parts.append(indices.astype(np.int64))

        core_parts.append(core_mask[indices])

    if feature_parts:

        atom_coord_world = np.concatenate(coord_world_parts, axis=0).astype(np.float32)

        atom_coord_local = np.concatenate(coord_local_parts, axis=0).astype(np.float32)

        atom_feat = np.concatenate(feature_parts, axis=0).astype(np.float32)

        atom_batch_index = np.concatenate(batch_parts, axis=0).astype(np.int64)

        atom_global_indices = np.concatenate(global_parts, axis=0).astype(np.int64)

        atom_is_in_core_box = np.concatenate(core_parts, axis=0).astype(bool)

        centered = np.concatenate(centered_parts, axis=0).astype(np.float32)

    else:

        atom_coord_world = np.zeros((0, 3), dtype=np.float32)

        atom_coord_local = np.zeros((0, 3), dtype=np.float32)

        atom_feat = np.zeros((0, 49), dtype=np.float32)

        atom_batch_index = np.zeros((0,), dtype=np.int64)

        atom_global_indices = np.zeros((0,), dtype=np.int64)

        atom_is_in_core_box = np.zeros((0,), dtype=bool)

        centered = atom_coord_world

    offsets = np.cumsum(np.asarray(counts, dtype=np.int64))

    batch["atom_coord_world"] = torch.from_numpy(atom_coord_world).to(device=device)

    batch["atom_coord_local_voxel"] = torch.from_numpy(atom_coord_local).to(
        device=device
    )

    batch["atom_coord_centered_world"] = torch.from_numpy(centered).to(device=device)

    batch["atom_feat"] = torch.from_numpy(atom_feat).to(device=device)

    batch["atom_batch_index"] = torch.from_numpy(atom_batch_index).to(device=device)

    batch["atom_global_indices"] = torch.from_numpy(atom_global_indices).to(
        device=device
    )

    batch["atom_counts"] = torch.tensor(counts, dtype=torch.long, device=device)

    batch["atom_offsets"] = torch.from_numpy(offsets).to(device=device)

    batch["atom_label"] = torch.zeros(
        (atom_feat.shape[0],), dtype=torch.long, device=device
    )

    batch["atom_is_in_core_box"] = torch.from_numpy(atom_is_in_core_box).to(
        device=device
    )
