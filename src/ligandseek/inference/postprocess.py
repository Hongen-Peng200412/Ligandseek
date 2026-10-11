"""Apply paper component filtering and write predictions on a ZYX grid.

Candidates contain voxel counts, probability scores, ZYX bounds/centers, and XYZ world centers in Angstroms. Prediction output directories contain native probability, mask, instance and candidate files plus grid and parameter metadata.
"""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from scipy import ndimage

PAPER_THRESHOLDS = {
    "unet_raw": {"deposited": 0.86, "inferred": 0.86},
    "unet_diff": {"deposited": 0.87, "inferred": 0.87},
    "unet_full": {"deposited": 0.95, "inferred": 0.98},
    "ligandseek_pv": {"deposited": 0.79, "inferred": 0.88},
}

@dataclass(frozen=True)
class VoxelCandidate:
    instance_id: int
    voxel_count: int
    score_mean: float
    score_max: float
    center_voxel_zyx: tuple[float, float, float]
    center_world_xyz: tuple[float, float, float]
    bbox_min_zyx: tuple[int, int, int]
    bbox_max_zyx: tuple[int, int, int]
    @property
    def score(self) -> float:
        return self.score_mean
    @property
    def center_zyx(self) -> tuple[float, float, float]:
        return self.center_voxel_zyx

@dataclass(frozen=True)
class PostprocessResult:
    prob_map: np.ndarray
    score_map: np.ndarray
    binary_mask_raw: np.ndarray
    instance_label_raw: np.ndarray
    binary_mask_filtered: np.ndarray
    instance_label_filtered: np.ndarray
    candidates: list[VoxelCandidate]
    @property
    def instance_label(self) -> np.ndarray:
        return self.instance_label_filtered

def _build_candidates(instance_label: np.ndarray, score_map: np.ndarray, origin: np.ndarray, voxel_size: np.ndarray) -> list[VoxelCandidate]:
    label = np.asarray(instance_label, dtype=np.int32)
    positive_mask = label > 0
    if not bool(positive_mask.any()):
        return []
    coords_zyx = np.nonzero(positive_mask)
    labels = label[positive_mask].astype(np.int64, copy=False)
    scores = np.asarray(score_map, dtype=np.float32)[positive_mask].astype(np.float64, copy=False)
    max_label = int(labels.max())
    counts = np.bincount(labels, minlength=max_label + 1)
    score_sum = np.bincount(labels, weights=scores, minlength=max_label + 1)
    score_max = np.full(max_label + 1, -np.inf, dtype=np.float64)
    np.maximum.at(score_max, labels, scores)
    coord_sum = [np.bincount(labels, weights=coords_zyx[axis], minlength=max_label + 1) for axis in range(3)]
    bbox_min = np.full((max_label + 1, 3), np.iinfo(np.int64).max, dtype=np.int64)
    bbox_max = np.full((max_label + 1, 3), np.iinfo(np.int64).min, dtype=np.int64)
    for axis in range(3):
        np.minimum.at(bbox_min[:, axis], labels, coords_zyx[axis])
        np.maximum.at(bbox_max[:, axis], labels, coords_zyx[axis])
    candidates: list[VoxelCandidate] = []
    for instance_id in np.flatnonzero(counts > 0):
        if int(instance_id) == 0:
            continue
        center_zyx = np.array([coord_sum[axis][instance_id] / counts[instance_id] for axis in range(3)], dtype=np.float64)
        candidates.append(VoxelCandidate(instance_id=int(instance_id), voxel_count=int(counts[instance_id]), score_mean=float(score_sum[instance_id] / counts[instance_id]), score_max=float(score_max[instance_id]), center_voxel_zyx=tuple((float(v) for v in center_zyx.tolist())), center_world_xyz=voxel_zyx_to_world_xyz(tuple((float(v) for v in center_zyx)), origin, voxel_size), bbox_min_zyx=tuple((int(v) for v in bbox_min[instance_id].tolist())), bbox_max_zyx=tuple((int(v) for v in bbox_max[instance_id].tolist()))))
    return candidates

def _merge_instances_by_center_distance(instance_label: np.ndarray, origin: np.ndarray, voxel_size: np.ndarray, merge_dist: float) -> np.ndarray:
    label = np.asarray(instance_label, dtype=np.int32)
    positive_mask = label > 0
    if not bool(positive_mask.any()):
        return np.zeros_like(label, dtype=np.int32)
    coords_zyx = np.nonzero(positive_mask)
    labels = label[positive_mask].astype(np.int64, copy=False)
    max_label = int(labels.max())
    counts = np.bincount(labels, minlength=max_label + 1)
    instance_ids = np.flatnonzero(counts > 0)
    instance_ids = instance_ids[instance_ids > 0]
    if instance_ids.shape[0] <= 1:
        return label.astype(np.int32, copy=True)
    coord_sum = [np.bincount(labels, weights=coords_zyx[axis], minlength=max_label + 1) for axis in range(3)]
    centers_zyx = np.stack([coord_sum[axis][instance_ids] / counts[instance_ids] for axis in range(3)], axis=1)
    origin_float = np.asarray(origin, dtype=np.float64).reshape(3)
    voxel_size_float = np.asarray(voxel_size, dtype=np.float64).reshape(3)
    centers_world_xyz = np.column_stack((origin_float[0] + (centers_zyx[:, 2] + 0.5) * voxel_size_float[0], origin_float[1] + (centers_zyx[:, 1] + 0.5) * voxel_size_float[1], origin_float[2] + (centers_zyx[:, 0] + 0.5) * voxel_size_float[2]))
    parent = np.arange(instance_ids.shape[0], dtype=np.int32)

    def find(local_index: int) -> int:
        while int(parent[local_index]) != int(local_index):
            parent[local_index] = parent[int(parent[local_index])]
            local_index = int(parent[local_index])
        return int(local_index)

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parent[right_root] = left_root
    merge_dist2 = float(merge_dist) * float(merge_dist)
    for left_index in range(instance_ids.shape[0] - 1):
        dist2 = np.sum((centers_world_xyz[left_index + 1:] - centers_world_xyz[left_index]) ** 2, axis=1)
        close_offsets = np.flatnonzero(dist2 < merge_dist2)
        for offset in close_offsets:
            union(left_index, left_index + 1 + int(offset))
    root_to_new_id: dict[int, int] = {}
    new_ids = np.zeros(max_label + 1, dtype=np.int32)
    for (local_index, instance_id) in enumerate(instance_ids):
        root = find(int(local_index))
        if root not in root_to_new_id:
            root_to_new_id[root] = len(root_to_new_id) + 1
        new_ids[int(instance_id)] = root_to_new_id[root]
    return new_ids[label]

def _filter_small_components(instance_label: np.ndarray, min_component_voxels: int) -> np.ndarray:
    if min_component_voxels < 1:
        raise ValueError(f'min_component_voxels must be >= 1, got {min_component_voxels}')
    label = np.asarray(instance_label, dtype=np.int32)
    counts = np.bincount(label.ravel())
    keep = counts >= int(min_component_voxels)
    keep[0] = False
    new_ids = np.zeros_like(counts, dtype=np.int32)
    new_ids[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return new_ids[label]

def voxel_zyx_to_world_xyz(voxel_zyx: tuple[float, float, float], origin: np.ndarray, voxel_size: np.ndarray) -> tuple[float, float, float]:
    (z, y, x) = [float(v) for v in voxel_zyx]
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    voxel_size = np.asarray(voxel_size, dtype=np.float32).reshape(3)
    return (float(origin[0] + (x + 0.5) * voxel_size[0]), float(origin[1] + (y + 0.5) * voxel_size[1]), float(origin[2] + (z + 0.5) * voxel_size[2]))

# ================================================================================================
def postprocess_ligand_probability_map(ligand_probability: np.ndarray, threshold: float = 0.86, min_component_voxels: int = 10, merge_center_distance: float = 5.0, voxel_size_xyz: np.ndarray | tuple[float, float, float] | None = None, origin_xyz: np.ndarray | tuple[float, float, float] | None = None) -> PostprocessResult:
    """Threshold with >=, remove small six-connected components, then transitively merge centers at a strict distance < merge_center_distance Angstroms."""
    probability = np.asarray(ligand_probability, dtype=np.float32)
    if probability.ndim != 3:
        raise ValueError(f"ligand_probability must have shape (D,H,W), got {probability.shape}")
    voxel_size = np.ones(3, dtype=np.float32) if voxel_size_xyz is None else np.asarray(voxel_size_xyz, dtype=np.float32).reshape(3)
    origin = np.zeros(3, dtype=np.float32) if origin_xyz is None else np.asarray(origin_xyz, dtype=np.float32).reshape(3)
    binary_mask_raw = probability >= float(threshold)
    instance_label_raw, _ = ndimage.label(binary_mask_raw, structure=ndimage.generate_binary_structure(3, 1))
    instance_label_raw = instance_label_raw.astype(np.int32, copy=False)
    base_label = _filter_small_components(instance_label_raw, min_component_voxels)
    if float(merge_center_distance) > 0.0:
        base_label = _merge_instances_by_center_distance(base_label, origin, voxel_size, float(merge_center_distance))
    score_map = probability.astype(np.float32, copy=True)
    return PostprocessResult(prob_map=probability, score_map=score_map, binary_mask_raw=binary_mask_raw, instance_label_raw=instance_label_raw, binary_mask_filtered=base_label > 0, instance_label_filtered=base_label, candidates=_build_candidates(base_label, score_map, origin, voxel_size))


def write_prediction_outputs(probability: np.ndarray, origin: np.ndarray, voxel_size: np.ndarray, hardmask: np.ndarray | None, model_name: str, receptor_mode: str, output_dir: str | Path, threshold: float | None = None) -> PostprocessResult:
    """Write native-format probability, mask, instance and candidate artifacts, grid_metadata.npz and run_metadata.json.

    Probability/hardmask use ZYX axes; origin/spacing use XYZ axes. Supplied receptor home voxels are zeroed. Omitted threshold uses the frozen model/receptor value (raw: 0.86). Metadata records effective parameters and clearing mode without input paths.
    """
    effective_threshold = PAPER_THRESHOLDS[model_name][receptor_mode] if threshold is None else float(threshold)
    probability = np.asarray(probability, dtype=np.float32)
    occupancy = np.zeros(probability.shape, dtype=np.int64) if hardmask is None else np.asarray(hardmask, dtype=np.int64)
    if probability.shape != occupancy.shape:
        raise ValueError(f"shape mismatch: probability={probability.shape}, hardmask={occupancy.shape}")
    probability = (probability * (1.0 - occupancy.astype(np.float32))).astype(np.float32)
    post = postprocess_ligand_probability_map(probability, threshold=effective_threshold, min_component_voxels=10, merge_center_distance=5.0, voxel_size_xyz=voxel_size, origin_xyz=origin)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "ligand_pred.npz", ligand_pred=probability)
    np.savez(output / "ligand_mask_filtered.npz", binary_mask_filtered=post.binary_mask_filtered.astype(np.bool_), score_map=post.score_map.astype(np.float32))
    np.savez(output / "instance_label_filtered.npz", instance_label_filtered=post.instance_label_filtered.astype(np.int32))
    np.savez(output / "grid_metadata.npz", origin=np.asarray(origin, dtype=np.float32), voxel_size=np.asarray(voxel_size, dtype=np.float32), hardmask=occupancy)
    (output / "voxel_candidates.json").write_text(json.dumps([asdict(candidate) for candidate in post.candidates], indent=2) + "\n", encoding="utf-8")
    metadata = {"model": model_name, "receptor_mode": receptor_mode, "receptor_clearing": hardmask is not None, "threshold": effective_threshold, "threshold_source": "paper_default" if threshold is None else "user", "min_component_voxels": 10, "connectivity_policy": "7_none", "merge_dist": 5.0, "axis_order": "ZYX"}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return post
