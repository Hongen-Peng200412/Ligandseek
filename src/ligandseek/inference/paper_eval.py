"""Evaluate fixed-protocol predictions from label files and receptor occupancy."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ligandseek.inference.metrics import (
    binary_voxel_metrics,
    instance_metrics,
    topk_success,
    voxel_pr_auc,
)
from ligandseek.inference.postprocess import postprocess_ligand_probability_map


DEFAULT_CLASS_MAPPING = [0, 1, 1, 1, 1]


@dataclass(frozen=True)
class SampleEvalResult:
    sample_id: str
    dice: float
    precision: float
    recall: float
    pr_auc: float | None
    coverage_precision_03: float
    coverage_recall_03: float
    coverage_f1_03: float
    coverage_precision_06: float
    coverage_recall_06: float
    coverage_f1_06: float
    one_to_one_precision_03: float
    one_to_one_recall_03: float
    one_to_one_f1_03: float
    one_to_one_precision_06: float
    one_to_one_recall_06: float
    one_to_one_f1_06: float
    top3_success_06: int
    candidates: int
    num_pred_instances: int
    num_gt_instances: int
    coverage_precision_true_positives_03: int
    coverage_recall_true_positives_03: int
    coverage_precision_true_positives_06: int
    coverage_recall_true_positives_06: int
    one_to_one_true_positives_03: int
    one_to_one_true_positives_06: int


def load_records(json_path: str | Path, data_root: str | Path) -> list[dict[str, str]]:
    root = Path(data_root)
    records = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for record in records:
        out.append({key: str(root / value) for key, value in record.items()})
    return out


def load_labels_npz(
    path: str | Path,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    grid_shape_zyx: tuple[int, int, int],
    class_mapping: list[int] | None = None,
    distance_threshold: float = 1.7,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "gt_ligand_mask" in data and "gt_instance_label" in data:
            return data["gt_ligand_mask"].astype(bool), data["gt_instance_label"].astype(np.int32)
        if "ligand_mask" in data and "instance_label" in data:
            return data["ligand_mask"].astype(bool), data["instance_label"].astype(np.int32)
        if "ligand_candidate_ids" in data and "ligand_class_ids" in data:
            return _build_gt_from_ligand_coords_npz(
                data=data,
                origin=origin,
                voxel_size=voxel_size,
                grid_shape_zyx=grid_shape_zyx,
                class_mapping=DEFAULT_CLASS_MAPPING if class_mapping is None else class_mapping,
                distance_threshold=distance_threshold,
            )
        keys = ", ".join(sorted(data.files))
        raise KeyError(f"Unsupported labels.npz schema. Found keys: {keys}")


def build_receptor_occupancy_mask(
    atom_coords_world: np.ndarray,
    origin_xyz: np.ndarray,
    voxel_size_xyz: np.ndarray,
    grid_shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Map receptor atoms to their floor-indexed home voxels on a ZYX grid."""

    mask = np.zeros(grid_shape_zyx, dtype=bool)
    coords = np.asarray(atom_coords_world, dtype=np.float32)
    if coords.size == 0:
        return mask

    origin = np.asarray(origin_xyz, dtype=np.float32).reshape(3)
    voxel_size = np.asarray(voxel_size_xyz, dtype=np.float32).reshape(3)
    coords_voxel_xyz = (coords - origin[None, :]) / voxel_size[None, :]
    voxel_xyz = np.floor(coords_voxel_xyz).astype(np.int64, copy=False)
    depth, height, width = (int(value) for value in grid_shape_zyx)
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


def evaluate_probability_map(
    sample_id: str,
    probability: np.ndarray,
    labels_npz_path: str | Path,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    threshold: float,
    min_component_voxels: int = 10,
    ligand_gt_distance_threshold: float = 1.7,
    receptor_occupancy_mask: np.ndarray | None = None,
) -> SampleEvalResult:
    gt_mask, gt_instance = load_labels_npz(
        labels_npz_path,
        origin=origin,
        voxel_size=voxel_size,
        grid_shape_zyx=tuple(int(v) for v in probability.shape),
        distance_threshold=ligand_gt_distance_threshold,
    )
    post = postprocess_ligand_probability_map(
        ligand_probability=probability,
        threshold=threshold,
        min_component_voxels=min_component_voxels,
        voxel_size_xyz=voxel_size,
        origin_xyz=origin,
    )
    pred_mask = post.instance_label > 0
    voxel = binary_voxel_metrics(pred_mask, gt_mask)
    inst03 = instance_metrics(post.instance_label, gt_instance, coverage_threshold=0.3)
    inst06 = instance_metrics(post.instance_label, gt_instance, coverage_threshold=0.6)
    valid_mask = (
        None
        if receptor_occupancy_mask is None
        else ~np.asarray(receptor_occupancy_mask, dtype=bool)
    )
    return SampleEvalResult(
        sample_id=sample_id,
        dice=voxel.dice,
        precision=voxel.precision,
        recall=voxel.recall,
        pr_auc=voxel_pr_auc(probability, gt_mask, valid_mask=valid_mask),
        coverage_precision_03=inst03.coverage_precision,
        coverage_recall_03=inst03.coverage_recall,
        coverage_f1_03=inst03.coverage_f1,
        coverage_precision_06=inst06.coverage_precision,
        coverage_recall_06=inst06.coverage_recall,
        coverage_f1_06=inst06.coverage_f1,
        one_to_one_precision_03=inst03.one_to_one_precision,
        one_to_one_recall_03=inst03.one_to_one_recall,
        one_to_one_f1_03=inst03.one_to_one_f1,
        one_to_one_precision_06=inst06.one_to_one_precision,
        one_to_one_recall_06=inst06.one_to_one_recall,
        one_to_one_f1_06=inst06.one_to_one_f1,
        top3_success_06=topk_success(
            post.instance_label,
            gt_instance,
            post.candidates,
            coverage_threshold=0.6,
            k=3,
        ),
        candidates=len(post.candidates),
        num_pred_instances=inst06.num_pred_instances,
        num_gt_instances=inst06.num_gt_instances,
        coverage_precision_true_positives_03=(
            inst03.coverage_precision_true_positives
        ),
        coverage_recall_true_positives_03=inst03.coverage_recall_true_positives,
        coverage_precision_true_positives_06=(
            inst06.coverage_precision_true_positives
        ),
        coverage_recall_true_positives_06=inst06.coverage_recall_true_positives,
        one_to_one_true_positives_03=inst03.one_to_one_true_positives,
        one_to_one_true_positives_06=inst06.one_to_one_true_positives,
    )


def write_eval_json(
    path: str | Path, results: list[SampleEvalResult], threshold: float
) -> dict[str, object]:
    payload: dict[str, object] = {
        "threshold": float(threshold),
        "samples": [asdict(result) for result in results],
        "mean": _mean_results(results),
        "summary": _summarize_results(results),
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _mean_results(results: list[SampleEvalResult]) -> dict[str, float | None]:
    fields = (
        "dice",
        "precision",
        "recall",
        "pr_auc",
        "coverage_precision_03",
        "coverage_recall_03",
        "coverage_f1_03",
        "coverage_precision_06",
        "coverage_recall_06",
        "coverage_f1_06",
        "one_to_one_precision_03",
        "one_to_one_recall_03",
        "one_to_one_f1_03",
        "one_to_one_precision_06",
        "one_to_one_recall_06",
        "one_to_one_f1_06",
        "top3_success_06",
        "candidates",
    )
    return {
        field: _mean_or_none([getattr(result, field) for result in results])
        for field in fields
    }


def _summarize_results(results: list[SampleEvalResult]) -> dict[str, object]:
    pr_auc_values = [result.pr_auc for result in results if result.pr_auc is not None]
    return {
        "num_samples": len(results),
        "voxel": {
            "dice_mean": _mean_or_none([result.dice for result in results]),
            "precision_mean": _mean_or_none([result.precision for result in results]),
            "recall_mean": _mean_or_none([result.recall for result in results]),
        },
        "pr_auc_macro": _mean_or_none(pr_auc_values),
        "pr_auc_num_valid": len(pr_auc_values),
        "top3_success_ratio_06": _mean_or_none(
            [result.top3_success_06 for result in results]
        ),
        "candidate_count_mean": _mean_or_none([result.candidates for result in results]),
        "coverage_03": _instance_summary(
            results,
            "coverage_precision_true_positives_03",
            "coverage_recall_true_positives_03",
        ),
        "coverage_06": _instance_summary(
            results,
            "coverage_precision_true_positives_06",
            "coverage_recall_true_positives_06",
        ),
        "one_to_one_03": _instance_summary(
            results,
            "one_to_one_true_positives_03",
            "one_to_one_true_positives_03",
        ),
        "one_to_one_06": _instance_summary(
            results,
            "one_to_one_true_positives_06",
            "one_to_one_true_positives_06",
        ),
    }


def _instance_summary(
    results: list[SampleEvalResult],
    precision_true_positive_field: str,
    recall_true_positive_field: str,
) -> dict[str, float | int]:
    num_pred = sum(result.num_pred_instances for result in results)
    num_gt = sum(result.num_gt_instances for result in results)
    precision_true_positives = sum(
        int(getattr(result, precision_true_positive_field)) for result in results
    )
    recall_true_positives = sum(
        int(getattr(result, recall_true_positive_field)) for result in results
    )
    precision = _safe_div(precision_true_positives, num_pred)
    recall = _safe_div(recall_true_positives, num_gt)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "precision_true_positives": precision_true_positives,
        "recall_true_positives": recall_true_positives,
        "num_pred_instances": num_pred,
        "num_gt_instances": num_gt,
    }


def _mean_or_none(values: list[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def _build_gt_from_ligand_coords_npz(
    data: np.lib.npyio.NpzFile,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    grid_shape_zyx: tuple[int, int, int],
    class_mapping: list[int],
    distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    ligand_candidate_ids = data["ligand_candidate_ids"].astype(np.int64, copy=False)
    ligand_class_ids = data["ligand_class_ids"].astype(np.int64, copy=False)
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    voxel_size = np.asarray(voxel_size, dtype=np.float32).reshape(3)
    centers = _voxel_centers_xyz(grid_shape_zyx, origin, voxel_size)
    gt_instance = np.zeros(grid_shape_zyx, dtype=np.int32)
    next_instance_id = 1

    for candidate_id, class_id in zip(ligand_candidate_ids.tolist(), ligand_class_ids.tolist()):
        mapped_class_id = _map_class_id(int(class_id), class_mapping)
        if mapped_class_id <= 0:
            continue
        coord_key = f"ligand_coords_{int(candidate_id)}"
        if coord_key not in data.files:
            continue
        ligand_coords = data[coord_key].astype(np.float32, copy=False)
        if ligand_coords.shape[0] == 0:
            continue
        distances, _ = cKDTree(ligand_coords).query(centers, k=1)
        mask = distances.reshape(grid_shape_zyx) <= float(distance_threshold)
        writable = mask & (gt_instance == 0)
        if writable.any():
            gt_instance[writable] = next_instance_id
            next_instance_id += 1
    return gt_instance > 0, gt_instance


def _map_class_id(class_id: int, class_mapping: list[int]) -> int:
    if class_id < 0 or class_id >= len(class_mapping):
        return 0
    return int(class_mapping[class_id])


def _voxel_centers_xyz(
    grid_shape_zyx: tuple[int, int, int],
    origin_xyz: np.ndarray,
    voxel_size_xyz: np.ndarray,
) -> np.ndarray:
    depth, height, width = tuple(int(v) for v in grid_shape_zyx)
    z_idx, y_idx, x_idx = np.mgrid[0:depth, 0:height, 0:width]
    return np.stack(
        [
            x_idx.ravel() * voxel_size_xyz[0] + origin_xyz[0] + voxel_size_xyz[0] * 0.5,
            y_idx.ravel() * voxel_size_xyz[1] + origin_xyz[1] + voxel_size_xyz[1] * 0.5,
            z_idx.ravel() * voxel_size_xyz[2] + origin_xyz[2] + voxel_size_xyz[2] * 0.5,
        ],
        axis=1,
    ).astype(np.float32)
