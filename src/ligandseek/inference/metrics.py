"""Compute paper voxel and instance metrics without changing labels or thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class BinaryVoxelMetrics:
    dice: float
    precision: float
    recall: float


@dataclass(frozen=True)
class InstanceOverlapStats:
    pred_ids: np.ndarray
    gt_ids: np.ndarray
    pred_coverage: np.ndarray
    gt_coverage: np.ndarray


@dataclass(frozen=True)
class InstanceMetrics:
    coverage_precision: float
    coverage_recall: float
    coverage_f1: float
    one_to_one_precision: float
    one_to_one_recall: float
    one_to_one_f1: float
    num_pred_instances: int
    num_gt_instances: int
    coverage_precision_true_positives: int
    coverage_recall_true_positives: int
    one_to_one_true_positives: int


def binary_voxel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> BinaryVoxelMetrics:
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, gt={gt.shape}")
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    dice = 2.0 * tp / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn > 0 else 0.0
    return BinaryVoxelMetrics(dice=dice, precision=precision, recall=recall)


def build_instance_overlap_stats(
    pred_instance_label: np.ndarray, gt_instance_label: np.ndarray
) -> InstanceOverlapStats:
    """Compute both directional voxel coverage matrices for predicted and GT instances."""

    pred = np.asarray(pred_instance_label, dtype=np.int64)
    gt = np.asarray(gt_instance_label, dtype=np.int64)
    if pred.shape != gt.shape:
        raise ValueError(
            f"shape mismatch: pred_instance_label={pred.shape}, "
            f"gt_instance_label={gt.shape}"
        )

    pred_positive = pred > 0
    gt_positive = gt > 0
    pred_ids, pred_inverse = np.unique(pred[pred_positive], return_inverse=True)
    gt_ids, gt_inverse = np.unique(gt[gt_positive], return_inverse=True)
    num_pred = int(pred_ids.size)
    num_gt = int(gt_ids.size)

    pred_sizes = np.bincount(pred_inverse, minlength=num_pred).astype(np.int64)
    gt_sizes = np.bincount(gt_inverse, minlength=num_gt).astype(np.int64)
    overlap = np.zeros((num_pred, num_gt), dtype=np.int64)

    if num_pred > 0 and num_gt > 0:
        paired = pred_positive & gt_positive
        if paired.any():
            paired_pred = np.searchsorted(pred_ids, pred[paired])
            paired_gt = np.searchsorted(gt_ids, gt[paired])
            overlap = np.bincount(
                paired_pred * num_gt + paired_gt,
                minlength=num_pred * num_gt,
            ).reshape(num_pred, num_gt)

    pred_coverage = np.divide(
        overlap,
        pred_sizes[:, None],
        out=np.zeros((num_pred, num_gt), dtype=np.float64),
        where=pred_sizes[:, None] > 0,
    )
    gt_coverage = np.divide(
        overlap,
        gt_sizes[None, :],
        out=np.zeros((num_pred, num_gt), dtype=np.float64),
        where=gt_sizes[None, :] > 0,
    )
    return InstanceOverlapStats(
        pred_ids=pred_ids,
        gt_ids=gt_ids,
        pred_coverage=pred_coverage,
        gt_coverage=gt_coverage,
    )


def instance_metrics(
    pred_instance_label: np.ndarray,
    gt_instance_label: np.ndarray,
    coverage_threshold: float = 0.6,
) -> InstanceMetrics:
    """Compute directional coverage and Hungarian matches using geometric-mean coverage."""

    stats = build_instance_overlap_stats(pred_instance_label, gt_instance_label)
    num_pred = int(stats.pred_ids.size)
    num_gt = int(stats.gt_ids.size)
    pred_coverage = stats.pred_coverage
    gt_coverage = stats.gt_coverage

    coverage_precision_true_positives = 0
    coverage_recall_true_positives = 0
    one_to_one_true_positives = 0
    if num_pred > 0 and num_gt > 0:
        coverage_precision_true_positives = int(
            (pred_coverage >= float(coverage_threshold)).any(axis=1).sum()
        )
        coverage_recall_true_positives = int(
            (gt_coverage >= float(coverage_threshold)).any(axis=0).sum()
        )

        match_score = np.sqrt(pred_coverage * gt_coverage)
        row_indices, col_indices = linear_sum_assignment(match_score, maximize=True)
        matched_pred_coverage = pred_coverage[row_indices, col_indices]
        matched_gt_coverage = gt_coverage[row_indices, col_indices]
        one_to_one_true_positives = int(
            (
                (matched_pred_coverage >= float(coverage_threshold))
                & (matched_gt_coverage >= float(coverage_threshold))
            ).sum()
        )
    coverage_precision = _safe_div(coverage_precision_true_positives, num_pred)
    coverage_recall = _safe_div(coverage_recall_true_positives, num_gt)
    one_to_one_precision = _safe_div(one_to_one_true_positives, num_pred)
    one_to_one_recall = _safe_div(one_to_one_true_positives, num_gt)
    return InstanceMetrics(
        coverage_precision=coverage_precision,
        coverage_recall=coverage_recall,
        coverage_f1=_f1(coverage_precision, coverage_recall),
        one_to_one_precision=one_to_one_precision,
        one_to_one_recall=one_to_one_recall,
        one_to_one_f1=_f1(one_to_one_precision, one_to_one_recall),
        num_pred_instances=num_pred,
        num_gt_instances=num_gt,
        coverage_precision_true_positives=coverage_precision_true_positives,
        coverage_recall_true_positives=coverage_recall_true_positives,
        one_to_one_true_positives=one_to_one_true_positives,
    )


def topk_success(
    pred_instance_label: np.ndarray,
    gt_instance_label: np.ndarray,
    candidates: Sequence[object],
    coverage_threshold: float = 0.6,
    k: int = 3,
) -> int:
    """Check whether a top-K mean-probability candidate reaches coverage in both directions."""

    stats = build_instance_overlap_stats(pred_instance_label, gt_instance_label)
    if stats.pred_ids.size == 0 or stats.gt_ids.size == 0:
        return 0

    ranked_ids = [
        int(candidate.instance_id)
        for candidate in sorted(candidates, key=lambda candidate: float(candidate.score), reverse=True)
    ]
    selected_ids = np.asarray(ranked_ids[: int(k)], dtype=np.int64)
    if selected_ids.size == 0:
        return 0

    row_indices = np.searchsorted(stats.pred_ids, selected_ids)
    if (
        (row_indices >= stats.pred_ids.size).any()
        or not np.array_equal(stats.pred_ids[row_indices], selected_ids)
    ):
        raise ValueError("candidate instance IDs do not match pred_instance_label.")

    valid_pair = (
        stats.pred_coverage[row_indices] >= float(coverage_threshold)
    ) & (stats.gt_coverage[row_indices] >= float(coverage_threshold))
    return int(valid_pair.any())


def voxel_pr_auc(
    score_map: np.ndarray,
    gt_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float | None:
    """Compute Average Precision over valid voxels; receptor home voxels should be excluded."""

    scores = np.asarray(score_map, dtype=np.float64)
    gt = np.asarray(gt_mask, dtype=bool)
    valid = np.ones(scores.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if not (scores.shape == gt.shape == valid.shape):
        raise ValueError(
            f"shape mismatch: score_map={scores.shape}, gt_mask={gt.shape}, "
            f"valid_mask={valid.shape}"
        )

    valid_scores = scores[valid]
    valid_labels = gt[valid].astype(np.int64, copy=False)
    num_positive = int(valid_labels.sum())
    if num_positive == 0:
        return None

    order = np.argsort(-valid_scores, kind="mergesort")
    sorted_scores = valid_scores[order]
    sorted_labels = valid_labels[order]
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)
    score_boundaries = np.r_[
        np.where(np.diff(sorted_scores) != 0.0)[0],
        sorted_scores.shape[0] - 1,
    ]
    tp = cumulative_tp[score_boundaries]
    fp = cumulative_fp[score_boundaries]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(num_positive)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
