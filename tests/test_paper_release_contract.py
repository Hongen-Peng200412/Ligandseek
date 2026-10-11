from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ligandseek.inference.metrics import instance_metrics, voxel_pr_auc
from ligandseek.inference.paper_eval import (
    _summarize_results,
    evaluate_probability_map,
    write_eval_json,
)
from ligandseek.inference.postprocess import postprocess_ligand_probability_map
from ligandseek.inference.sliding_window import (
    WindowSpec,
    _build_gaussian_weight,
    _empty_atom_batch,
    _fill_atom_batch,
    _merge_probability,
    predict_volume,
)


def test_gaussian_stitching_uses_original_window_weight() -> None:
    weight = _build_gaussian_weight((4, 4, 4), gaussian_sigma_ratio=0.5)
    assert weight[1, 1, 1] > weight[0, 0, 0]

    value_sum = np.zeros((4, 4, 6), dtype=np.float32)
    weight_sum = np.zeros_like(value_sum)
    first = WindowSpec(position_zyx=(0, 0, 0), shape_zyx=(4, 4, 4))
    second = WindowSpec(position_zyx=(0, 0, 2), shape_zyx=(4, 4, 4))
    _merge_probability(value_sum, weight_sum, np.ones((4, 4, 4), dtype=np.float32), first, weight)
    _merge_probability(value_sum, weight_sum, np.zeros((4, 4, 4), dtype=np.float32), second, weight)

    merged = value_sum / weight_sum
    expected = weight[1, 1, 2] / (weight[1, 1, 2] + weight[1, 1, 0])
    assert merged[1, 1, 2] == pytest.approx(expected)
    assert merged[1, 1, 0] == pytest.approx(1.0)
    assert merged[1, 1, 5] == pytest.approx(0.0)


def test_atom_batch_keeps_aabb_buffer_and_core_flags() -> None:
    batch = _empty_atom_batch(
        batch_size=1,
        window_size=4,
        channels=1,
        device=torch.device("cpu"),
    )
    spec = WindowSpec(position_zyx=(10, 20, 30), shape_zyx=(2, 3, 4))
    origin = np.asarray([100.0, 200.0, 300.0], dtype=np.float32)
    voxel_size = np.asarray([2.0, 4.0, 8.0], dtype=np.float32)
    atom_coords = np.asarray(
        [
            [160.0, 280.0, 380.0],
            [164.0, 292.0, 396.0],
            [152.0, 272.0, 372.0],
            [172.0, 300.0, 404.0],
        ],
        dtype=np.float32,
    )
    atom_features = np.zeros((atom_coords.shape[0], 49), dtype=np.float32)

    _fill_atom_batch(
        batch=batch,
        specs=[spec],
        atom_coords_world=atom_coords,
        atom_features=atom_features,
        origin_xyz=origin,
        voxel_size_xyz=voxel_size,
        atom_buffer_radius_world=8.0,
        device=torch.device("cpu"),
    )

    assert batch["atom_global_indices"].tolist() == [0, 1, 2]
    assert batch["atom_is_in_core_box"].tolist() == [True, False, False]
    assert batch["atom_counts"].tolist() == [3]
    assert batch["atom_offsets"].tolist() == [3]
    np.testing.assert_allclose(
        batch["atom_coord_local_voxel"][2].numpy(),
        np.asarray([-4.0, -2.0, -1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        batch["atom_coord_centered_world"][0].numpy(),
        np.asarray([-4.0, -6.0, -8.0], dtype=np.float32),
    )


def test_predict_volume_screens_receptor_home_voxels() -> None:
    class ConstantModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_hardmask: torch.Tensor | None = None

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            self.last_hardmask = batch["hardmask"].detach().cpu()
            batch_size, _, depth, height, width = batch["voxel_grid"].shape
            logits = torch.full(
                (batch_size, 1, depth, height, width), 2.0, device=batch["voxel_grid"].device
            )
            return {"voxel_logits_ligand": logits}

    model = ConstantModel()
    probability = predict_volume(
        model=model,
        exp_volume=np.zeros((4, 4, 4), dtype=np.float32),
        sim_volume=None,
        density_channels=["exp_clipnorm_nopost"],
        window_size=4,
        stride=2,
        output_mask_atom_coords_world=np.asarray(
            [[1.1, 2.1, 3.1]], dtype=np.float32
        ),
        origin_xyz=np.zeros(3, dtype=np.float32),
        voxel_size_xyz=np.ones(3, dtype=np.float32),
    )
    assert model.last_hardmask is not None
    assert not bool(model.last_hardmask.any())
    assert probability[3, 2, 1] == pytest.approx(0.0)
    assert probability[0, 0, 0] > 0.8


def test_postprocess_uses_world_distance_and_transitive_merging() -> None:
    transitive_probability = np.zeros((1, 1, 5), dtype=np.float32)
    transitive_probability[0, 0, [0, 2, 4]] = 0.9
    transitive = postprocess_ligand_probability_map(
        transitive_probability,
        threshold=0.5,
        min_component_voxels=1,
        merge_center_distance=2.1,
        voxel_size_xyz=np.ones(3, dtype=np.float32),
    )
    assert len(transitive.candidates) == 1
    assert transitive.candidates[0].voxel_count == 3

    anisotropic_probability = np.zeros((1, 1, 9), dtype=np.float32)
    anisotropic_probability[0, 0, [0, 4, 8]] = 0.9
    anisotropic = postprocess_ligand_probability_map(
        anisotropic_probability,
        threshold=0.5,
        min_component_voxels=1,
        merge_center_distance=5.0,
        voxel_size_xyz=np.asarray([2.0, 1.0, 1.0], dtype=np.float32),
    )
    assert len(anisotropic.candidates) == 3


def test_instance_metrics_use_prediction_coverage_and_geometric_matching() -> None:
    pred = np.zeros(12, dtype=np.int32)
    gt = np.zeros(12, dtype=np.int32)
    pred[0:3] = 1
    pred[3:8] = 2
    gt[0] = 1
    gt[1] = 2
    gt[3:5] = 1
    gt[5:12] = 2

    metrics = instance_metrics(pred, gt, coverage_threshold=0.3)
    assert metrics.coverage_precision == pytest.approx(1.0)
    assert metrics.coverage_recall == pytest.approx(1.0)
    assert metrics.one_to_one_true_positives == 2
    assert metrics.one_to_one_f1 == pytest.approx(1.0)

    large_prediction = np.ones(10, dtype=np.int32)
    small_gt = np.zeros(10, dtype=np.int32)
    small_gt[:2] = 1
    direction_metrics = instance_metrics(
        large_prediction, small_gt, coverage_threshold=0.6
    )
    assert direction_metrics.coverage_precision == pytest.approx(0.0)
    assert direction_metrics.coverage_recall == pytest.approx(1.0)
    assert direction_metrics.coverage_precision_true_positives == 0
    assert direction_metrics.coverage_recall_true_positives == 1


def test_pr_auc_excludes_receptor_occupancy_voxels() -> None:
    scores = np.asarray([0.99, 0.90, 0.10], dtype=np.float32)
    gt = np.asarray([False, True, False])
    valid = np.asarray([False, True, True])
    assert voxel_pr_auc(scores, gt, valid_mask=valid) == pytest.approx(1.0)


def test_evaluation_json_reports_pr_auc_top3_and_global_pr(tmp_path: Path) -> None:
    probability = np.full((5, 5, 5), 0.05, dtype=np.float32)
    probability[2, 2, 2] = 0.90
    gt_mask = np.zeros_like(probability, dtype=bool)
    gt_mask[2, 2, 2] = True
    gt_instance = gt_mask.astype(np.int32)
    labels_path = tmp_path / "labels.npz"
    np.savez(labels_path, gt_ligand_mask=gt_mask, gt_instance_label=gt_instance)

    result = evaluate_probability_map(
        sample_id="toy",
        probability=probability,
        labels_npz_path=labels_path,
        origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.ones(3, dtype=np.float32),
        threshold=0.5,
        min_component_voxels=1,
        receptor_occupancy_mask=np.zeros_like(gt_mask),
    )
    assert result.pr_auc == pytest.approx(1.0)
    assert result.top3_success_06 == 1

    metrics_path = tmp_path / "metrics.json"
    payload = write_eval_json(metrics_path, [result], threshold=0.5)
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["threshold"] == pytest.approx(0.5)
    assert saved["summary"]["pr_auc_macro"] == pytest.approx(1.0)
    assert saved["summary"]["top3_success_ratio_06"] == pytest.approx(1.0)
    assert saved["summary"]["coverage_06"]["precision"] == pytest.approx(1.0)
    assert saved["summary"]["one_to_one_06"]["recall"] == pytest.approx(1.0)

    asymmetric = replace(
        result,
        num_pred_instances=1,
        num_gt_instances=1,
        coverage_precision_true_positives_03=0,
        coverage_recall_true_positives_03=1,
        coverage_precision_true_positives_06=0,
        coverage_recall_true_positives_06=1,
    )
    asymmetric_summary = _summarize_results([asymmetric])
    assert asymmetric_summary["coverage_06"]["precision"] == pytest.approx(0.0)
    assert asymmetric_summary["coverage_06"]["recall"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "model,setting,override,expected",
    [
        ("unet_raw", "deposited", None, 0.86),
        ("unet_raw", "inferred", None, 0.86),
        ("unet_raw", "deposited", 0.42, 0.42),
        ("unet_diff", "deposited", None, 0.87),
        ("unet_full", "inferred", None, 0.98),
        ("ligandseek_pv", "deposited", None, 0.79),
        ("ligandseek_pv", "inferred", None, 0.88),
    ],
)
def test_paper_evaluation_uses_frozen_default_or_user_threshold(
    monkeypatch, tmp_path, model, setting, override, expected
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("run_paper_eval", repo_root / "scripts/run_paper_eval.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    args = ["run_paper_eval.py", "--manifest", "manifest.json", "--data-root", "data", "--model", model, "--setting", setting, "--output-dir", str(tmp_path)]
    if override is not None:
        args.extend(["--threshold", str(override)])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(script, "load_records", lambda *_: [{
        "map_path": "experimental.map", "cif_gt_path": "deposited.cif", "cif_path": "inferred.cif",
        "sim_map_path": "deposited.mrc", "sim_map_path_cryoatom": "inferred.mrc", "labels_npz_path": "toy/labels.npz",
    }])
    observed = {}

    def predict(**kwargs):
        assert kwargs["threshold"] == override
        assert kwargs["receptor_mode"] == setting
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        np.savez(output / "grid_metadata.npz", origin=np.zeros(3), voxel_size=np.ones(3), hardmask=np.zeros((1, 1, 1)))
        return SimpleNamespace(prob_map=np.zeros((1, 1, 1), dtype=np.float32))

    def evaluate(**kwargs):
        observed["evaluation_threshold"] = kwargs["threshold"]
        return object()

    def write(_path, _results, threshold):
        observed["reported_threshold"] = threshold
        return {"summary": {}}

    monkeypatch.setattr(script, "run_prediction", predict)
    monkeypatch.setattr(script, "evaluate_probability_map", evaluate)
    monkeypatch.setattr(script, "write_eval_json", write)
    script.main()
    assert observed == {"evaluation_threshold": expected, "reported_threshold": expected}
