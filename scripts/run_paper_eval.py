"""Evaluate a supplied split through the single-map prediction path."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from ligandseek.inference.paper_eval import evaluate_probability_map, load_records, write_eval_json
from ligandseek.inference.postprocess import PAPER_THRESHOLDS
from ligandseek.inference.prediction import run_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON records with paths relative to --data-root.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    parser.add_argument("--model", required=True, choices=tuple(PAPER_THRESHOLDS))
    parser.add_argument("--setting", default="deposited", choices=("deposited", "inferred"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, help="Override the frozen model/setting threshold.")
    args = parser.parse_args()
    threshold = PAPER_THRESHOLDS[args.model][args.setting] if args.threshold is None else args.threshold
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.manifest, args.data_root)
    if args.limit is not None:
        records = records[:args.limit]
    results = []
    for record in records:
        sample_id = Path(record["labels_npz_path"]).parent.name
        sample_dir = output_dir / sample_id
        receptor_path = record["cif_gt_path"] if args.setting == "deposited" else record["cif_path"]
        sim_path = None if args.model == "unet_raw" else record["sim_map_path" if args.setting == "deposited" else "sim_map_path_cryoatom"]
        post = run_prediction(model_name=args.model, map_path=record["map_path"], checkpoint_path=args.checkpoint_dir, output_dir=sample_dir, receptor_path=receptor_path, sim_path=sim_path, atom_feature_path=None, receptor_mode=args.setting, threshold=args.threshold, device=args.device, batch_size=args.batch_size, seed=args.seed)
        with np.load(sample_dir / "grid_metadata.npz", allow_pickle=False) as grid:
            result = evaluate_probability_map(sample_id=sample_id, probability=post.prob_map, labels_npz_path=record["labels_npz_path"], origin=grid["origin"], voxel_size=grid["voxel_size"], threshold=threshold, receptor_occupancy_mask=grid["hardmask"] > 0)
        results.append(result)
    if results:
        metrics_path = output_dir / "metrics.json"
        payload = write_eval_json(metrics_path, results, threshold=threshold)
        print(json.dumps({"threshold": threshold, "metrics_file": str(metrics_path), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
