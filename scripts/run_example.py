"""Run a prepared example through the same CLI used for new experimental maps."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("unet_raw", "unet_diff", "unet_full", "ligandseek_pv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("8z0f", "7z24", "9bxc"), default="8z0f")
    parser.add_argument("--source", choices=("deposited", "inferred", "both"), default="deposited")
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--assets-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, help="Override the default for each selected model.")
    args = parser.parse_args()
    example = args.assets_root / "examples" / args.example
    models = MODELS if args.model == "all" else (args.model,)
    sources = ("deposited", "inferred") if args.source == "both" else (args.source,)
    required = [example / "experimental.map", example / "ground_truth.npz"]
    for source in sources:
        required.append(example / source / "receptor.cif")
        if any(model != "unet_raw" for model in models):
            required.append(example / source / "simulated.mrc")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing example inputs: {missing}. Run scripts/download_assets.py first.")
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from ligandseek.inference.paper_eval import evaluate_probability_map

    summary = []
    for source in sources:
        for model in models:
            output = args.output_root / args.example / source / model
            command = [sys.executable, str(ROOT / "scripts" / "predict.py"), "--model", model, "--map", str(example / "experimental.map"), "--receptor", str(example / source / "receptor.cif"), "--checkpoint", str(args.assets_root / "checkpoints"), "--output", str(output), "--receptor-mode", source, "--device", args.device, "--batch-size", str(args.batch_size), "--seed", str(args.seed)]
            if model != "unet_raw":
                command.extend(["--sim", str(example / source / "simulated.mrc")])
            if args.threshold is not None:
                command.extend(["--threshold", str(args.threshold)])
            subprocess.run(command, cwd=ROOT, check=True)
            with np.load(output / "ligand_pred.npz", allow_pickle=False) as prediction, np.load(output / "grid_metadata.npz", allow_pickle=False) as grid:
                metadata = json.loads((output / "run_metadata.json").read_text())
                result = evaluate_probability_map(sample_id=args.example, probability=prediction["ligand_pred"], labels_npz_path=example / "ground_truth.npz", origin=grid["origin"], voxel_size=grid["voxel_size"], threshold=metadata["threshold"], receptor_occupancy_mask=grid["hardmask"] > 0)
            metrics = asdict(result)
            (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            summary.append({"example": args.example, "source": source, "model": model, "threshold": metadata["threshold"], "dice": result.dice, "pr_auc": result.pr_auc, "num_candidates": result.candidates, "top3_success_06": result.top3_success_06})
            print(json.dumps(summary[-1]), flush=True)
    summary_path = args.output_root / args.example / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(summary_path.read_text()) if summary_path.is_file() else []
    updates = {(row["source"], row["model"]): row for row in previous}
    updates.update({(row["source"], row["model"]): row for row in summary})
    summary_path.write_text(json.dumps(list(updates.values()), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
