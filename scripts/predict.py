"""Predict ligand density with a released Ligandseek model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ligand density and instances from an experimental map.")
    parser.add_argument("--model", choices=["unet_raw", "unet_diff", "unet_full", "ligandseek_pv"], default="unet_raw")
    parser.add_argument("--map", required=True, help="Experimental .map/.mrc density.")
    parser.add_argument("--receptor", "--receptor-cif", dest="receptor", help="Receptor .cif/.pdb; optional for raw, preserves the paper output constraint.")
    parser.add_argument("--sim", "--sim-map", dest="sim", help="Prepared receptor-only simulated map; required for diff/full/PV.")
    parser.add_argument("--atom-features", help="Prepared 49D receptor features; alternative to --receptor.")
    parser.add_argument("--checkpoint", "--checkpoint-dir", dest="checkpoint", default="artifacts/checkpoints", help="Released checkpoint file or directory.")
    parser.add_argument("--output", "--output-dir", dest="output", required=True)
    parser.add_argument("--receptor-mode", choices=["deposited", "inferred"], default="deposited")
    parser.add_argument("--threshold", type=float, help="Override the frozen threshold; raw defaults to 0.86.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from ligandseek.inference.prediction import run_prediction
    run_prediction(
        model_name=args.model,
        map_path=args.map,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        receptor_path=args.receptor,
        sim_path=args.sim,
        atom_feature_path=args.atom_features,
        receptor_mode=args.receptor_mode,
        threshold=args.threshold,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
