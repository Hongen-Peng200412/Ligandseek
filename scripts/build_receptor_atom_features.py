from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ligandseek.parse.atom_features import save_atom_feature_npz
from ligandseek.parse.receptor_features import load_receptor_from_structure, save_receptor_atom_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 49-dimensional receptor atom features from an atom table.")
    parser.add_argument("--input", required=True, help="CSV/JSON atom table or receptor .cif/.mmcif/.pdb structure.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--no-density", action="store_true", help="Set the nine local-density channels to zero.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.suffix.lower() in {".cif", ".mmcif", ".pdb", ".ent"}:
        features = load_receptor_from_structure(input_path, compute_density=not args.no_density)
        save_receptor_atom_features(args.output, features)
        print(f"atoms={features.features.shape[0]} feature_dim={features.features.shape[1]} output={args.output}")
        return

    rows = _read_rows(input_path)
    coords = np.asarray([[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows], dtype=np.float32)
    elements = [row["element"] for row in rows]
    residues = [row["resname"] for row in rows]
    features = save_atom_feature_npz(
        args.output,
        coords=coords,
        elements=elements,
        residue_names=residues,
        compute_density=not args.no_density,
    )
    print(f"atoms={features.shape[0]} feature_dim={features.shape[1]} output={args.output}")


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
