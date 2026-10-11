from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ligandseek.parse.atom_features import compute_atom_features
from ligandseek.parse.structure_parser import parse_structure


@dataclass(frozen=True)
class ReceptorAtomFeatures:
    coords: np.ndarray
    features: np.ndarray
    elements: np.ndarray
    residue_names: np.ndarray


def load_receptor_atom_features(path: str | Path) -> ReceptorAtomFeatures:
    data = np.load(path, allow_pickle=True)
    coords = np.asarray(data["coords"], dtype=np.float32)
    features = np.asarray(data["features"], dtype=np.float32)
    elements = np.asarray(data.get("elements", np.full((coords.shape[0],), "X", dtype=object)), dtype=object)
    residue_names = np.asarray(data.get("residue_names", data.get("res_names", np.full((coords.shape[0],), "X", dtype=object))), dtype=object)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Receptor feature npz must contain coords with shape (N, 3).")
    if features.ndim != 2 or features.shape[0] != coords.shape[0] or features.shape[1] != 49:
        raise ValueError("Receptor feature npz must contain features with shape (N, 49).")
    return ReceptorAtomFeatures(coords=coords, features=features, elements=elements, residue_names=residue_names)


def load_receptor_from_structure(path: str | Path, compute_density: bool = True, error_dir: str | Path | None = None) -> ReceptorAtomFeatures:
    coords, elements, residue_names = parse_receptor_atoms(path, error_dir=error_dir)
    features = compute_atom_features(
        coords=coords,
        elements=elements,
        residue_names=residue_names,
        compute_density=compute_density,
    )
    return ReceptorAtomFeatures(
        coords=coords,
        features=features,
        elements=np.asarray(elements, dtype=object),
        residue_names=np.asarray(residue_names, dtype=object),
    )


def parse_receptor_atoms(path: str | Path, error_dir: str | Path | None = None) -> tuple[np.ndarray, list[str], list[str]]:
    """Select receptor heavy atoms using the paper first-model, modified-residue, and altloc rules."""
    parsed = parse_structure(
        str(path),
        str(error_dir) if error_dir is not None else str(Path(path).parent),
        sample_id=Path(path).stem,
        require_ligand=False,
        select_first_model=True,
        allow_incomplete_backbone=True,
    )
    if parsed is None:
        raise RuntimeError(f"Cannot extract receptor atoms from {path}; inspect error_logs for details.")
    return parsed.atom_coords, parsed.atom_elements, parsed.atom_res_names


def save_receptor_atom_features(
    output_path: str | Path,
    features: ReceptorAtomFeatures,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        coords=features.coords.astype(np.float32),
        features=features.features.astype(np.float32),
        elements=np.asarray(features.elements, dtype=object),
        residue_names=np.asarray(features.residue_names, dtype=object),
        feature_layout=np.asarray(
            "0:6 element_onehot; 6:31 residue_onehot; 31:39 residue_properties; 39 atom_mass; 40:49 local_density",
            dtype=object,
        ),
    )


