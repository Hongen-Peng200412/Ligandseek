"""Compute the paper 49-dimensional atom features and logarithmic neighbor counts."""

import numpy as np
from typing import Optional, Tuple
from scipy.spatial import cKDTree

from .config import (
    NUM_ELEMENTS,
    NUM_RESIDUE_TYPES,
    ELEMENT_TO_IDX,
    RESIDUE_TO_IDX,
    ATOM_MASS,
    RESIDUE_PHYSIO,
    DENSITY_BIN_EDGES,
    ATOM_FEATURE_DIM,
    GRAPH_CUTOFF,
    get_element_onehot,
    get_residue_onehot,
    get_residue_physio,
    get_atom_mass,
)
from .structure_parser import ParsedStructure


DENSE_MATRIX_THRESHOLD = 8000  


def compute_local_density_sparse(
    atom_coords: np.ndarray,
    bin_edges: np.ndarray = DENSITY_BIN_EDGES
) -> np.ndarray:


    n_atoms = atom_coords.shape[0]
    
    n_bins = len(bin_edges) - 1
    if n_atoms == 0:
        return np.zeros((0, n_bins), dtype=np.float32)


    tree = cKDTree(atom_coords)
    
    density = np.zeros((n_atoms, n_bins), dtype=np.float32)
    
    cumulative_counts = np.zeros(n_atoms, dtype=np.int32)
    
    finite_edges = [e for e in bin_edges if e < np.inf]
    
    max_radius = finite_edges[-1] if finite_edges else 18.0
    
    neighbors_max = tree.query_ball_point(atom_coords, r=max_radius, workers=1)
    
    cumulative_counts = np.array([len(n) for n in neighbors_max], dtype=np.int32)


    prev_counts = cumulative_counts.copy()
    for i in range(n_bins - 1, -1, -1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        
        if upper >= np.inf:
            density[:, i] = 0
        else:
            if lower > 0:
                
                neighbors_lower = tree.query_ball_point(atom_coords, r=lower, workers=1)
                counts_lower = np.array([len(n) for n in neighbors_lower], dtype=np.int32)
                
                density[:, i] = (prev_counts - counts_lower).astype(np.float32)
                prev_counts = counts_lower
            else:
                
                density[:, i] = (prev_counts - 1).astype(np.float32)


    density = np.log1p(np.maximum(density, 0))  
    
    return density


def compute_parsed_atom_features(
    parsed_data: ParsedStructure,
    compute_density: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:


    n_atoms = len(parsed_data.atom_coords)
    
    if n_atoms == 0:
        return np.zeros((0, ATOM_FEATURE_DIM), dtype=np.float32)


    atom_features = np.zeros((n_atoms, ATOM_FEATURE_DIM), dtype=np.float32)


    offset_element = 0                                    
    offset_residue = offset_element + NUM_ELEMENTS        
    offset_physio = offset_residue + NUM_RESIDUE_TYPES    
    offset_mass = offset_physio + 8                       
    offset_density = offset_mass + 1                      
    
    for i in range(n_atoms):
        
        element = parsed_data.atom_elements[i]
        
        resname = parsed_data.atom_res_names[i]


        atom_features[i, offset_element:offset_element + NUM_ELEMENTS] = get_element_onehot(element)


        atom_features[i, offset_residue:offset_residue + NUM_RESIDUE_TYPES] = get_residue_onehot(resname)


        atom_features[i, offset_physio:offset_physio + 8] = get_residue_physio(resname)


        atom_features[i, offset_mass] = get_atom_mass(element)


    if compute_density:
        
        density = compute_local_density_sparse(parsed_data.atom_coords)
        atom_features[:, offset_density:offset_density + 9] = density
    
    return atom_features


def save_atoms_npz(
    parsed_data: ParsedStructure,
    atom_features: np.ndarray,
    output_path: str
) -> None:


    np.savez_compressed(
        output_path,
        coords=parsed_data.atom_coords,
        features=atom_features,
        elements=np.array(parsed_data.atom_elements, dtype=object),
        res_indices=parsed_data.atom_res_indices,
        chain_indices=parsed_data.atom_chain_indices,
        res_names=np.array(parsed_data.atom_res_names, dtype=object),
        atom_names=np.array(parsed_data.atom_names, dtype=object),
    )


def compute_atom_features(coords: np.ndarray, elements, residue_names, compute_density: bool = True) -> np.ndarray:
    from types import SimpleNamespace
    atoms = SimpleNamespace(atom_coords=np.asarray(coords, dtype=np.float32), atom_elements=list(elements), atom_res_names=list(residue_names))
    return compute_parsed_atom_features(atoms, compute_density=compute_density)


compute_local_density = compute_local_density_sparse


def save_atom_feature_npz(output_path, coords: np.ndarray, elements, residue_names, compute_density: bool = True) -> np.ndarray:
    from pathlib import Path
    elements = list(elements)
    residue_names = list(residue_names)
    features = compute_atom_features(coords, elements, residue_names, compute_density)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, coords=np.asarray(coords, dtype=np.float32), features=features, elements=np.asarray(elements), residue_names=np.asarray(residue_names))
    return features
