"""Recover backbone coordinates used by the receptor parser completeness checks."""

import numpy as np
from typing import Optional


def reconstruct_protein_backbone(
    residues: list[dict],
    res_idx: int,
    missing_atoms: list[str]
) -> dict[str, np.ndarray]:


    if len(residues) == 0 or res_idx < 0 or res_idx >= len(residues):
        return {}


    reconstructed = {}


    res = residues[res_idx]


    CA_C_BOND = 1.52
    
    C_N_BOND = 1.33
    
    N_CA_BOND = 1.46


    if 'CA' in missing_atoms:
        
        prev_ca = _get_prev_ca(residues, res_idx)
        
        next_ca = _get_next_ca(residues, res_idx)

        if prev_ca is not None and next_ca is not None:
            
            reconstructed['CA'] = (np.asarray(prev_ca, dtype=np.float64) + np.asarray(next_ca, dtype=np.float64)) / 2.0
        elif prev_ca is not None:
            
            reconstructed['CA'] = np.asarray(prev_ca, dtype=np.float64) + np.array([3.8, 0.0, 0.0])
        elif next_ca is not None:
            
            reconstructed['CA'] = np.asarray(next_ca, dtype=np.float64) - np.array([3.8, 0.0, 0.0])


    ca_raw = reconstructed.get('CA', res.get('ca_coord'))
    ca_coord = np.asarray(ca_raw, dtype=np.float64) if ca_raw is not None else None


    if 'N' in missing_atoms and ca_coord is not None:
        
        prev_c = _get_prev_c(residues, res_idx)

        if prev_c is not None:
            
            prev_c_arr = np.asarray(prev_c, dtype=np.float64)
            direction = ca_coord - prev_c_arr
            
            dist = np.linalg.norm(direction)

            if dist > 1e-6:
                
                direction = direction / dist
                
                reconstructed['N'] = prev_c_arr + direction * C_N_BOND
        else:
            
            reconstructed['N'] = ca_coord - np.array([N_CA_BOND, 0.0, 0.0])


    if 'C' in missing_atoms and ca_coord is not None:
        
        next_n = _get_next_n(residues, res_idx)

        if next_n is not None:
            
            next_n_arr = np.asarray(next_n, dtype=np.float64)
            direction = next_n_arr - ca_coord
            
            dist = np.linalg.norm(direction)

            if dist > 1e-6:
                
                direction = direction / dist
                
                reconstructed['C'] = ca_coord + direction * CA_C_BOND
        else:
            
            reconstructed['C'] = ca_coord + np.array([CA_C_BOND, 0.0, 0.0])

    return reconstructed


def reconstruct_nucleotide_backbone(
    residues: list[dict],
    res_idx: int,
    missing_atoms: list[str]
) -> dict[str, np.ndarray]:


    if len(residues) == 0 or res_idx < 0 or res_idx >= len(residues):
        return {}


    reconstructed = {}


    res = residues[res_idx]


    C4_C1_DIST = 2.5
    
    C1_N_DIST = 1.48
    
    BACKBONE_SPACING = 5.9


    if "C4'" in missing_atoms:
        
        prev_c4 = _get_prev_c4(residues, res_idx)
        
        next_c4 = _get_next_c4(residues, res_idx)

        if prev_c4 is not None and next_c4 is not None:
            
            reconstructed["C4'"] = (np.asarray(prev_c4, dtype=np.float64) + np.asarray(next_c4, dtype=np.float64)) / 2.0
        elif prev_c4 is not None:
            
            reconstructed["C4'"] = np.asarray(prev_c4, dtype=np.float64) + np.array([BACKBONE_SPACING, 0.0, 0.0])
        elif next_c4 is not None:
            
            reconstructed["C4'"] = np.asarray(next_c4, dtype=np.float64) - np.array([BACKBONE_SPACING, 0.0, 0.0])


    c4_raw = reconstructed.get("C4'", res.get("c4p_coord"))
    c4_coord = np.asarray(c4_raw, dtype=np.float64) if c4_raw is not None else None


    if "C1'" in missing_atoms and c4_coord is not None:
        
        reconstructed["C1'"] = c4_coord + np.array([0.0, C4_C1_DIST, 0.0])


    c1_raw = reconstructed.get("C1'", res.get("c1p_coord"))
    c1_coord = np.asarray(c1_raw, dtype=np.float64) if c1_raw is not None else None


    if ("N1" in missing_atoms or "N9" in missing_atoms) and c1_coord is not None:
        
        res_name = res.get('name', '')

        if res_name in {'A', 'G', 'DA', 'DG'}:


            reconstructed["N9"] = c1_coord + np.array([0.0, 0.0, C1_N_DIST])
        elif res_name in {'U', 'C', 'T', 'DT', 'DC'}:


            reconstructed["N1"] = c1_coord + np.array([0.0, 0.0, C1_N_DIST])

    return reconstructed


def _get_prev_ca(residues: list[dict], res_idx: int) -> Optional[np.ndarray]:


    if res_idx > 0:
        
        prev_res = residues[res_idx - 1]
        if prev_res.get('type') == 'protein':
            return prev_res.get('ca_coord')
    return None


def _get_next_ca(residues: list[dict], res_idx: int) -> Optional[np.ndarray]:


    if res_idx < len(residues) - 1:
        
        next_res = residues[res_idx + 1]
        if next_res.get('type') == 'protein':
            return next_res.get('ca_coord')
    return None


def _get_prev_c(residues: list[dict], res_idx: int) -> Optional[np.ndarray]:


    if res_idx > 0:
        
        prev_res = residues[res_idx - 1]
        if prev_res.get('type') == 'protein':
            return prev_res.get('c_coord')
    return None


def _get_next_n(residues: list[dict], res_idx: int) -> Optional[np.ndarray]:


    if res_idx < len(residues) - 1:
        
        next_res = residues[res_idx + 1]
        if next_res.get('type') == 'protein':
            return next_res.get('n_coord')
    return None


def _get_prev_c4(residues: list[dict], res_idx: int) -> Optional[np.ndarray]:


    if res_idx > 0:
        
        prev_res = residues[res_idx - 1]
        if prev_res.get('type') == 'nucleotide':
            return prev_res.get("c4p_coord")
    return None


def _get_next_c4(residues: list[dict], res_idx: int) -> Optional[np.ndarray]:


    if res_idx < len(residues) - 1:
        
        next_res = residues[res_idx + 1]
        if next_res.get('type') == 'nucleotide':
            return next_res.get("c4p_coord")
    return None
