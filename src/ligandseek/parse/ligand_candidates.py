"""Extract HETATM candidates and detect covalently linked modified receptor residues."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from scipy.spatial import cKDTree
from Bio.PDB.Atom import DisorderedAtom

from .config import (
    WATER_RESIDUES,
    HETATM_EXCLUSION_LIST,
    Modified_Residues,
    METAL_ELEMENTS,
    COVALENT_BOND_THRESHOLD,
    is_protein_residue,
    is_nucleotide_residue,
)


def is_connected_to(residue, chain, threshold: float = COVALENT_BOND_THRESHOLD) -> bool:


    def _are_ids_adjacent(id1, id2) -> bool:

        seq1, ins1 = id1[1], id1[2]
        seq2, ins2 = id2[1], id2[2]
        
        diff = seq2 - seq1
        if diff == 1:
            return True
        elif diff == 0:
            if ins1 == ' ' and ins2 == 'A':
                return True
            if ins1 != ' ' and ins2 != ' ' and (ord(ins2) - ord(ins1) == 1):
                return True
        return False
    
    def _is_covalently_bonded(res1, res2, thresh: float) -> bool:


        if 'C' in res1 and 'N' in res2:
            try:
                diff = res1['C'] - res2['N']
                if diff < thresh:
                    return True
            except Exception:
                pass


        o3_names = ["O3'", "O3*", "O3"]
        for name in o3_names:
            if name in res1 and 'P' in res2:
                try:
                    diff = res1[name] - res2['P']
                    if diff < thresh:
                        return True
                except Exception:
                    pass
        
        return False


    res_list = list(chain.get_residues())
    
    try:
        idx = res_list.index(residue)
    except ValueError:
        return False


    curr_idx = idx
    while curr_idx > 0:
        prev_idx = curr_idx - 1
        curr_res = res_list[curr_idx]
        prev_res = res_list[prev_idx]
        
        if not _are_ids_adjacent(prev_res.id, curr_res.id):
            break
        
        if not _is_covalently_bonded(prev_res, curr_res, threshold):
            break
        
        if prev_res.id[0].startswith('H_'):
            curr_idx -= 1
        elif prev_res.id[0] == ' ':
            return True
        else:
            break


    curr_idx = idx
    while curr_idx < len(res_list) - 1:
        next_idx = curr_idx + 1
        curr_res = res_list[curr_idx]
        next_res = res_list[next_idx]
        
        if not _are_ids_adjacent(curr_res.id, next_res.id):
            break
        
        if not _is_covalently_bonded(curr_res, next_res, threshold):
            break
        
        if next_res.id[0].startswith('H_'):
            curr_idx += 1
        elif next_res.id[0] == ' ':
            return True
        else:
            break
    
    return False


@dataclass
class LigandCandidate:


    candidate_id: int
    resname: str
    chain_id: str
    res_id: int
    insertion_code: str


    coords: np.ndarray
    
    center: np.ndarray
    
    n_heavy_atoms: int


    molecular_weight: float = 0.0


    is_metal_ion: bool = False
    
    is_peptide_like: bool = False
    
    is_nucleotide_like: bool = False


    is_covalent: bool = False
    
    n_contact_receptor_atoms: int = 0
    
    n_contact_receptor_residues: int = 0


    polymer_length: int = 1


    has_only_organic_elements: Optional[bool] = None
    in_af3_ligand_exclusion: Optional[bool] = None
    in_af3_ion_list: Optional[bool] = None
    min_dist_to_receptor: Optional[float] = None
    resname_count_in_structure: Optional[int] = None


def _get_heavy_atom_coords(residue) -> np.ndarray:


    coords_list = []
    for atom in residue:
        
        atom_name = atom.get_name().strip().upper()
        
        if atom_name and atom_name[0] == 'H':
            continue
        if len(atom_name) > 1 and atom_name[0].isdigit() and atom_name[1] == 'H':
            continue
        
        if isinstance(atom, DisorderedAtom):
            coord = atom.disordered_get_list()[0].get_coord()
        else:
            coord = atom.get_coord()
        coords_list.append(coord)

    if len(coords_list) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.array(coords_list, dtype=np.float32)

def _get_atom_element(atom) -> Optional[str]:


    if hasattr(atom, 'element') and atom.element:
        return atom.element.strip().upper()
    return None


def _are_covalently_bonded_pair(res1, res2, threshold: float = COVALENT_BOND_THRESHOLD) -> bool:


    seq1 = res1.id[1]
    seq2 = res2.id[1]
    diff = seq2 - seq1
    if diff != 0 and diff != 1:
        return False


    if 'C' in res1 and 'N' in res2:
        try:
            dist = res1['C'] - res2['N']
            if dist < threshold:
                return True
        except Exception:
            pass


    for o3_name in ["O3'", "O3*", "O3"]:
        if o3_name in res1 and 'P' in res2:
            try:
                dist = res1[o3_name] - res2['P']
                if dist < threshold:
                    return True
            except Exception:
                pass

    return False


def _compute_molecular_weight(residue) -> float:


    total_mass = 0.0
    for atom in residue:
        
        atom_name = atom.get_name().strip().upper()
        
        if atom_name and atom_name[0] == 'H':
            continue
        if len(atom_name) > 1 and atom_name[0].isdigit() and atom_name[1] == 'H':
            continue
        
        if isinstance(atom, DisorderedAtom):
            total_mass += atom.disordered_get_list()[0].mass
        else:
            total_mass += atom.mass
    return total_mass


def _is_single_atom_metal(residue) -> bool:


    heavy_atoms = []
    for atom in residue:
        atom_name = atom.get_name().strip().upper()
        if atom_name and atom_name[0] == 'H':
            continue
        if len(atom_name) > 1 and atom_name[0].isdigit() and atom_name[1] == 'H':
            continue
        heavy_atoms.append(atom)

    if len(heavy_atoms) != 1:
        return False


    element = _get_atom_element(heavy_atoms[0])
    if element is None:
        return False
    return element in METAL_ELEMENTS


def compute_contact_attributes(
    candidates: List['LigandCandidate'],
    receptor_coords: np.ndarray,
    receptor_res_indices: np.ndarray,
    threshold: float,
) -> None:


    if len(candidates) == 0 or len(receptor_coords) == 0:
        return


    receptor_tree = cKDTree(receptor_coords)

    for candidate in candidates:
        
        lig_coords = candidate.coords
        if lig_coords.shape[0] == 0:
            candidate.n_contact_receptor_atoms = 0
            candidate.n_contact_receptor_residues = 0
            continue


        neighbor_lists = receptor_tree.query_ball_point(lig_coords, r=threshold)


        contacted_atom_indices = set()
        for neighbors in neighbor_lists:
            contacted_atom_indices.update(neighbors)


        candidate.n_contact_receptor_atoms = len(contacted_atom_indices)


        residue_hit_counts: Dict[int, int] = {}
        for atom_idx in contacted_atom_indices:
            
            res_idx = int(receptor_res_indices[atom_idx])
            residue_hit_counts[res_idx] = residue_hit_counts.get(res_idx, 0) + 1


        candidate.n_contact_receptor_residues = sum(
            1 for count in residue_hit_counts.values() if count >= 2
        )


def _compute_polymer_segments(model) -> Dict[Tuple, int]:


    polymer_lengths = {}

    for chain in model:
        
        chain_id = chain.get_id()
        
        res_list = list(chain.get_residues())


        segments = []
        
        current_segment = []

        for i, residue in enumerate(res_list):
            het_flag = residue.id[0]
            if not het_flag.startswith('H_'):
                
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                continue


            res_key = (chain_id, residue.id[1], residue.id[2])

            if not current_segment:
                
                current_segment = [res_key]
            else:
                
                prev_res = res_list[i - 1]
                prev_het = prev_res.id[0]

                if prev_het.startswith('H_') and _are_covalently_bonded_pair(prev_res, residue):
                    
                    current_segment.append(res_key)
                else:
                    
                    segments.append(current_segment)
                    current_segment = [res_key]


        if current_segment:
            segments.append(current_segment)


        for segment in segments:
            
            seg_len = len(segment)
            for res_key in segment:
                polymer_lengths[res_key] = seg_len

    return polymer_lengths


def find_all_hetatm_candidates(
    model,
    use_exclusion_resnames: bool = True,
    exclusion_resnames: Optional[Set[str]] = None,
    exclude_covalent_modified_residues: bool = True,
) -> Tuple[List[LigandCandidate], int]:


    polymer_lengths = _compute_polymer_segments(model)


    candidates = []
    
    global_id = 0
    
    water_count = 0
    
    active_exclusion_resnames = HETATM_EXCLUSION_LIST if exclusion_resnames is None else set(exclusion_resnames)

    for chain in model:
        
        chain_id = chain.get_id()

        for residue in chain:
            het_flag = residue.id[0]
            
            if not het_flag.startswith('H_'):
                continue
            
            resname = residue.resname.strip().upper()
            
            if use_exclusion_resnames and resname in active_exclusion_resnames:
                if resname in WATER_RESIDUES:
                    water_count += 1
                continue
            
            if exclude_covalent_modified_residues and resname in Modified_Residues and is_connected_to(residue, chain):
                continue


            coords = _get_heavy_atom_coords(residue)
            
            n_heavy = coords.shape[0]
            if n_heavy == 0:
                
                continue
            
            center = np.mean(coords, axis=0)


            is_metal = _is_single_atom_metal(residue)
            
            is_pep = is_protein_residue(resname)
            
            is_nuc = is_nucleotide_residue(resname)
            
            is_cov = is_connected_to(residue, chain)


            res_key = (chain_id, residue.id[1], residue.id[2])
            
            poly_len = polymer_lengths.get(res_key, 1)


            mw = _compute_molecular_weight(residue)


            candidate = LigandCandidate(
                candidate_id=global_id,
                resname=resname,
                chain_id=chain_id,
                res_id=residue.id[1],
                insertion_code=residue.id[2].strip(),
                coords=coords,
                center=center,
                n_heavy_atoms=n_heavy,
                molecular_weight=mw,
                is_metal_ion=is_metal,
                is_peptide_like=is_pep,
                is_nucleotide_like=is_nuc,
                is_covalent=is_cov,
                polymer_length=poly_len,
            )
            candidates.append(candidate)
            global_id += 1

    return candidates, water_count


def save_candidates_npz(
    candidates: List[LigandCandidate],
    water_count: int,
    output_path: str
) -> None:


    n = len(candidates)
    
    save_dict = {
        'n_candidates': np.int32(n),
        'water_count': np.int32(water_count),
    }
    if n == 0:
        np.savez_compressed(output_path, **save_dict)
        return


    save_dict['resnames'] = np.array([c.resname for c in candidates], dtype=object)
    save_dict['chain_ids'] = np.array([c.chain_id for c in candidates], dtype=object)
    save_dict['res_ids'] = np.array([c.res_id for c in candidates], dtype=np.int32)
    save_dict['n_heavy_atoms'] = np.array([c.n_heavy_atoms for c in candidates], dtype=np.int32)
    save_dict['is_metal_ion'] = np.array([c.is_metal_ion for c in candidates], dtype=bool)
    save_dict['is_peptide_like'] = np.array([c.is_peptide_like for c in candidates], dtype=bool)
    save_dict['is_nucleotide_like'] = np.array([c.is_nucleotide_like for c in candidates], dtype=bool)
    save_dict['is_covalent'] = np.array([c.is_covalent for c in candidates], dtype=bool)
    save_dict['polymer_length'] = np.array([c.polymer_length for c in candidates], dtype=np.int32)
    save_dict['molecular_weight'] = np.array([c.molecular_weight for c in candidates], dtype=np.float32)
    save_dict['insertion_codes'] = np.array([c.insertion_code for c in candidates], dtype=object)
    save_dict['centers'] = np.array([c.center for c in candidates], dtype=np.float32)


    for c in candidates:
        save_dict[f'candidate_coords_{c.candidate_id}'] = c.coords

    np.savez_compressed(output_path, **save_dict)


def load_candidates_npz(path: str) -> Tuple[List[LigandCandidate], int]:


    data = np.load(path, allow_pickle=True)
    
    n = int(data['n_candidates'])
    
    water_count = int(data['water_count'])

    if n == 0:
        return [], water_count


    candidates = []
    for i in range(n):
        
        coords = data[f'candidate_coords_{i}']
        
        mw = float(data['molecular_weight'][i]) if 'molecular_weight' in data else 0.0
        candidate = LigandCandidate(
            candidate_id=i,
            resname=str(data['resnames'][i]),
            chain_id=str(data['chain_ids'][i]),
            res_id=int(data['res_ids'][i]),
            insertion_code=str(data['insertion_codes'][i]),
            coords=coords,
            center=data['centers'][i],
            n_heavy_atoms=int(data['n_heavy_atoms'][i]),
            molecular_weight=mw,
            is_metal_ion=bool(data['is_metal_ion'][i]),
            is_peptide_like=bool(data['is_peptide_like'][i]),
            is_nucleotide_like=bool(data['is_nucleotide_like'][i]),
            is_covalent=bool(data['is_covalent'][i]),
            polymer_length=int(data['polymer_length'][i]),
        )
        candidates.append(candidate)

    return candidates, water_count
