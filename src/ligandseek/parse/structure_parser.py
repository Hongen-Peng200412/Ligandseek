"""Parse the first PDB/mmCIF model into receptor atoms and candidates using the paper rules."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import warnings

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Atom import DisorderedAtom

from .config import (
    ALLOWED_ELEMENTS,
    ALLOWED_RESIDUES,
    ELEMENT_TO_IDX,
    RESIDUE_TO_IDX,
    is_protein_residue,
    is_nucleotide_residue,
    is_purine,
    is_pyrimidine,
    Modified_Residues,
    COVALENT_BOND_THRESHOLD,
)
from .error_logger import return_error_info, ErrorType
from .ligand_candidates import LigandCandidate, find_all_hetatm_candidates, is_connected_to
from .coordinate_reconstruction import reconstruct_protein_backbone, reconstruct_nucleotide_backbone


@dataclass
class ParsedStructure:


    atom_coords: np.ndarray
    
    atom_elements: List[str]
    
    atom_res_indices: np.ndarray
    
    atom_chain_indices: np.ndarray
    
    atom_res_names: List[str]
    
    atom_names: List[str]


    res_coords: np.ndarray = None
    
    res_names: List[str] = field(default_factory=list)
    
    res_types: List[str] = field(default_factory=list)
    
    res_chain_indices: np.ndarray = None
    
    res_seq_numbers: np.ndarray = None


    backbone_n_coords: np.ndarray = None
    
    backbone_ca_coords: np.ndarray = None
    
    backbone_c_coords: np.ndarray = None


    backbone_c4p_coords: np.ndarray = None
    
    backbone_c1p_coords: np.ndarray = None
    
    backbone_n_base_coords: np.ndarray = None


    backbone_complete_mask: np.ndarray = None


    ligand_candidates: List[LigandCandidate] = field(default_factory=list)
    
    water_count: int = 0
    
    num_candidates: int = 0


def infer_element_from_atom_name(atom_name: str) -> Optional[str]:


    name = atom_name.strip().upper()
    if not name:
        return None
    
    if name[0] == 'H':
        return None
    if name[0].isdigit() and len(name) > 1 and name[1] == 'H':
        return None
    
    first_char = name[0]
    
    if first_char in ['C', 'N', 'O', 'S', 'P']:
        return first_char
    return None


def parse_structure(
    file_path: str,
    error_dir: str,
    sample_id: Optional[str] = None,
    require_ligand: bool = True,
    select_first_model: bool = False,
    allow_incomplete_backbone: bool = False,
    use_exclusion_resnames: bool = True,
    exclusion_resnames: Optional[Set[str]] = None,
    exclude_covalent_modified_residues: bool = True,
) -> Optional[ParsedStructure]:


    if sample_id is None:
        sample_id = Path(file_path).stem
    
    if not Path(file_path).exists():
        return_error_info(file_path, -1, ErrorType.FILE_NOT_FOUND,
                         f"File not found: {file_path}", error_dir, sample_id)
        return None
    
    file_ext = Path(file_path).suffix.lower()
    if file_ext in ['.pdb']:
        parser = PDBParser(QUIET=True)
    elif file_ext in ['.cif', '.mmcif']:
        parser = MMCIFParser(QUIET=True)
    else:
        return_error_info(file_path, -1, ErrorType.PARSE_ERROR,
                         f"Unsupported file format: {file_ext}", error_dir, sample_id)
        return None


    try:
        structure = parser.get_structure(sample_id, file_path)
    except Exception as e:
        return_error_info(file_path, -1, ErrorType.PARSE_ERROR,
                         f"Failed to parse structure: {str(e)}", error_dir, sample_id)
        return None
    
    try:
        model = structure[0]
    except (KeyError, IndexError):
        return_error_info(file_path, -1, ErrorType.EMPTY_STRUCTURE,
                         "No model found in structure", error_dir, sample_id)
        return None


    if not select_first_model:
        
        num_models = len(list(structure.get_models()))
        if num_models > 1:
            return_error_info(file_path, -1, ErrorType.PARSE_ERROR,
                             f"Structure contains {num_models} models, skipping (set select_first_model=True to use the first model)",
                             error_dir, sample_id)
            return ">1 model"


    ligand_candidates, water_count = find_all_hetatm_candidates(
        model,
        use_exclusion_resnames=use_exclusion_resnames,
        exclusion_resnames=exclusion_resnames,
        exclude_covalent_modified_residues=exclude_covalent_modified_residues,
    )
    
    if require_ligand and len(ligand_candidates) == 0:
        return_error_info(file_path, -1, ErrorType.NO_LIGAND,
                         "No candidate ligand detected after exclusion rules",
                         error_dir, sample_id)
        return None


    atom_data = []           
    residue_info = {}        
    residue_order = []       
    chain_to_idx = {}        
    
    for chain in model:
        chain_id = chain.get_id()
        if chain_id not in chain_to_idx:
            chain_to_idx[chain_id] = len(chain_to_idx)
        chain_idx = chain_to_idx[chain_id]
        
        for residue in chain:
            het_flag = residue.id[0]  
            res_seq = residue.id[1]   
            i_code = residue.id[2]    
            resname = residue.resname.strip().upper()  


            is_modified_receptor = (
                het_flag.startswith('H_')
                and resname in Modified_Residues
                and is_connected_to(residue, chain)
            )
            if het_flag != ' ' and not is_modified_receptor:
                continue


            if not (is_protein_residue(resname) or is_nucleotide_residue(resname)):
                
                continue


            res_key = (chain_id, res_seq, i_code)


            if res_key not in residue_info:
                residue_info[res_key] = {
                    'name': resname,
                    'chain_idx': chain_idx,
                    'res_seq': res_seq,
                    'type': 'protein' if is_protein_residue(resname) else 'nucleotide',
                    'atom_coords': [],
                    
                    'n_coord': None,
                    'ca_coord': None,
                    'c_coord': None,
                    
                    'c4p_coord': None,
                    'c1p_coord': None,
                    'n_base_coord': None,  
                }
                residue_order.append(res_key)


            for atom in residue:
                atom_name = atom.get_name().strip().upper()
                
                if isinstance(atom, DisorderedAtom):
                    coord = atom.disordered_get_list()[0].get_coord()
                else:
                    coord = atom.get_coord()
                
                element = None
                if hasattr(atom, 'element') and atom.element:
                    element = atom.element.strip().upper()
                    if element == 'SE':
                        element = 'S'  
                    elif element not in ALLOWED_ELEMENTS:
                        element = infer_element_from_atom_name(atom_name)
                else:
                    element = infer_element_from_atom_name(atom_name)
                
                if element is None:
                    continue


                residue_info[res_key]['atom_coords'].append(coord.tolist())
                
                if is_protein_residue(resname):
                    if atom_name == 'N':
                        residue_info[res_key]['n_coord'] = coord.tolist()
                    elif atom_name == 'CA':
                        residue_info[res_key]['ca_coord'] = coord.tolist()
                    elif atom_name == 'C':
                        residue_info[res_key]['c_coord'] = coord.tolist()
                else:  
                    if atom_name == "C4'":
                        residue_info[res_key]['c4p_coord'] = coord.tolist()
                    elif atom_name == "C1'":
                        residue_info[res_key]['c1p_coord'] = coord.tolist()
                    elif atom_name == 'N9' and is_purine(resname):
                        residue_info[res_key]['n_base_coord'] = coord.tolist()
                    elif atom_name == 'N1' and is_pyrimidine(resname):
                        residue_info[res_key]['n_base_coord'] = coord.tolist()


                res_idx = residue_order.index(res_key)
                atom_data.append({
                    'coords': coord.tolist(),
                    'element': element,
                    'res_idx': res_idx,
                    'chain_idx': chain_idx,
                    'res_name': resname,
                    'atom_name': atom_name,
                })


    if len(atom_data) == 0:
        return_error_info(file_path, -1, ErrorType.EMPTY_STRUCTURE,
                         "No valid atoms found", error_dir, sample_id)
        return None
    if len(residue_order) == 0:
        return_error_info(file_path, -1, ErrorType.EMPTY_STRUCTURE,
                         "No valid residues found", error_dir, sample_id)
        return None


    n_atoms = len(atom_data)
    n_residues = len(residue_order)


    atom_coords = np.array([d['coords'] for d in atom_data], dtype=np.float32)
    atom_elements = [d['element'] for d in atom_data]
    atom_res_indices = np.array([d['res_idx'] for d in atom_data], dtype=np.int32)
    atom_chain_indices = np.array([d['chain_idx'] for d in atom_data], dtype=np.int32)
    atom_res_names = [d['res_name'] for d in atom_data]
    atom_names = [d['atom_name'] for d in atom_data]


    res_names = []
    res_types = []
    res_coords = []
    res_chain_indices = []
    res_seq_numbers = []


    backbone_n_coords = []
    backbone_ca_coords = []
    backbone_c_coords = []
    backbone_c4p_coords = []
    backbone_c1p_coords = []
    backbone_n_base_coords = []
    backbone_complete_mask = []
    
    for res_key in residue_order:   
        res = residue_info[res_key]
        res_names.append(res['name'])
        res_types.append(res['type'])
        res_chain_indices.append(res['chain_idx'])
        res_seq_numbers.append(res['res_seq'])


        if res['type'] == 'protein':
            missing = []
            if res['n_coord'] is None:
                missing.append('N')
            if res['ca_coord'] is None:
                missing.append('CA')
            if res['c_coord'] is None:
                missing.append('C')

            if missing:
                if not allow_incomplete_backbone:
                    
                    return_error_info(
                        file_path,
                        -1,
                        ErrorType.INCOMPLETE_BACKBONE,
                        f"Protein residue {res['name']} at {res_key} missing backbone atoms: {','.join(missing)}",
                        error_dir,
                        sample_id,
                    )
                    return None
                else:


                    residues_list = [residue_info[k] for k in residue_order[:len(res_coords)]]
                    
                    current_idx_in_list = len(residues_list)
                    
                    current_pos = len(res_coords)
                    
                    context_list = residues_list + [res]
                    if current_pos + 1 < len(residue_order):
                        next_key = residue_order[current_pos + 1]
                        context_list.append(residue_info[next_key])
                    
                    reconstructed = reconstruct_protein_backbone(context_list, current_idx_in_list, missing)

                    if reconstructed:
                        
                        if 'N' in reconstructed:
                            res['n_coord'] = reconstructed['N']
                        if 'CA' in reconstructed:
                            res['ca_coord'] = reconstructed['CA']
                        if 'C' in reconstructed:
                            res['c_coord'] = reconstructed['C']
                        
                        backbone_complete_mask.append(False)
                    else:
                        
                        return_error_info(
                            file_path,
                            -1,
                            ErrorType.INCOMPLETE_BACKBONE,
                            f"Protein residue {res['name']} at {res_key} missing backbone atoms: {','.join(missing)} (reconstruction failed)",
                            error_dir,
                            sample_id,
                        )
                        return None
            else:
                
                backbone_complete_mask.append(True)

            res_coords.append(res['ca_coord'])

        else:  
            missing = []
            if res['c4p_coord'] is None:
                missing.append("C4'")
            if res['c1p_coord'] is None:
                missing.append("C1'")
            if res['n_base_coord'] is None:
                
                if is_purine(res['name']):
                    missing.append('N9')
                elif is_pyrimidine(res['name']):
                    missing.append('N1')
                else:
                    missing.append('N1/N9')  

            if missing:
                if not allow_incomplete_backbone:
                    
                    return_error_info(
                        file_path,
                        -1,
                        ErrorType.INCOMPLETE_BACKBONE,
                        f"Nucleotide residue {res['name']} at {res_key} missing backbone atoms: {','.join(missing)}",
                        error_dir,
                        sample_id,
                    )
                    return None
                else:


                    residues_list = [residue_info[k] for k in residue_order[:len(res_coords)]]
                    
                    current_idx_in_list = len(residues_list)
                    
                    current_pos = len(res_coords)
                    
                    context_list = residues_list + [res]
                    if current_pos + 1 < len(residue_order):
                        next_key = residue_order[current_pos + 1]
                        context_list.append(residue_info[next_key])
                    
                    reconstructed = reconstruct_nucleotide_backbone(context_list, current_idx_in_list, missing)

                    if reconstructed:
                        
                        if "C4'" in reconstructed:
                            res['c4p_coord'] = reconstructed["C4'"]
                        if "C1'" in reconstructed:
                            res['c1p_coord'] = reconstructed["C1'"]
                        if "N1" in reconstructed:
                            res['n_base_coord'] = reconstructed["N1"]
                        if "N9" in reconstructed:
                            res['n_base_coord'] = reconstructed["N9"]
                        
                        backbone_complete_mask.append(False)
                    else:
                        
                        return_error_info(
                            file_path,
                            -1,
                            ErrorType.INCOMPLETE_BACKBONE,
                            f"Nucleotide residue {res['name']} at {res_key} missing backbone atoms: {','.join(missing)} (reconstruction failed)",
                            error_dir,
                            sample_id,
                        )
                        return None
            else:
                
                backbone_complete_mask.append(True)

            res_coords.append(res['c4p_coord'])


        if res['type'] == 'protein':
            backbone_n_coords.append(res['n_coord'])
            backbone_ca_coords.append(res['ca_coord'])
            backbone_c_coords.append(res['c_coord'])
            
            backbone_c4p_coords.append([np.nan, np.nan, np.nan])
            backbone_c1p_coords.append([np.nan, np.nan, np.nan])
            backbone_n_base_coords.append([np.nan, np.nan, np.nan])
        elif res['type'] == 'nucleotide':
            backbone_c4p_coords.append(res['c4p_coord'])
            backbone_c1p_coords.append(res['c1p_coord'])
            backbone_n_base_coords.append(res['n_base_coord'])
            
            backbone_n_coords.append([np.nan, np.nan, np.nan])
            backbone_ca_coords.append([np.nan, np.nan, np.nan])
            backbone_c_coords.append([np.nan, np.nan, np.nan])


    result = ParsedStructure(
        
        atom_coords=atom_coords,                                           
        atom_elements=atom_elements,                                       
        atom_res_indices=atom_res_indices,                                 
        atom_chain_indices=atom_chain_indices,                             
        atom_res_names=atom_res_names,                                     
        atom_names=atom_names,                                             


        res_coords=np.array(res_coords, dtype=np.float32),                 
        res_names=res_names,                                               
        res_types=res_types,                                               
        res_chain_indices=np.array(res_chain_indices, dtype=np.int32),     
        res_seq_numbers=np.array(res_seq_numbers, dtype=np.int32),         


        backbone_n_coords=np.array(backbone_n_coords, dtype=np.float32),   
        backbone_ca_coords=np.array(backbone_ca_coords, dtype=np.float32), 
        backbone_c_coords=np.array(backbone_c_coords, dtype=np.float32),   


        backbone_c4p_coords=np.array(backbone_c4p_coords, dtype=np.float32),   
        backbone_c1p_coords=np.array(backbone_c1p_coords, dtype=np.float32),   
        backbone_n_base_coords=np.array(backbone_n_base_coords, dtype=np.float32), 
        backbone_complete_mask=np.array(backbone_complete_mask, dtype=bool),   


        ligand_candidates=ligand_candidates,                               
        water_count=water_count,                                           
        num_candidates=len(ligand_candidates),                             


    )
    
    return result
