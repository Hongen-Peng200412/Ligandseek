"""Scientific constants for the paper receptor atom features and residue selection."""

import numpy as np
from typing import Dict, List


ALLOWED_ELEMENTS: List[str] = ['C', 'N', 'O', 'S', 'P', 'X']


NUM_ELEMENTS: int = len(ALLOWED_ELEMENTS)  


ELEMENT_TO_IDX: Dict[str, int] = {e: i for i, e in enumerate(ALLOWED_ELEMENTS)}


ATOM_MASS: Dict[str, float] = {
    'C': 12.011 / 32.0,   
    'N': 14.007 / 32.0,   
    'O': 15.999 / 32.0,   
    'S': 32.065 / 32.0,   
    'P': 30.974 / 32.0,   
    'X': 14.0 / 32.0,     
}


AMINO_ACIDS: List[str] = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
]


NUCLEOTIDES: List[str] = ['A', 'U', 'C', 'G']


DNA_NUCLEOTIDES: List[str] = ['DA', 'DT', 'DC', 'DG']


MODIFIED_RESIDUE_TO_PARENT: Dict[str, str] = {
    
    'MSE': 'MET',  
    'SEP': 'SER',  
    'TPO': 'THR',  
    'PTR': 'TYR',  
    'MLY': 'LYS',  
    'M3L': 'LYS',  
    'KCX': 'LYS',  
    'HYP': 'PRO',  
    'FME': 'MET',  
    'CME': 'CYS',  
    'CSO': 'CYS',  
    'OCS': 'CYS',  
    'SEC': 'CYS',  
    'PYL': 'LYS',  
    'PCA': 'GLU',  


    'PSU': 'U',    
    '5MC': 'C',    
    '5MU': 'U',    
    '1MA': 'A',    
    '2MG': 'G',    
    '7MG': 'G',    
    'M2G': 'G',    
    'OMG': 'G',    
    'OMC': 'C',    
}


Modified_Residues: set = set(MODIFIED_RESIDUE_TO_PARENT.keys())


RESIDUE_TYPES: List[str] = AMINO_ACIDS + NUCLEOTIDES + ['X']


NUM_RESIDUE_TYPES: int = len(RESIDUE_TYPES)  


RESIDUE_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate(RESIDUE_TYPES)}


RESIDUE_TO_IDX['DT'] = RESIDUE_TO_IDX['U']      
RESIDUE_TO_IDX['DA'] = RESIDUE_TO_IDX['A']      
RESIDUE_TO_IDX['DC'] = RESIDUE_TO_IDX['C']      
RESIDUE_TO_IDX['DG'] = RESIDUE_TO_IDX['G']      


for modified_resname, parent_resname in MODIFIED_RESIDUE_TO_PARENT.items():
    if parent_resname in RESIDUE_TO_IDX:
        RESIDUE_TO_IDX[modified_resname] = RESIDUE_TO_IDX[parent_resname]


ALLOWED_RESIDUES: set = (
    set(AMINO_ACIDS)
    | set(NUCLEOTIDES)
    | set(DNA_NUCLEOTIDES)
    | set(Modified_Residues)
    | {'X'}
)


AA_PHYSIO: Dict[str, List[int]] = {
    'ALA': [0, 1, 0, 0, 1, 0, 0, 1],  
    'ARG': [1, 0, 0, 1, 0, 1, 0, 0],  
    'ASN': [1, 0, 0, 0, 1, 0, 0, 1],  
    'ASP': [1, 0, 1, 0, 0, 0, 1, 0],  
    'CYS': [1, 0, 1, 0, 0, 0, 0, 1],  
    'GLN': [1, 0, 0, 0, 1, 0, 0, 1],  
    'GLU': [1, 0, 1, 0, 0, 0, 1, 0],  
    'GLY': [0, 1, 0, 0, 1, 0, 0, 1],  
    'HIS': [1, 0, 0, 1, 0, 0, 0, 1],  
    'ILE': [0, 1, 0, 0, 1, 0, 0, 1],  
    'LEU': [0, 1, 0, 0, 1, 0, 0, 1],  
    'LYS': [1, 0, 0, 1, 0, 1, 0, 0],  
    'MET': [0, 1, 0, 0, 1, 0, 0, 1],  
    'PHE': [0, 1, 0, 0, 1, 0, 0, 1],  
    'PRO': [0, 1, 0, 0, 1, 0, 0, 1],  
    'SER': [1, 0, 0, 0, 1, 0, 0, 1],  
    'THR': [1, 0, 0, 0, 1, 0, 0, 1],  
    'TRP': [0, 1, 0, 0, 1, 0, 0, 1],  
    'TYR': [1, 0, 1, 0, 0, 0, 0, 1],  
    'VAL': [0, 1, 0, 0, 1, 0, 0, 1],  
    'MSE': [0, 1, 0, 0, 1, 0, 0, 1],  
}


NUCLEOTIDE_PHYSIO: Dict[str, List[int]] = {
    'A':  [1, 0, 0, 1, 0, 0, 0, 1],  
    'U':  [1, 0, 0, 0, 1, 0, 0, 1],  
    'C':  [1, 0, 0, 0, 1, 0, 0, 1],  
    'G':  [1, 0, 0, 1, 0, 0, 0, 1],  
    'DA': [1, 0, 0, 1, 0, 0, 0, 1],  
    'DT': [1, 0, 0, 0, 1, 0, 0, 1],  
    'DC': [1, 0, 0, 0, 1, 0, 0, 1],  
    'DG': [1, 0, 0, 1, 0, 0, 0, 1],  
}


RESIDUE_PHYSIO: Dict[str, List[int]] = {**AA_PHYSIO, **NUCLEOTIDE_PHYSIO}
RESIDUE_PHYSIO['X'] = [0, 0, 0, 0, 1, 0, 0, 1]  


for modified_resname, parent_resname in MODIFIED_RESIDUE_TO_PARENT.items():
    if parent_resname in RESIDUE_PHYSIO:
        RESIDUE_PHYSIO[modified_resname] = RESIDUE_PHYSIO[parent_resname]


ELEMENT_ONEHOT_DIM: int = 6

RESIDUE_ONEHOT_DIM: int = 25

PHYSIO_DIM: int = 8

MASS_DIM: int = 1

DENSITY_DIM: int = 9


ATOM_FEATURE_DIM: int = ELEMENT_ONEHOT_DIM + RESIDUE_ONEHOT_DIM + PHYSIO_DIM + MASS_DIM + DENSITY_DIM  

RESIDUE_FEATURE_DIM: int = RESIDUE_ONEHOT_DIM + PHYSIO_DIM  


PROTEIN_BACKBONE_ATOMS: List[str] = ['N', 'CA', 'C']


NUCLEOTIDE_BACKBONE_ATOMS: List[str] = ["C4'", "C1'", "N1", "N9"]

PURINES: set = {'A', 'G', 'DA', 'DG'}

PYRIMIDINES: set = {'U', 'C', 'DT', 'DC'}


COVALENT_BOND_THRESHOLD: float = 1.8


GRAPH_CUTOFF: float = 10.0


MAX_DISTANCE_CUTOFF: float = 40.0


DENSITY_BIN_EDGES: np.ndarray = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0], dtype=np.float32)


WATER_RESIDUES: set = {'HOH', 'WAT', 'H2O', 'DOD'}


METAL_ELEMENTS: set = {
    
    'LI', 'NA', 'K', 'RB', 'CS',
    
    'BE', 'MG', 'CA', 'SR', 'BA',
    
    'SC', 'TI', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
    'Y', 'ZR', 'MO', 'RU', 'RH', 'PD', 'AG', 'CD',
    'HF', 'TA', 'W', 'RE', 'OS', 'IR', 'PT', 'AU', 'HG',
    
    'LA', 'CE', 'PR', 'ND', 'SM', 'EU', 'GD', 'TB', 'DY',
    'HO', 'ER', 'TM', 'YB', 'LU',
    
    'U',
    
    'AL', 'GA', 'IN', 'SN', 'TL', 'PB', 'BI',
}


HETATM_EXCLUSION_LIST: set = {
    
    'HOH', 'WAT', 'H2O', 'DOD',


    'GOL', 'EDO', 'MPD', 'PEG', 'PG4', 'P6G',
    'TRS', 'MES', 'HEP', 'ACT', 'CIT',
    'EOH', 'MOH', 'IPA', 'DMS', 'DTT', 'BME',
    'BU1', 'TBU',


    'UNX', 'UNL', 'UNK',


    'XE', 'KR',
}


CLASSIC_LIGAND_EXEMPTION_LIST: set = {
    'HEM', 'FAD', 'NAD', 'NADP', 'NAP', 'NDP',
    'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'GMP',
    'FES', 'SF4', 'PLP', 'MGD', 'COA', 'SAM',
}


def normalize_residue_name(resname: str) -> str:


    normalized = resname.upper().strip()
    return MODIFIED_RESIDUE_TO_PARENT.get(normalized, normalized)


def get_residue_onehot(resname: str) -> np.ndarray:


    onehot = np.zeros(NUM_RESIDUE_TYPES, dtype=np.float32)
    
    normalized_resname = normalize_residue_name(resname)
    
    idx = RESIDUE_TO_IDX.get(normalized_resname, RESIDUE_TO_IDX['X'])
    onehot[idx] = 1.0
    return onehot


def get_element_onehot(element: str) -> np.ndarray:


    onehot = np.zeros(NUM_ELEMENTS, dtype=np.float32)
    
    idx = ELEMENT_TO_IDX.get(element.upper(), ELEMENT_TO_IDX['X'])
    onehot[idx] = 1.0
    return onehot


def get_residue_physio(resname: str) -> np.ndarray:


    normalized_resname = normalize_residue_name(resname)
    
    physio = RESIDUE_PHYSIO.get(normalized_resname, RESIDUE_PHYSIO['X'])
    return np.array(physio, dtype=np.float32)


def get_atom_mass(element: str) -> float:


    return ATOM_MASS.get(element.upper(), ATOM_MASS['X'])


def is_protein_residue(resname: str) -> bool:


    normalized_resname = normalize_residue_name(resname)
    return normalized_resname in set(AMINO_ACIDS)


def is_nucleotide_residue(resname: str) -> bool:


    normalized_resname = normalize_residue_name(resname)
    return normalized_resname in set(NUCLEOTIDES) | set(DNA_NUCLEOTIDES)


def is_purine(resname: str) -> bool:

    normalized_resname = normalize_residue_name(resname)
    return normalized_resname in PURINES


def is_pyrimidine(resname: str) -> bool:

    normalized_resname = normalize_residue_name(resname)
    return normalized_resname in PYRIMIDINES
