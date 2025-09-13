import numpy as np
from pathlib import Path
from typing import Tuple

POS_CHARGED_AMINOS = ['ARG', 'LYS']
NEG_CHARGED_AMINOS = ['ASP', 'GLU']

def get_amino_charge(amino: str) -> int:
    if amino in POS_CHARGED_AMINOS:
        return 1
    elif amino in NEG_CHARGED_AMINOS:
        return -1
    else:
        return 0

def get_charges(filename: str) -> Tuple[int, int, int]:
    if filename.startswith('S66'):
        return 0, 0, 0
    elif filename.startswith('SSI'):
        split = filename.split('-')
        aa1 = split[1][3:]
        aa2 = split[2][3:]
        charge1 = get_amino_charge(aa1)
        charge2 = get_amino_charge(aa2)
        if charge1 == charge2 or charge1 + charge2 != 0: 
            return 0, 0, 0
        else:
            return charge1, charge2, 0
    elif filename.startswith('C_'): # IL174
        return 1, -1, 0
    elif filename.startswith('C'): # extraILs
        if filename.startswith('C0491_A0090') or filename.startswith('C2004_A0073'):
            return -1, 1, 0
        else:
            return 1, -1, 0

def get_charge_from_position(xyz_path: Path, position: np.ndarray) -> int | None:
    position = np.array(position)

    charges = get_charges(xyz_path.name)

    with open(xyz_path) as fd:
        lines = fd.readlines() # note preserves \n characters 
        try:
            num_atoms_m1 = int(lines[0])
            m1_start = 2
            for line in lines[m1_start:m1_start+num_atoms_m1]:
                split = line.strip().split()
                xyz = np.array([float(num) for num in split[1:]])
                if np.linalg.norm(xyz - position) < 1e-6:
                    return charges[0]

            return charges[1] 
        except ValueError:
            print(f'Error trying to parse xyz file {path}')
            return None
