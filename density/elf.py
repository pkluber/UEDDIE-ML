from .params import DescriptorParams

import numpy as np
from dataclasses import dataclass

@dataclass
class ElF():
    """ Class defining the electronic descriptors used by MLCF. ElF stands for ELectronic Fingerprint

    Parameters
    ----------

        value: dict or np.ndarray
             value of descriptor can either be a complex (dict) or real (np.ndarray) tensor.
        angles: np.ndarray (3)
            angles by which ElF was rotated into local coordinate system (used to rotate forces into same CS).
        basis: dict
            basis for elf representation
        species: str,
            atomic species (element symbol)
        unitcell: np.ndarray (3,3)
            unitcell of the system (used by fold_back_coords during alingment)
        position: np.ndarray (3,)
            position of the atom in the original xyz file

    """
    value: dict | np.ndarray
    angles: np.ndarray
    params: DescriptorParams
    species: str
    unitcell: np.ndarray
    position: np.ndarray
