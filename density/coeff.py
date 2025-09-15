from .elf import ElF
from .utils import get_charge_from_position
import h5py
import json
import numpy as np

from dataclasses import asdict
from pathlib import Path

class CoeffWrapper:
    def __init__(self, path: Path):
        self.path = path
        self.elfs = []

        if path.is_file() and path.suffix == '.coeff':
            self.load()

    def load(self):
        with h5py.File(self.path, 'r') as fd:
            values = fd['value'][:]
            angles = fd['angles'][:]
            species = fd['species'][:]
            print(species)
            positions = fd['position'][:]
            charges = fd['charge'][:]

    def add_elf(self, elf: ElF):
        self.elfs.append(elf)

    def save(self):
        with h5py.File(self.path, 'w') as fd: 
            # Save params as metadata
            params = []
            for elf in self.elfs:
                params.append(elf.params)
            params = [asdict(p) for p in params]
            fd.attrs['params'] = json.dumps(params)
            
            # Save system name as metadata
            fd.attrs['system'] = self.path.stem
            
            # Now go to save the meat
            values = []
            angles = []
            species = []
            positions = []
            charges = [] 
            for elf in self.elfs:
                values.append(elf.value)
                angles.append(elf.angles)
                species.append(elf.species)
                positions.append(elf.position)
                charges.append(get_charge_from_position(self.path.parent / f'{self.path.stem}.xyz', elf.position))

            fd['value'] = np.array(values)
            fd['angles'] = np.array(angles)
            fd['species'] = species
            fd['position'] = np.array(positions)
            fd['charge'] = np.array(charges)

