import numpy as np
import cupy as cp
from pyscf import gto, lib
from pyscf.tools.cubegen import Cube
from gpu4pyscf import scf, mp, df, dft

from pathlib import Path

import argparse

parser = argparse.ArgumentParser(description='Get the deformation density for any dimer .xyz structure.')
parser.add_argument('method', type=str, help='Density type. Choose from HF, MP2, LDA, or PBE0')
parser.add_argument('--resolution', type=float, default=0.1, help='.cube resolution')
parser.add_argument('--extension', type=float, default=5, help='Extension on sides of .cube file')

args = parser.parse_args()

DATA_DIR = Path('data')
DEFAULT_RESOLUTION = 0.1
DEFAULT_EXTENSION = 5.0

def dimer_cube_difference(path: Path, method: str, resolution: float = DEFAULT_RESOLUTION, extension: float = DEFAULT_EXTENSION):


def dimer_cube_differences(method: str, resolution: float = DEFAULT_RESOLUTION, extension: float = DEFAULT_EXTENSION):

    
