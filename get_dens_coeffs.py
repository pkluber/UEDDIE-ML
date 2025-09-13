from density.params import DescriptorParams
from density.density import Density
from ase import Atoms
from ase.units import Bohr  # To convert between A and Bohrs
import numpy as np
import scipy
from scipy.special import sph_harm

from pathlib import Path
from collections import defaultdict
from io import TextIOWrapper
from typing import Tuple

DEFAULT_PARAMS = DescriptorParams(r_o=2.5, r_i=0.0, n_rad=4, n_l=3, gamma=0)
DEFAULT_ATOM_PARAMS = defaultdict(lambda: DEFAULT_PARAMS)
DEFAULT_ALIGN_METHOD = 'elf'

# Adapted from EDDIE-ML: https://github.com/lowkc/eddie-ml/blob/main/density/read_cubes.py#L10
def _read_cube_header(f: TextIOWrapper) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Read the title
    title = f.readline().strip()
    # skip the second line
    f.readline()

    def read_grid_line(line) -> Tuple[int, np.ndarray]:
        """Read a grid line from the cube file"""
        words = line.split()
        return (
            int(words[0]),
            np.array([float(words[1]), float(words[2]), float(words[3])], float)
            # all coordinates in a cube file are in atomic units
        )

    # number of atoms and origin of the grid
    natom, origin = read_grid_line(f.readline())
    # numer of grid points in A direction and step vector A, and so on
    shape0, axis0 = read_grid_line(f.readline())
    shape1, axis1 = read_grid_line(f.readline())
    shape2, axis2 = read_grid_line(f.readline())
    shape = np.array([shape0, shape1, shape2], int)
    axes = np.array([axis0, axis1, axis2])

    cell = np.array(axes*shape.reshape(-1,1))
    grid = shape

    def read_coordinate_line(line: str) -> Tuple[int, float, np.ndarray]:
        """Read an atom number and coordinate from the cube file"""
        words = line.split()
        return (
            int(words[0]), float(words[1]),
            np.array([float(words[2]), float(words[3]), float(words[4])], float)
            # all coordinates in a cube file are in atomic units
        )

    numbers = np.zeros(natom, int)
    pseudo_numbers = np.zeros(natom, float)
    coordinates = np.zeros((natom, 3), float)
    for i in range(natom):
        numbers[i], pseudo_numbers[i], coordinates[i] = read_coordinate_line(f.readline())
        # If the pseudo_number field is zero, we assume that no effective core
        # potentials were used.
        if pseudo_numbers[i] == 0.0:
            pseudo_numbers[i] = numbers[i]

    return origin, coordinates, numbers, cell, grid

# Also adapted from EDDIE-ML: https://github.com/lowkc/eddie-ml/blob/main/density/read_cubes.py#L59
def _read_cube_data(f: TextIOWrapper, grid: np.ndarray) -> np.ndarray:
    data = np.zeros(tuple(grid), float)
    tmp = data.ravel()
    counter = 0
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        words = line.split()
        for word in words:
            tmp[counter] = float(word)
            counter += 1
    return data

def get_density(cube_path: Path) -> Density:
    with open(cube_path) as fd:
        origin, _, _, cell, grid = _read_cube_header(fd)
        data = _read_cube_data(fd, grid)
        return Density(data, cell*Bohr, grid, origin*Bohr)

def get_atoms(cube_path: Path) -> Atoms:
    with open(cube_path) as fd:
        _, coordinates, numbers, _, _ = _read_cube_header(f)
    coordinates *= Bohr # Get atom positions in Ang
    atoms = Atoms(numbers, coordinates)
    return atoms

def g(r: float, r_i: float, r_o: float, a: int, gamma: float):
    def g_(r):
        return (r-r_i)**2 * (r_o-r)**(a+2) * np.exp(-gamma*(r/r_o)**(1/4))

    # Now normalize in L2 
    delta = (r_o-r_i)/1000
    r_grid = np.arange(r_i, r_o, delta)
    N = np.sqrt(np.sum(g_(r_grid)*g_(r_grid) * delta))
    
    # Return normalized g_ value 
    return g_(r) / N

def S(r_i: float, r_o: float, gamma: float, n_max: int):
    S = np.zeros((nmax, nmax))
    
    # Setup integration grid for left Riemannian sum 
    delta = (r_o-r_i)/1000
    r_grid = np.arange(r_i, r_o, delta)
    
    for i in range(n_max):
        g_i = g(r_grid, r_i, r_o, i+1, gamma)
        for j in range(i, n_max):
            g_j = g(r_grid, r_i, r_o, j+1, gamma)
            S[i,j] = np.sum(g_i * g_j * delta)

    for i in range(n_max):
        for j in range(j+1, n_max):
            S[j,i] = S[i,j]

    return S

def W(r_i: float, r_o: float, gamma: float, n_max: int):
    return scipy.linalg.sqrtm(np.linalg.pinv(S(r_i, r_o, gamma, n_max)))

def radials(r: np.ndarray, r_i: float, r_o: float, gamma: float, n_max: int):
    W = W(r_i, r_o, gamma, n_max)
    
    result = np.zeros([n_max] + list(r.shape))
    for k in range(n_max):
        rad = g(r, r_i, r_o, k+1, gamma)
        for j in range(n_max):
            result[j] += W[j, k] * rad

    result[:, r > r_o] = 0
    result[:, r < r_i] = 0

    return result

def calculate_dens_coeffs(cube_path: Path, params: dict[str, DescriptorParams] = DEFAULT_ATOM_PARAMS, align_method: str = DEFAULT_ALIGN_METHOD, overwrite: bool = False) -> bool:
    coeff_path = cube_path.parent / f'{cube_path.stem}.coeff'
    if coeff_path.is_file() and not overwrite:
        print(f'Found .coeff file for {cube_path.name}, not overwriting...')
        return True

    print(f'Calculating deformation density coefficients for {cube_path.name}...')
    
    density, atoms = get_density(cube_path), get_atoms(cube_path)
    atom_positions = atoms.get_positions()
    atom_species = atom.get_chemical_symbols()
    


def calculate_dens_coeffs_all(data_dir: Path, params: dict[str, DescriptorParams] = DEFAULT_ATOM_PARAMS, align_method: str = DEFAULT_ALIGN_METHOD, overwrite: bool = False):
    for path in data_dir.rglob('*.cube'):
        if path.is_file() and path.suffix == '.cube':
            calculate_dens_coeffs(path, params, align_method, overwrite=overwrite)
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Get the deformation density coefficients for all deformation density .cube files in a folder.')
    parser.add_argument('--align', type=str, default=DEFAULT_ALIGN_METHOD, help='ElF alignment method')
    parser.add_argument('--path', type=str, default='data', help='Path containing cube files.')
    parser.add_argument('--input', type=str, default='', help='Input file to use for generating a single .coeff file')
    parser.add_argument('--overwrite', type=bool, default=False, help='Whether to overwrite pre-existing .cube/.rho files')

    args = parser.parse_args()

    if len(args.input) > 0:
        calculate_dens_coeffs(Path(args.input), align_method=args.align, overwrite=args.overwrite)
    else:
        calculate_dens_coeffs_all(Path(args.path), align_method=args.align, overwrite=args.overwrite)

