from density.density import Density
from density.elf import ElF
from density.geom import get_nncs_angles, get_elfcs_angles, get_casimir
from density.geom import make_real, rotate_tensor, fold_back_coords, power_spectrum, transform
from density.params import DescriptorParams
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
        _, coordinates, numbers, _, _ = _read_cube_header(fd)
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
    S = np.zeros((n_max, n_max))
    
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
    W_matrix = W(r_i, r_o, gamma, n_max)
    
    result = np.zeros([n_max] + list(r.shape), dtype=np.complex128)
    for k in range(n_max):
        rad = g(r, r_i, r_o, k+1, gamma)
        for j in range(n_max):
            result[j] += W_matrix[j, k] * rad

    result[:, r > r_o] = 0
    result[:, r < r_i] = 0

    return result

# Adapted from EDDIE-ML: https://github.com/lowkc/eddie-ml/blob/main/density/real_space.py#482
def orient_elf(i, elf, all_pos, mode):
    '''
    Takes an ELF and orients it according to the rule specified in mode.

    Parameters
        i: int
            index of the atom in all_pos
        elf: ELF
            ELF to orient
        all_pos: np.ndarray
            positions of all atoms in the system
        mode: str
            {'elf' : use the ELF algorithm to orient the fingerprint
            'nn': use the nearest neighbour algorithm
            'casimir': take Casimir norm of complex tensor
            'neutral': keep alignment unchanged}
    
    Returns
        ELF
            oriented version of elf
    '''
    if mode == 'elf':
        angles_getter = get_elfcs_angles
    elif mode == 'nn':
        angles_getter = get_nncs_angles
    elif mode == 'neutral':
        pass
    elif mode == 'casimir':
        pass
    elif mode == 'power_spectrum':
        pass
    else:
        raise Exception('Unknown!! orientation mode {}'.format(mode))

    if (mode.lower() == 'neutral') or (mode == 'casimir') or (mode == 'power_spectrum'):
        angles = np.array([0,0,0])
    else:
        angles = angles_getter(i, fold_back_coords(i, all_pos, elf.unitcell), elf.value)

    if mode == 'casimir':
        oriented = get_casimir(elf.value)
        oriented = np.asarray(list(oriented.values()))
        elf_oriented = ElF(oriented, angles, elf.basis, elf.species, elf.unitcell, elf.position)
    elif mode == 'neutral':
        oriented = make_real(rotate_tensor(elf.value, np.array(angles), True))
        elf_oriented = ElF(oriented, angles, elf.basis, elf.species, elf.unitcell, elf.position)
    else:
        elf_transformed = transform(elf.value)
        elf_transformed = np.stack([val for val in elf_transformed.values()]).reshape(-1)
        n_l = elf.basis[f'n_l_{elf.species.lower()}']
        n = elf.basis[f'n_rad_{elf.species.lower()}']
        ps = power_spectrum(elf_transformed.reshape(1,-1), n_l-1, n, cgs=None)
        oriented = ps.reshape(-1)
        elf_oriented = ElF(oriented, angles, elf.basis, elf.species, elf.unitcell, elf.position)
    return elf_oriented

def calculate_dens_coeffs(cube_path: Path, params: dict[str, DescriptorParams] = DEFAULT_ATOM_PARAMS, align_method: str = DEFAULT_ALIGN_METHOD, overwrite: bool = False) -> bool:
    coeff_path = cube_path.parent / f'{cube_path.stem}.coeff'
    if coeff_path.is_file() and not overwrite:
        print(f'Found .coeff file for {cube_path.name}, not overwriting...')
        return True

    print(f'Calculating deformation density coefficients for {cube_path.name}...')
    
    density, atoms = get_density(cube_path), get_atoms(cube_path)
    atom_positions = atoms.get_positions()
    atom_species = atoms.get_chemical_symbols()
    
    X, Y, Z = density.mesh_3d()
    for i in range(len(atoms)):
        atom_pos = atom_positions[i]
        atom_element = atom_species[i]
        atom_params = params[atom_element]

        # Center coords around atom
        X_atom, Y_atom, Z_atom = (X - atom_pos[0]), (Y - atom_pos[1]), (Z - atom_pos[2])
        
        # Get density points within cutoff radius
        R_atom = np.sqrt(X_atom**2 + Y_atom**2 + Z_atom**2)

        # Calculate Theta (spherical coordinates)
        theta_eps = 1e-7
        Theta_atom = np.arccos(Z_atom/R_atom, where=(R_atom >= theta_eps))
        Theta_atom[R_atom < theta_eps] = 0

        # Calculate Phi (spherical coordinates)
        Phi_atom = np.arctan2(Y_atom, X_atom)

        # Apply mask to cut down on density points to evaluate, flattens grid shape
        mask = (R_atom < atom_params.r_o) * (R_atom >= atom_params.r_i)
        X_masked, Y_masked, Z_masked = X_atom[mask], Y_atom[mask], Z_atom[mask]
        R_masked, Theta_masked, Phi_masked = R_atom[mask], Theta_atom[mask], Phi_atom[mask]
        
        # Evaluate density at density points
        rho, V_cell = density.evaluate_at(X_masked, Y_masked, Z_masked)

        # Now get the spherical harmonics for our points
        angs = []
        for l in range(atom_params.n_l):
            angs.append([])
            for m in range(-l, l+1):
                angs[l].append(sph_harm(m, l, Phi_masked, Theta_masked))

        # Now get the radial part
        rads = radials(R_masked, atom_params.r_i, atom_params.r_o, atom_params.gamma, atom_params.n_rad)

        # Finally compute unaligned coefficients
        coeffs = {}
        for n in range(atom_params.n_rad):
            for l in range(atom_params.n_l):
                for m in range(2*l + 1):
                    coeffs[f'{n},{l},{m-l}'] = np.sum(angs[l][m]*rads[n]*rho) * V_cell

        elf = ElF(coeffs, [0, 0, 0], params, atom_element, density.unitcell, atom_pos)

        # Compute the aligned elf
        elf = orient_elf(i, elf, atom_positions, align_method)

        print(elf.value)

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

