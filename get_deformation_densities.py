import numpy as np
from pyscf import gto, lib
from pyscf.tools.cubegen import Cube

try:
    from gpu4pyscf import scf, mp, df, dft
    import cupy as cp
    print('GPU4PySCF detected and in use')
    GPU = True
except ImportError:
    from pyscf import scf, mp, df, dft
    print('GPU4PySCF not found and will not be used')
    GPU = False

from utils import get_charges

from pathlib import Path
from typing import Tuple

DEFAULT_RESOLUTION = 0.1
DEFAULT_EXTENSION = 5.0

def get_atom_monomers_and_dimer(xyz_file: Path) -> Tuple[str, str, str]:
    with open(xyz_file, 'r') as fd:
        lines = fd.readlines()

    num_atoms_m1 = int(lines[0])
    m1_start = 2 
    xyz_m1 = ''.join(lines[m1_start:m1_start+num_atoms_m1])
    
    num_atoms_m2 = int(lines[m1_start + num_atoms_m1])
    m2_start = m1_start + num_atoms_m1 + 2
    xyz_m2 = ''.join(lines[m2_start:m2_start+num_atoms_m2])

    xyz_dimer = xyz_m1+xyz_m2 

    return xyz_m1, xyz_m2, xyz_dimer 

def get_mol(atom: str, basis: str, charge: int) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.charge = charge
    mol.spin = 0  # shouldn't be any spin 
    mol.build()

    return mol

def get_monomers_and_dimer_mol(xyz_path: Path, basis: str = 'cc-pVTZ') -> Tuple[gto.Mole, gto.Mole, gto.Mole]:
    atom_m1, atom_m2, atom_dimer = get_atom_monomers_and_dimer(xyz_path)
    charge_m1, charge_m2, charge_dimer = get_charges(xyz_path.name)
    
    return get_mol(atom_m1, basis, charge_m1), get_mol(atom_m2, basis, charge_m2), get_mol(atom_dimer, basis, charge_dimer)

def density_or_none(mf: scf.hf.SCF) -> np.ndarray | None:
    if not mf.converged:
        return None
    return mf.make_rdm1(ao_repr=True)

def hf(mol: gto.Mole) -> np.ndarray | None:
    mf = scf.RHF(mol)
    mf.kernel()
    return density_or_none(mf)

def mp2(mol: gto.Mole) -> np.ndarray | None:
    mf = scf.RHF(mol)
    mf.kernel()
    mp2 = mp.MP2(mf)
    mp2.kernel()
    return density_or_none(mp2)

def pbe0(mol: gto.Mole) -> np.ndarray | None:
    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    return density_or_none(mf)

def lda(mol: gto.Mole) -> np.ndarray | None:
    mf = dft.RKS(mol)
    mf.xc = 'lda'
    mf = mf.newton()
    mf.kernel()
    return density_or_none(mf)

# Adapted from EDDIE-ML: https://github.com/lowkc/eddie-ml/blob/main/density/cube_utils.py#L238
def generate_uniform_grid(molecule: gto.Mole, spacing=0.2, extension=4, rotate=False, verbose=True):
    '''
    molecule = a pySCF mol object
    spacing = the increment between grid points
    extension = the amount to extend the cube on each side of the molecule
    rotate = when True, the molecule is rotated so the axes of the cube file
    are aligned with the principle axes of rotation of the molecule.
    '''
    numbers = molecule.atom_charges()
    pseudo_numbers = molecule.atom_charges()
    coordinates = molecule.atom_coords()
    # calculate the centre of mass of the nuclear charges
    totz = np.sum(pseudo_numbers)
    com = np.dot(pseudo_numbers, coordinates) / totz
    
    if rotate:
        # calculate moment of inertia tensor:
        itensor = np.zeros([3,3])
        for i in range(pseudo_numbers.shape[0]):
            xyz = coordinates[i] - com
            r = np.linalg.norm(xyz)**2.0
            tempitens = np.diag([r,r,r])
            tempitens -= np.outer(xyz.T, xyz)
            itensor += pseudo_numbers[i] * tempitens
        _, v = np.linalg.eigh(itensor)
        new_coords = np.dot((coordinates - com), v)
        axes = spacing * v
        
    else:
        # use the original coordinates
        new_coords = coordinates
        # compute the unit vectors of the cubic grid's coordinate system
        axes = np.diag([spacing, spacing, spacing])
        
    # max and min value of the coordinates
    max_coordinate = np.amax(new_coords, axis=0)
    min_coordinate = np.amin(new_coords, axis=0)
    # compute the required number of points along each axis
    shape = (max_coordinate - min_coordinate + 2.0*extension) / spacing
    shape = np.ceil(shape)
    shape = np.array(shape, int)
    origin = com - np.dot((0.5*shape), axes)
    
    npoints_x, npoints_y, npoints_z = shape
    npoints = npoints_x * npoints_y * npoints_z # total number of grid points
    
    points = np.zeros((npoints, 3)) # array to store coordinates of grid points
    coords = np.array(np.meshgrid(np.arange(npoints_x), np.arange(npoints_y),
                                np.arange(npoints_z)))
    coords = np.swapaxes(coords, 1, 2)
    coords = coords.reshape(3, -1)
    coords = coords.T
    points = coords.dot(axes)
    # compute coordinates of grid points relative to the origin
    points += origin

    if verbose:
        print('Cube origin: {}'.format(origin))
    
    return points, origin

def generate_density(cube: Cube, mol: gto.Mole, dm: np.ndarray) -> np.ndarray:
    nx, ny, nz = cube.nx, cube.ny, cube.nz 
    blksize = min(80000, cube.get_ngrids())

    rho = np.empty(cube.get_ngrids())
    for ip0, ip1 in lib.prange(0, cube.get_ngrids(), blksize):
        ao = mol.eval_gto('GTOval', cube.get_coords()[ip0:ip1]) 
        if GPU:
            ao = cp.asarray(ao).T

        rho_chunk = dft.numint.eval_rho(mol, ao, dm)
        if GPU:
            rho_chunk = rho_chunk.get()

        rho[ip0:ip1] = rho_chunk
    rho = rho.reshape(nx, ny, nz)

    return rho

def dimer_cube_difference(xyz_path: Path, method: str, resolution: float = DEFAULT_RESOLUTION, extension: float = DEFAULT_EXTENSION, overwrite: bool = False) -> bool:
    cube_path = xyz_path.parent / f'{xyz_path.stem}.cube'
    if cube_path.is_file() and not overwrite:
        print(f'Found .cube file for {xyz_path.name}, not overwriting...')
        return True

    print(f'Generating deformation density for {xyz_path.name}...')
    mol_m1, mol_m2, mol_dimer = get_monomers_and_dimer_mol(xyz_path)
    
    method = method.strip().upper()
    if method not in ['HF', 'MP2', 'PBE0', 'LDA']:
        raise ValueError('Methods currently implemented: HF, MP2, PBE0, and LDA only.')
    
    if method == 'HF':
        dm_m1, dm_m2, dm_dimer = hf(mol_m1), hf(mol_m2), hf(mol_dimer)
    elif method == 'MP2':
        dm_m1, dm_m2, dm_dimer = mp2(mol_m1), mp2(mol_m2), mp2(mol_dimer)
    elif method == 'PBE0':
        dm_m1, dm_m2, dm_dimer = pbe0(mol_m1), pbe0(mol_m2), pbe0(mol_dimer)
    else:
        dm_m1, dm_m2, dm_dimer = lda(mol_m1), lda(mol_m2), lda(mol_dimer)

    if dm_dimer is None or dm_m1 is None or dm_m2 is None:
        print(f'Calculations failed to converge for {xyz_path.name}!')
        return False  # Failed deformation density calculation if method failed to converge

    _, origin = generate_uniform_grid(mol_dimer, spacing=resolution, extension=extension, rotate=False, verbose=False)    

    cube_dimer = Cube(mol_dimer, resolution=resolution, margin=extension, origin=origin)
    nx, ny, nz = cube_dimer.nx, cube_dimer.ny, cube_dimer.nz
    box = np.diag(cube_dimer.box)
    rho_dimer = generate_density(cube_dimer, mol_dimer, dm_dimer)

    cube_m1 = Cube(mol_m1, nx, ny, nz, margin=extension, origin=origin, extent=box)
    rho_m1 = generate_density(cube_m1, mol_m1, dm_m1)

    cube_m2 = Cube(mol_m2, nx, ny, nz, margin=extension, origin=origin, extent=box)
    rho_m2 = generate_density(cube_m2, mol_m2, dm_m2)

    rho_def = rho_dimer - rho_m1 - rho_m2
    
    cube_dimer.write(rho_def, str(xyz_path.parent / f'{xyz_path.stem}.cube'), comment=f'{method}/cc-pVTZ deformation density for {xyz_path.name}')
    print(f'Done generating deformation density for {xyz_path.name}!')
    return True

def dimer_cube_differences(data_dir: Path, method: str, resolution: float = DEFAULT_RESOLUTION, extension: float = DEFAULT_EXTENSION, overwrite: bool = False):
    for path in data_dir.rglob('*.xyz'):
        if path.is_file() and path.suffix == '.xyz':
            dimer_cube_difference(path, method, resolution=resolution, extension=extension, overwrite=overwrite)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Get the deformation density for any dimer .xyz structure.')
    parser.add_argument('method', type=str, help='Density type. Choose from HF, MP2, LDA, or PBE0')
    parser.add_argument('--resolution', type=float, default=0.1, help='.cube resolution')
    parser.add_argument('--extension', type=float, default=5, help='Extension on sides of .cube file')
    parser.add_argument('--path', type=str, default='data/bcurves', help='Relative path to data directory')
    parser.add_argument('--input', type=str, default='', help='Input file to use for generating a single .cube file')
    parser.add_argument('--overwrite', type=bool, default=False, help='Whether to overwrite pre-existing .cube/.rho files')

    args = parser.parse_args()

    if len(args.input) > 0:
        dimer_cube_difference(Path(args.input), args.method, args.resolution, args.extension, args.overwrite)
    else:
        dimer_cube_differences(Path(args.path), args.method, args.resolution, args.extension, args.overwrite) 
