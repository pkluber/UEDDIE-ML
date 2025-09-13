from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class DescriptorParams:
    r_o: float    # Outer radial cutoff 
    r_i: float    # Inner radial cutoff
    n_rad: int    # Max radial degree to use 
    n_l: int      # Max angular degree to use
    gamma: float  # Dampening parameter

DEFAULT_PARAMS = DescriptorParams(r_o=2.5, r_i=0.0, n_rad=4, n_l=3, gamma=0)
DEFAULT_ATOM_PARAMS = defaultdict(lambda: DEFAULT_PARAMS)
DEFAULT_REORIENTATION_METHOD = 'elf'

def calculate_dens_coeffs(cube_path: Path, params: dict[str, DescriptorParams] = DEFAULT_ATOM_PARAMS, reorientation_method: str = DEFAULT_REORIENTATION_METHOD, overwrite: bool = False) -> bool:
    coeff_path = cube_path.parent / f'{cube_path.stem}.coeff'
    if coeff_path.is_file() and not overwrite:
        print(f'Found .coeff file for {cube_path.name}, not overwriting...')
        return True

    print(f'Calculating deformation density coefficients for {cube_path.name}...')


def calculate_dens_coeffs_all(data_dir: Path, params: dict[str, DescriptorParams] = DEFAULT_ATOM_PARAMS, reorientation_method: str = DEFAULT_REORIENTATION_METHOD, overwrite: bool = False):
    for path in data_dir.rglob('*.cube'):
        if path.is_file() and path.suffix == '.cube':
            calculate_dens_coeffs(path, params, alignment_method, overwrite=overwrite)
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Get the deformation density coefficients for all deformation density .cube files in a folder.')
    parser.add_argument('--reorient', type=str, default=DEFAULT_REORIENTATION_METHOD, help='ElF reorientation method')
    parser.add_argument('--path', type=str, default='data', help='Path containing cube files.')
    parser.add_argument('--input', type=str, default='', help='Input file to use for generating a single .coeff file')
    parser.add_argument('--overwrite', type=bool, default=False, help='Whether to overwrite pre-existing .cube/.rho files')

    args = parser.parse_args()

    if len(args.input) > 0:
        calculate_dens_coeffs(Path(args.input), reorientation_method=args.reorient, overwrite=args.overwrite)
    else:
        generate_dens_coeffs_all(Path(args.path), reorientation_method=args.reorient, overwrite=args.overwrite)

