from density.params import DescriptorParams

from pathlib import Path
from collections import defaultdict

DEFAULT_PARAMS = DescriptorParams(r_o=2.5, r_i=0.0, n_rad=4, n_l=3, gamma=0)
DEFAULT_ATOM_PARAMS = defaultdict(lambda: DEFAULT_PARAMS)
DEFAULT_ALIGN_METHOD = 'elf'

def calculate_dens_coeffs(cube_path: Path, params: dict[str, DescriptorParams] = DEFAULT_ATOM_PARAMS, align_method: str = DEFAULT_ALIGN_METHOD, overwrite: bool = False) -> bool:
    coeff_path = cube_path.parent / f'{cube_path.stem}.coeff'
    if coeff_path.is_file() and not overwrite:
        print(f'Found .coeff file for {cube_path.name}, not overwriting...')
        return True

    print(f'Calculating deformation density coefficients for {cube_path.name}...')


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

