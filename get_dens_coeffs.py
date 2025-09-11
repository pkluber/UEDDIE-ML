from pathlib import Path
from dataclasses import dataclass

@dataclass
class DescriptorParams:
    r_o: float    # Outer radial cutoff 
    r_i: float    # Inner radial cutoff
    n_rad: int    # Max radial degree to use 
    n_l: int      # Max angular degree to use
    gamma: float  # Dampening parameter

@dataclass
class AtomDescriptorParams:
    params_h: DescriptorParams
    params_c: DescriptorParams
    params_n: DescriptorParams
    params_o: DescriptorParams

def calculate_dens_coeffs(data_dir: Path, output_name: str, params: AtomDescriptorParams, alignment_method: str):
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Get the deformation density coefficients for all deformation density .cube files in a folder.')
    parser.add_argument('--output', type=str, default='output', help='Name of output file.')
    parser.add_argument('--path', type=str, default='data', help='Path containing cube files.')
    parser.add_argument('--align', type=str, default='elf', help='Alignment method')
    args = parser.parse_args()

    default_params = DescriptorParams(r_o=2.5, r_i=0.0, n_rad=4, n_l=3, gamma=0)
    default_atom_params = AtomDescriptorParams(params_h=default_params, params_c=default_params,
                                               params_n=default_params, params_o=default_params)

    generate_dens_coeffs(Path(args.path), args.output, default_atom_params, args.align)

