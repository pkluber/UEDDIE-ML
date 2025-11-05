from get_deformation_densities import dimer_cube_difference, CubeParams
from get_dens_coeffs import calculate_dens_coeffs, CoeffParams
from make_dataset import make_dataset

from density.coeff import CoeffWrapper

import numpy as np
import torch
from joblib import load
from pathlib import Path
from typing import Tuple

def evaluate(xyz_path: Path, charges: Tuple[int, int] = (0, 0), uses_pca: bool = True):
    # Generate .cube file
    cube_params = load('cube_params.joblib')
    dimer_cube_difference(xyz_path, cube_params, overwrite=False, charges=charges)
    
    # Generate .coeff file
    dens_path = xyz_path.parent / f'{xyz_path.stem}.cube'
    coeff_params = load('coeff_params.joblib')
    calculate_dens_coeffs(dens_path, coeff_params, overwrite=True, charges=charges)

    # Load .coeff file
    coeffs_path = xyz_path.parent / f'{xyz_path.stem}.coeff'
    coeffs = CoeffWrapper(coeffs_path)
    coeffs.load()
 
    # Create X, E, C needed to evaluate model
    species_conversion = {'H': 0, 'C': 1, 'N': 2, 'O': 3}

    X_list = []
    E_list = []
    C_list = []
    for elf in coeffs.get_elfs():
        X_list.append(elf.value)
        E_list.append(species_conversion[elf.species])
        C_list.append(elf.charge)
    
    X, E, C = np.array(X_list), np.array(E_list), np.array(C_list)

    # Load PCA object if applicable
    if uses_pca:
        pca = load('pca.joblib')

        # Apply PCA to X 
        X = pca.transform(X)

        # Get the 99% cutoff we used previously
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
        
        # Apply to X
        X = X[:, :n_components_99]
    
    X = torch.from_numpy(X).unsqueeze(0)
    E = torch.from_numpy(E).unsqueeze(0)
    C = torch.from_numpy(C).unsqueeze(0)
    
    # Need to load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load('model.pt', weights_only=False, map_location=device) 
    model.eval()

    # Also load the X and Y scalers
    scaler_x = load('scaler_train.joblib')
    scaler_y = load('scaler_y_train.joblib')

    # Apply X scaling
    X_shape = X.shape
    X = scaler_x.fit_transform(X.reshape(-1, X_shape[-1]))
    X = X.reshape(*X_shape)
    X = torch.from_numpy(X)

    with torch.no_grad():
        X, E, C = X.to(device), E.to(device), C.to(device)
        y_pred = model(X, E, C)
        y_pred = np.array([y_pred.item()])
        y_pred = y_pred.reshape(1, 1)
        ie_model = scaler_y.inverse_transform(y_pred)[0, 0] * 627.509

        per_atom_IE = model.per_atom_IE 
        per_atom_e = E
    
    print(f'Predicted interaction energy (kcal/mol): {ie_model}') 

    # Code from visualize_energy.py

    # Convert to NumPy and kcal/mol
    per_atom_IE = per_atom_IE.cpu().numpy()
    per_atom_IE = per_atom_IE.reshape(per_atom_IE.shape[1], 1)
    per_atom_IE = scaler_y.inverse_transform(per_atom_IE) * 627.509

    scaler_baseline = (scaler_y.inverse_transform(np.array([[0]])) * 627.509)[0][0]

    with open(xyz_path) as fd:
        lines = fd.readlines()
        natoms = int(lines[0])
        natoms += int(lines[natoms+2])

    per_atom_IE = per_atom_IE[:, 0] - scaler_baseline
    per_atom_IE = per_atom_IE[:natoms]
    per_atom_IE += scaler_baseline / natoms 

    element_map = {0: 'H', 1: 'C', 2: 'N', 3: 'O'}
    per_atom_e = per_atom_e.squeeze(0).cpu().numpy()
    
    for x in range(natoms):
        elem = element_map[per_atom_e[x]]
        atom_contrib = per_atom_IE[x]

        print(f'{elem} {atom_contrib:.1f}')

    np.savetxt('eval_per_atom_ie.npy', per_atom_IE, fmt='%.1f')

    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate the model for an out-of-distribution system.')
    parser.add_argument('--input', type=str, help='Input file to use for evaluating the model')
    parser.add_argument('--charges', nargs=2, type=int, default=(0, 0), help='Charges of the monomers')
    parser.add_argument('--pca', type=bool, default=True, help='Whether PCA needs to be applied')

    args = parser.parse_args()

    evaluate(Path(args.input), charges=args.charges, uses_pca=args.pca)


