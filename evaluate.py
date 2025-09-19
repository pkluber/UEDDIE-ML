from get_deformation_densities import dimer_cube_difference
from get_dens_coeffs import calculate_dens_coeffs
from make_dataset import make_dataset

from density.coeff import CoeffWrapper

import numpy as np
from joblib import load
from pathlib import Path

def evaluate(xyz_path: Path, charges: Tuple[int, int] = (0, 0), uses_pca: bool = True):
    # Generate .rho file
    dimer_cube_difference(xyz_path, 'LDA' grid_type='becke', overwrite=True, charges=charges)
    
    # Generate .coeff file
    dens_path = xyz_path.parent / f'{xyz_path.stem}.rho'
    calculate_dens_coeffs(dens_path, charges=charges)

    # Load .coeff file
    coeffs_path = xyz_path.parent / f'{xyz_path.stem}.coeff'
    coeffs = CoeffWrapper(coeffs_path)
 
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
    model.disable_multi_gpu(device)
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
        y_pred = model(X, E, C)
        y_pred = np.array([y_pred.item()])
        y_pred = y_pred.reshape(1, 1)
        ie_model = scaler_y.inverse_transform(y_pred)[0, 0] * 627.509
    
    print('Predicted interaction energy (kcal/mol): {ie_model}') 
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate the model for an out-of-distribution system.')
    parser.add_argument('--input', type=str, help='Input file to use for evaluating the model')
    parser.add_argument('--charges', nargs=2, type=int, default=(0, 0), help='Charges of the monomers')
    parser.add_argument('--pca', type=bool, default=True, help='Whether PCA needs to be applied')

    args = parser.parse_args()

    evaluate(Path(args.input), charges=args.charges, uses_pca=args.pca)


