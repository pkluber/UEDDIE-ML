# UEDDIE-ML
Unified GPU implementation of Dr. Kaycee Low's EDDIE-ML

## Requirements
- `get_deformation_densities.py` requires `numpy` and `pyscf`, and optionally but recommended `gpu4pyscf` 
- `get_dens_coeffs.py` requires `numpy`, `ase`, `scipy`, `spherical_functions`, `sympy`, and 'h5py'
- `make_dataset.py` requires `numpy`, `h5py`, `matplotlib`, and `scikit-learn`
- `get_energies.py` requires Psi4 and the `psi4` Python package
- `train.py` requires `numpy`, `h5py`, `scikit-learn`, `joblib`, `torch`, `matplotlib` 

You may need different virutal environments to run each script, because oftentimes Psi4 and PySCF conflict.
