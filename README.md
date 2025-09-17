# UEDDIE-ML
Unified EDDIE-ML, or UEDDIE-ML (pronounced "oodie-ML"), is a transformer model that predicts interaction energies from atomistic deformation density coefficients for both neutral and charged systems, based on Dr. Kaycee Low's original EDDIE-ML. The main innovation is the introduction of a charge scaling factor to scale atomistic contributions in the atomistic approximation based on monomer charge. This produces test metrics less than 1 kcal/mol, thereby reaching chemical accuracy statistically. 

![UEDDIE-ML results scatterplot showing the actual vs. the predicted interaction energy in kcal/mol on the test set. The best-fit line is nearly linear with an R^2 value of 0.999.](test_results.png "UEDDIE-ML test results")

## Quickstart
In the repository directory, simply run `pip install -r requirements.txt` then run `python test.py`

To load the pretrained model for inference,
```python
# Because the model was trained using GPUs, we need to set map_location to cpu
model = torch.load('model.pt', weights_only=False, map_location='cpu')

# Make sure multi-GPU model parallelism is disabled 
model.disable_multi_gpu('cpu')

# Flag the model for inference mode 
model.eval()

# Load the dataset and scalers
dataset = UEDDIEDataset()
scaler_x, scaler_y = dataset.load_and_apply_scalers()

# Load a sample 
x, e, c, y, name = dataset.get(0, return_name=True)

# Reshape to (1, ...) for batch dimension of 1 
x, e, c = x.unsqueeze(0), e.unsqueeze(0), c.unsqueeze(0)

# Evaluate the model
with torch.no_grad():
    y_pred = model(x, e, c)

# Invert y scaling and convert from Hartrees to kcal/mol 
y_pred = np.array([y_pred.item()])
y_pred = y_pred.reshape(1, 1)
ie_model = scaler_y.inverse_transform(y_pred)[0, 0] * 627.509

# Print the final interaction energy
print(ie_model)
```

## Overview
This repository implements the following scientific computing and machine learning pipeline:
1. Deformation densities are calculated using (GPU4)PySCF and stored as .cube files
2. Those .cube files, encoding uniform deformation density grids, are read and atomistic descriptors are calculated from them and stored as .coeff files
3. The dataset is created and dimensionality reduction (PCA) is applied to form the features, creating `output.hdf5`
4. Interaction energies are calculated (in Hartrees) using the CPU-based open-source quantum chemistry engine Psi4, creating `energies.dat`
5. The atomistic machine learning model UEDDIE-ML is trained and tested using a roughly 80/20 split  by default of the dataset and interaction energies 

## Requirements
- `get_deformation_densities.py` requires `numpy` and `pyscf`, and optionally but recommended `gpu4pyscf` 
- `get_dens_coeffs.py` requires `numpy`, `ase`, `scipy`, `spherical_functions`, `sympy`, and `h5py`
- `make_dataset.py` requires `numpy`, `h5py`, `matplotlib`, and `scikit-learn`
- `get_energies.py` requires Psi4 and the `psi4` Python package
- `train.py` requires `numpy`, `h5py`, `scikit-learn`, `joblib`, `torch`, `matplotlib`, and 'seaborn' 

You may need different environments to run each script, because oftentimes Psi4, PySCF, and PyTorch conflict, especially in a HPC module environment.
