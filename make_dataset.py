from density.coeff import CoeffWrapper
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict
from dataclasses import asdict

def reduce_dims(X: np.ndarray, plot: bool = True) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find number of components for 95% and 99% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
    
    # Extract first {n_components_99} features, extracting 99% of variance
    X_reduced = pca.transform(X_scaled)[:, :n_components_99]

    if plot:
        # Plot cumulative explained variance
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, linestyle='-')

        # Add horizontal lines at 95% and 99%
        plt.hlines(y=0.95, xmin=0, xmax=n_components_95, color='r', linestyle='--', label="95% Variance")
        plt.hlines(y=0.99, xmin=0, xmax=n_components_99, color='g', linestyle='--', label="99% Variance")

        # Add vertical lines where they intersect
        plt.vlines(x=n_components_95, ymin=0, ymax=cumulative_variance[n_components_95], color='r', linestyle='--')
        plt.vlines(x=n_components_99, ymin=0, ymax=cumulative_variance[n_components_99], color='g', linestyle='--')

        # Labels and title
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance vs. Number of Components')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig('cumvar.png')

    print(f'Reduced dataset dimensionality from {X.shape[1]} to {X_reduced.shape[1]}')
    return X_reduced

def make_dataset(data_dir: Path, output_path: Path, overwrite: bool = True, pca: bool = True) -> bool:
    if output_path.is_file() and not overwrite:
        return True

    elf_attribs = defaultdict(lambda: [])
    for path in data_dir.rglob('*.coeff'):
        if path.is_file() and path.suffix == '.coeff':
            coeff_file = CoeffWrapper(path)
            coeff_file.load()
            elfs = coeff_file.get_elfs()
            system = coeff_file.get_system_name()

            for elf in elfs:
                elf_dict = asdict(elf)
                
                for key, value in elf_dict.items():
                    if key != 'unitcell' and key != 'params':
                        elf_attribs[key].append(value)
                
                elf_attribs['system'].append(system)
    
    if pca:
        elf_attribs['value'] = reduce_dims(np.array(elf_attribs['value']))

    with h5py.File(output_path, 'w') as fd:
        for key, value in elf_attribs.items():
            fd[key] = value

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Make the dataset (i.e. output.hdf5) to be used in the models from the .coeff files')
    parser.add_argument('--path', type=str, default='data', help='Path containing .coeff files')
    parser.add_argument('--output', type=str, default='output', help='Name of the output file')
    parser.add_argument('--overwrite', type=bool, default=True, help='Whether to overwrite pre-existing .hdf5 files')
    parser.add_argument('--pca', type=bool, default=True, help='Whether to apply PCA to the descriptors')

    args = parser.parse_args()

    make_dataset(Path(args.path), Path(f'{args.output.strip()}.hdf5'), overwrite=args.overwrite)
