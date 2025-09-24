import numpy as np
import torch

import argparse

parser = argparse.ArgumentParser(description='Generate dissociation curve for a given system.')
parser.add_argument('--input', type=str, default='C_c3mim_A_dca_P2-d2_83', help='Input system name')
parser.add_argument('--natoms', type=int, default=27, help='Input system name')
args = parser.parse_args()

# Evaluate model
from dataset import UEDDIEDataset
from model import UEDDIENetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('model.pt', weights_only=False, map_location=device)
model.disable_multi_gpu(device)
model.eval()

dataset = UEDDIEDataset()
_, scaler_y = dataset.load_and_apply_scalers()

with torch.no_grad():
    for x, e, c, y, name in [dataset.get(x, return_name=True) for x in range(len(dataset))]:
        if name == args.input: 
            x = x.unsqueeze(0)
            e = e.unsqueeze(0)
            c = c.unsqueeze(0)

            _ = model(x, e, c)

            per_atom_IE = model.per_atom_IE
            per_atom_e = e

# Convert to NumPy and kcal/mol
per_atom_IE = per_atom_IE.cpu().numpy()
per_atom_IE = per_atom_IE.reshape(per_atom_IE.shape[1], 1)
per_atom_IE = scaler_y.inverse_transform(per_atom_IE) * 627.509

scaler_baseline = (scaler_y.inverse_transform(np.array([[0]])) * 627.509)[0][0]

per_atom_IE = per_atom_IE[:, 0] - scaler_baseline
per_atom_IE = per_atom_IE[:args.natoms]
per_atom_IE += scaler_baseline / args.natoms 

element_map = {0: 'H', 1: 'C', 2: 'N', 3: 'O'}
per_atom_e = per_atom_e.squeeze(0).cpu().numpy()

for x in range(args.natoms):
    elem = element_map[per_atom_e[x]]
    atom_contrib = per_atom_IE[x]

    print(f'{elem} {atom_contrib:.1f}')

np.savetxt('per_atom_ie.npy', per_atom_IE, fmt='%.1f')
