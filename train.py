import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import numpy as np
import dataset
from model import UEDDIENetwork
from pathlib import Path
from joblib import dump

# Needed for 64-bit precision
torch.set_default_dtype(torch.float64)

# Setup devices
devices = []
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    devices.append(torch.device('cpu'))
else:
    for x in range(num_gpus):
        print(f'Found GPU {x}: {torch.cuda.get_device_name(x)}')
        devices.append(torch.device(f'cuda:{x}'))

device = devices[0]

# Load datasets
train_dataset, validation_dataset, test_dataset = dataset.get_train_validation_test_datasets()
total_dataset = len(train_dataset) + len(validation_dataset) + len(test_dataset)
train_split_percent = int(100 * len(train_dataset) / total_dataset)
validation_split_percent = int(100 * len(validation_dataset) / total_dataset)
print(f'Using {train_split_percent}/{validation_split_percent}/{100-train_split_percent-validation_split_percent} train/validation/test split')

# Scale data
scaler_x, scaler_y = train_dataset.scale_and_save_scalers()
validation_dataset.apply_scalers(scaler_x, scaler_y)
test_dataset.apply_scalers(scaler_x, scaler_y)

# Save scaled datasets for later testing
dump(train_dataset, 'dataset_train.joblib')
dump(validation_dataset, 'dataset_validation.joblib')
dump(test_dataset, 'dataset_test.joblib')

# Initialize dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

# Initialize model
x_sample, _, _, _ = next(iter(train_dataloader))
d_model = x_sample.shape[-1]
model = UEDDIENetwork(d_model, num_heads=4, d_ff=128, depth_e=5, depth_c=5)
if len(devices) == 1:
    model.to(device) 

# Loss and stuff
loss_function = nn.MSELoss()
optimizer = optim.AdamW(list(model.parameters()), lr=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=60)

print(f'Beginning training using primarily device={device}!', flush=True)

# Early stopping
early_stopping_patience = 100
best_val_loss = float('inf')
epochs_no_improve = 0

train_losses = []
val_losses = []

n_epoch = 2000
for epoch in range(n_epoch): 
    # Training...
    model.train()

    train_loss = 0
    for X, E, C, Y in train_dataloader:
        X, E, C, Y = X.to(device, non_blocking=True), E.to(device, non_blocking=True), C.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        Y_pred = model(X, E, C)

        loss = loss_function(Y_pred, Y) 
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)
    
    # Validation...
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, E, C, Y in validation_dataloader:
            X, E, C, Y = X.to(device, non_blocking=True), E.to(device, non_blocking=True), C.to(device, non_blocking=True), Y.to(device, non_blocking=True)

            Y_pred = model(X, E, C)

            loss = loss_function(Y_pred, Y) 
            val_loss += loss.item()

    val_loss /= len(validation_dataloader)
    val_losses.append(val_loss)
    
    # Step plateau scheduler
    scheduler.step(val_loss)

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model, 'model.pt')
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch}')
        break
    
    # Output 
    if epoch % 5 == 0:
        print(f'Epoch {epoch}, train loss: {train_loss}, val loss: {val_loss}, LR: {optimizer.param_groups[0]["lr"]}', flush=True)
        np.save('losses_train.npy', np.array(train_losses))
        np.save('losses_validation.npy', np.array(val_losses))

test_loss = 0
with torch.no_grad():
    for X, E, C, Y in test_dataloader:
        X, E, C, Y = X.to(device), E.to(device), C.to(device), Y.to(device)
        Y_pred = model(X, E, C)
        loss = loss_function(Y_pred, Y)
        test_loss += loss.item()

print(f'Test average loss: {test_loss / len(test_dataset)}')

import test

test.test(model=model, test_dataset=test_dataset, scaler_y=scaler_y)
