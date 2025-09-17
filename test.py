from dataset import UEDDIEDataset
from model import UEDDIENetwork

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import torch 

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from joblib import load
from typing import Tuple

# Returns the predicted followed by the actual interaction energy (in kcal/mol)
def predict(model: UEDDIENetwork | None = None, test_dataset: UEDDIEDataset | None = None,
            scaler_y: RobustScaler | None = None) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model is None:
        model = torch.load('model.pt', weights_only=False, map_location=device)
    model.disable_multi_gpu(device)
    model.eval()

    if test_dataset is None:
        test_dataset = load('dataset_test.joblib')

    if scaler_y is None: 
        scaler_y = load('scaler_y_train.joblib')

    ies = []
    ies_pred = []
    with torch.no_grad():
        for x, e, c, y, name in [test_dataset.get(x, return_name=True) for x in range(len(test_dataset))]:
            print(f'Testing {name}...')
            x, e, c = x.to(device), e.to(device), c.to(device)
            x = x.unsqueeze(0)
            e = e.unsqueeze(0)
            c = c.unsqueeze(0)

            y_pred = model(x, e, c).cpu()
            y_pred = np.array([y_pred.item()])
            y_pred = y_pred.reshape(1, 1)
            ie_pred = scaler_y.inverse_transform(y_pred)[0, 0] * 627.509  # kcal/mol

            y = y.cpu().item()
            y = np.array([y]).reshape(1, 1)
            ie = scaler_y.inverse_transform(y)[0, 0] * 627.509

            print(f'Predicted IE (kcal/mol): {ie_pred:.1f}')
            print(f'Actual IE    (kcal/mol): {ie:.1f}')

            ies.append(ie)
            ies_pred.append(ie_pred)

    ies = np.array(ies)
    ies_pred = np.array(ies_pred)

    return ies_pred, ies

def print_test_metrics(ies_pred: np.ndarray, ies: np.ndarray):
    mse = mean_squared_error(ies, ies_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ies, ies_pred)
    r2 = r2_score(ies, ies_pred)

    print(f'MSE:  {mse:.1f}')
    print(f'RMSE: {rmse:.1f}')
    print(f'MAE:  {mae:.1f}')
    print(f'R^2:  {r2:.3f}')

def plot_results(ies_pred: np.ndarray, ies: np.ndarray):
    # Best fit line
    reg = LinearRegression().fit(ies_pred.reshape(-1, 1), ies)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    # Visualize test results
    plt.scatter(ies_pred, ies, alpha=0.8)
    line = np.linspace(min(ies_pred), max(ies_pred), 100)
    plt.plot(line, line, 'r--', label='Ideal (y=x)')
    plt.plot(line, slope*line + intercept, 'g-', label='Best-fit line')
    plt.xlabel('Predicted $\\Delta E^{\\text{INT}}$ (kcal/mol)')
    plt.ylabel('Actual $\\Delta E^{\\text{INT}}$ (kcal/mol)')
    #plt.title('Predicted vs. Actual Interaction Energy')
    plt.legend()

    r2 = r2_score(ies, ies_pred)
    plt.text(
        0.05, 0.95, f'$R^2 = {r2:.3f}$',
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='square,pad=0.3')
    )

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=300)
    plt.close()

def plot_errors(ies_pred: np.ndarray, ies: np.ndarray):
    errors = ies_pred - ies
    
    # Bins should be every 0.2 kcal/mol 
    bin_size = 0.2 
    bin_edges = np.arange(np.floor(np.min(errors) / bin_size) * bin_size, np.ceil(np.max(errors) / bin_size) * bin_size + bin_size, bin_size)

    plt.hist(errors, bins=bin_edges, edgecolor='black', density=False)
    plt.xlabel('Residual errors (kcal/mol)')
    plt.xlim(np.floor(np.min(errors)), np.ceil(np.max(errors)) + bin_size)
    plt.xticks(np.arange(np.floor(np.min(errors)), np.ceil(np.max(errors))+1, 1))
    plt.grid(which='major', linewidth=1.0, alpha=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', linewidth=0.5, linestyle=':',  alpha=0.5)
    plt.ylabel('Frequency')
    plt.title('Histogram of Test Residual Errors')
    plt.savefig('test_errors.png', dpi=300)
    plt.close()

def plot_relative_errors(ies_pred: np.ndarray, ies: np.ndarray): 
    percent_errors = np.abs((ies_pred - ies) / ies) * 100
    
    # Bins should be every 100%
    bin_size = 100
    bin_edges = np.arange(np.floor(np.min(percent_errors) / bin_size) * bin_size, np.ceil(np.max(percent_errors) / bin_size) * bin_size + bin_size, bin_size)

    plt.hist(percent_errors, bins=bin_edges, edgecolor='black', density=False)

    plt.xlabel('Percent Errors')
    plt.xlim(np.floor(np.min(percent_errors)), np.ceil(np.max(percent_errors)) + bin_size)

    x_tick_length = 1000
    plt.xticks(np.arange(np.floor(np.min(percent_errors)/x_tick_length)*x_tick_length, np.ceil(np.max(percent_errors)/x_tick_length)*x_tick_length + x_tick_length, x_tick_length))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0))

    plt.grid(which='major', linewidth=1.0, alpha=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', linewidth=0.5, linestyle=':',  alpha=0.5)
    plt.ylabel('Frequency')
    plt.title('Histogram of Relative Errors')
    plt.savefig('test_relative_errors.png', dpi=300)
    plt.close()

def test(model: UEDDIENetwork | None = None, test_dataset: UEDDIEDataset | None = None,
         scaler_y: RobustScaler | None = None) -> Tuple[np.ndarray, np.ndarray]:
    ies_pred, ies = predict(model=model, test_dataset=test_dataset, scaler_y=scaler_y)

    # Print test metrics
    print_test_metrics(ies_pred, ies)

    # Visualize test results
    plot_results(ies_pred, ies)

    # Visualize absolute errors
    plot_errors(ies_pred, ies)

    # Visualize relative errors
    plot_relative_errors(ies_pred, ies)

if __name__ == '__main__':
    test()
