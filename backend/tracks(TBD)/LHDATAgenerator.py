# -*- coding: utf-8 -*-
"""

LATIN HYPERCUBE

@author: beatr
"""



import pandas as pd
#this was to fuggure out the collumes for the file

df = pd.read_csv('C:/Users/beatr/OneDrive/Desktop/senior stuff/latinHYper/20230130_allUDdata_forcrisis (1).csv')  # Update filename!


print("All columns in your CSV:")
print("=" * 60)
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n" + "=" * 60)
print(f"Total columns: {len(df.columns)}")



import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.preprocessing import StandardScaler

# how many samples to generate
N_SAMPLES = 300

# file

EXPERIMENTAL_DATA_PATH = 'C:/Users/beatr/OneDrive/Desktop/senior stuff/latinHYper/20230130_allUDdata_forcrisis (1).csv'

# these are the ranges i want to test - adjust based on what you see in the data
param_ranges = {
    'temperature_c': [60, 100],
    'time_h': [12, 24],
    'stir_rate': [300, 700],
    'ligand_loading_umol': [0.5, 2.5],
    'catalyst_loading_umol': [0.3, 1.0],
    'base_loading_umol': [20, 40]
}

# columns from the csv file
FEATURE_COLS = ['temperature_c', 'time_h', 'stir_rate', 'ligand_loading_umol', 
                'catalyst_loading_umol', 'base_loading_umol']
TARGET_COL = 'product_analyticalyield'

def load_experimental_data():
    print(f"Loading data from: {EXPERIMENTAL_DATA_PATH}")
    df = pd.read_csv(EXPERIMENTAL_DATA_PATH)
    print(f"Loaded {len(df)} rows")
    return df

def prepare_training_data(df):
    # check if we have all the columns we need
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    
    if missing:
        print(f"\nMissing columns: {missing}")
        print("Available columns:", list(df.columns))
        raise ValueError("Can't find required columns")
    
    if TARGET_COL not in df.columns:
        print(f"Can't find target column: {TARGET_COL}")
        raise ValueError("Target column missing")
    
    # drop any rows with missing data this was a issue earlier on thankfulluy got clener data sets now
    df_clean = df[FEATURE_COLS + [TARGET_COL]].dropna()
    
    X = df_clean[FEATURE_COLS].values
    y = df_clean[TARGET_COL].values
    
    print(f"\nUsing {len(X)} samples for training")
    print(f"Yield range: {y.min():.1f}% to {y.max():.1f}%")
    
    # show what ranges are actually in the data
    print("\nActual ranges in your data:")
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {col}: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")
    
    return X, y

def train_gp(X_train, y_train):
    print("\nTraining Gaussian Process model...")
    
    # standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # setup the GP kernel
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X_train.shape[1]),length_scale_bounds=(1e-2, 1e2), nu=2.5)
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,alpha=1e-6, normalize_y=True, random_state=42)
    
    gp.fit(X_scaled, y_train)
    
    # check how well it fit
    y_pred = gp.predict(X_scaled)
    r2 = 1 - np.sum((y_train - y_pred)**2) / np.sum((y_train - y_train.mean())**2)
    rmse = np.sqrt(np.mean((y_train - y_pred)**2))
    
    print(f"Training R2: {r2:.3f}")
    print(f"RMSE: {rmse:.2f}%")
    
    return gp, scaler

def generate_lhs(n_samples, param_ranges):
    # latin hypercube sampling
    n_dims = len(param_ranges)
    sampler = qmc.LatinHypercube(d=n_dims, seed=42)
    sample = sampler.random(n=n_samples)
    
    # scale to actual parameter ranges
    l_bounds = [v[0] for v in param_ranges.values()]
    u_bounds = [v[1] for v in param_ranges.values()]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    
    df = pd.DataFrame(sample_scaled, columns=param_ranges.keys())
    df.insert(0, 'sample_id', range(1, n_samples + 1))
    
    return df

# main code
if __name__ == "__main__":
    print("=" * 60)
    print("LHS sampling for reaction optimization")
    print("=" * 60)
    
    # load the data
    try:
        data = load_experimental_data()
        X_train, y_train = prepare_training_data(data)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("- CSV filename is correct")
        print("- Column names match FEATURE_COLS")
        raise
    
    # train the GP model
    gp_model, scaler = train_gp(X_train, y_train)
    
    #generate LHS samples
    print(f"\nGenerating {N_SAMPLES} LHS samples...")
    lhs_samples = generate_lhs(N_SAMPLES, param_ranges)
    print(f"Generated {len(lhs_samples)} samples")
    
    # use GP to predict yields fo reach sample
    print("\nPredicting yields...")
    X_lhs = lhs_samples[list(param_ranges.keys())].values
    X_lhs_scaled = scaler.transform(X_lhs)
    y_pred, sigma = gp_model.predict(X_lhs_scaled, return_std=True)
    
    # add predictions to the dataframe
    lhs_samples['predicted_yield'] = np.clip(y_pred, 0, 100)
    lhs_samples['uncertainty'] = sigma
    lhs_samples['lower_bound'] = np.clip(y_pred - 1.96*sigma, 0, 100)
    lhs_samples['upper_bound'] = np.clip(y_pred + 1.96*sigma, 0, 100)
    
    print(f"Mean predicted yield: {y_pred.mean():.1f}%")
    print(f"Best predicted yield: {y_pred.max():.1f}%")
    
    # find best conditions
    best_idx = y_pred.argmax()
    print(f"\nBest predicted conditions:")
    for param in param_ranges.keys():
        print(f"  {param}: {lhs_samples.iloc[best_idx][param]:.2f}")
    print(f"  Predicted yield: {y_pred[best_idx]:.1f}%")
    
    # save results
    
    output_file = f'lhs_predictions_{N_SAMPLES}samples.csv'
    lhs_samples.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    # show top 10
    print("\nTop 10 predictions:")
    
    print(lhs_samples.nlargest(10, 'predicted_yield')[['sample_id', 'temperature_c', 
          'time_h', 'predicted_yield']].to_string(index=False))
    
    # make some plots
    print("\nGenerating plots...")
    
    # plot 1: parameter distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (param, (low, high)) in enumerate(param_ranges.items()):
        axes[idx].hist(lhs_samples[param], bins=20, edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel(param.replace('_', ' '))
        axes[idx].set_ylabel('count')
        axes[idx].set_title(f'{param} [{low}, {high}]')
        axes[idx].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('parameter_distributions.png', dpi=150)
    plt.show()
    
    
    
    
    # plot 2: predicted yields
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(y_pred, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax1.axvline(y_train.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'training mean: {y_train.mean():.1f}%')
    ax1.set_xlabel('predicted yield (%)')
    ax1.set_ylabel('count')
    ax1.set_title('predicted yield distribution')
    ax1.legend()
    
    ax1.grid(alpha=0.3)
    
    sorted_idx = np.argsort(y_pred)
    ax2.plot(y_pred[sorted_idx], 'b-', label='prediction', linewidth=2)
    ax2.fill_between(range(len(y_pred)), 
                      lhs_samples['lower_bound'].values[sorted_idx],
                      lhs_samples['upper_bound'].values[sorted_idx],
                      alpha=0.3, color='blue', label='95% confidence')
    ax2.set_xlabel('sample (sorted)')
    ax2.set_ylabel('predicted yield (%)')
    ax2.set_title('predictions with uncertainty')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('yield_predictions.png', dpi=150)
    plt.show()
    
    
    
    # plot 3: which parameters matter most
    #still need to fix this potential issuse is not showing
    fig, ax = plt.subplots(figsize=(10, 6))
    correlations = []
    param_names = list(param_ranges.keys())
    
    
    for param in param_names:
        corr = np.corrcoef(lhs_samples[param], y_pred)[0, 1]
        correlations.append(corr)
    
    colors = ['red' if c < 0 else 'green' for c in correlations]
    ax.barh(range(len(param_names)), correlations, color=colors, alpha=0.7)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels([p.replace('_', ' ') for p in param_names])
    ax.set_xlabel('correlation with yield')
    ax.set_title('parameter importance')
    ax.axvline(x=0, color='black', linestyle='--')
    ax.grid(alpha=0.3, axis='x')
    
    for i, corr in enumerate(correlations):
        
        ax.text(corr + 0.02 if corr > 0 else corr - 0.02, i, f'{corr:.3f}', 
                va='center', ha='left' if corr > 0 else 'right')
    
    plt.tight_layout()
    
    plt.savefig('parameter_importance.png', dpi=150)
    plt.show()
    
    # plot 4: 2d heatmap of top parameters
    top_params_idx = np.argsort(np.abs(correlations))[-2:]
    top_params = [param_names[i] for i in top_params_idx]
    
    if len(top_params) >= 2:
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(lhs_samples[top_params[0]], lhs_samples[top_params[1]], 
                           c=y_pred, cmap='RdYlGn', s=100, alpha=0.7, 
                           edgecolors='black', vmin=0, vmax=100)
        ax.set_xlabel(top_params[0].replace('_', ' '))
        ax.set_ylabel(top_params[1].replace('_', ' '))
        ax.set_title(f'predicted yields')
        ax.grid(alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='yield (%)')
        
        # mark best point
        ax.scatter(lhs_samples.iloc[best_idx][top_params[0]], 
                  lhs_samples.iloc[best_idx][top_params[1]], 
                  s=300, marker='*', color='gold', edgecolors='black', 
                  linewidths=2, label='best prediction')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('yield_heatmap.png', dpi=150)
        plt.show()
    
    print("\n" + "=" * 60)
    print("Done!")
    
    print(f"Original data: {len(X_train)} reactions")
    print(f"Generated: {N_SAMPLES} new predictions")
    print(f"Files saved:")
    print(f"  - {output_file}")
    print(f"  - parameter_distributions.png")
    print(f"  - yield_predictions.png")
    print(f"  - parameter_importance.png")
    print(f"  - yield_heatmap.png")
    print("=" * 60)