import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.preprocessing import StandardScaler

# ====== CONFIGURATION ======
N_SAMPLES = 50  # Number of LHS samples to generate

# Data file here
EXPERIMENTAL_DATA_PATH = 'control_only.csv'  # Change this to your actual filename

param_ranges = {
    'concentration': [0.005, 0.02],  # Adjust based on your experimental range
    'volume': [10, 20],               # µL
    
}

#Column names in file
FEATURE_COLS = ['concentration', 'volume']
TARGET_COL = 'absorbance'

# ====== LOAD AND CLEAN DATA ======
def load_experimental_data():
    
    """Load experimental data from CSV"""
    print(f"Loading from: {EXPERIMENTAL_DATA_PATH}")
    df = pd.read_csv(EXPERIMENTAL_DATA_PATH)
    print(f"Loaded {len(df)} rows")
    print(f"\n Columns in your CSV:")
    
    
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    return df

def prepare_training_data(df):
    """Prepare training data from experimental dataset"""
    
    # Check which features are available
    available_features = []
    missing_features = []
    
    for col in FEATURE_COLS:
        if col in df.columns:
            available_features.append(col)
        else:
            missing_features.append(col)
    
    if missing_features:
        
        print(f"\n  Warning: Missing features in CSV: {missing_features}")
        print(f"   Available columns: {list(df.columns)}")
        print(f"\n   Available numeric columns you could use:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            print(f"     - {col}")
        raise ValueError("Missing required columns - please update FEATURE_COLS")
    
    # Check target column
    if TARGET_COL not in df.columns:
        print(f"\n Error: Target column '{TARGET_COL}' not found")
        print(f"   Available columns: {list(df.columns)}")
        raise ValueError(f"Target column '{TARGET_COL}' not found")
    
    # Remove rows with missing values
    df_clean = df[available_features + [TARGET_COL]].dropna()
    
    if len(df_clean) == 0:
        print(f"\n  Error: No valid data after removing missing values")
        raise ValueError("No valid training data")
    
    X = df_clean[available_features].values
    y = df_clean[TARGET_COL].values
    
    print(f"\n Prepared {len(X)} training samples with {X.shape[1]} features")
    print(f"  Features: {available_features}")
    print(f"  Target: {TARGET_COL}")
    print(f"  Target range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Show feature ranges from your data
    print(f"\n  Feature ranges in your data:")
    for i, feat in enumerate(available_features):
        print(f"    {feat}: [{X[:, i].min():.4f}, {X[:, i].max():.4f}]")
    
    return X, y, available_features

# ====== TRAIN GAUSSIAN PROCESS ======
def train_gaussian_process(X_train, y_train):
    """Train a Gaussian Process model"""
    print("\n" + "=" * 60)
    print("Training Gaussian Process Model")
    print("=" * 60)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define kernel
    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(X_train.shape[1]),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5
    )
    
    # Create GP model
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=0.01,
        normalize_y=True,
        random_state=42
    )
    
    print("Fitting GP model...")
    gp.fit(X_train_scaled, y_train)
    
    print(f" GP trained successfully!")
    print(f"  Optimized kernel: {gp.kernel_}")
    
    # Model evaluation on training data
    y_pred_train = gp.predict(X_train_scaled)
    r2 = 1 - np.sum((y_train - y_pred_train)**2) / np.sum((y_train - y_train.mean())**2)
    rmse = np.sqrt(np.mean((y_train - y_pred_train)**2))
    print(f"  Training R²: {r2:.3f}")
    print(f"  Training RMSE: {rmse:.4f}")
    
    return gp, scaler

# ====== GENERATE LHS SAMPLES ======
def generate_lhs_samples(n_samples, param_ranges):
    """Generate Latin Hypercube Samples"""
    n_dims = len(param_ranges)
    
    sampler = qmc.LatinHypercube(d=n_dims, seed=42)
    sample = sampler.random(n=n_samples)
    
    l_bounds = [v[0] for v in param_ranges.values()]
    u_bounds = [v[1] for v in param_ranges.values()]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    
    df = pd.DataFrame(sample_scaled, columns=param_ranges.keys())
    df.insert(0, 'sample_id', range(1, n_samples + 1))
    
    return df

#main thing to run
if __name__ == "__main__":
    print("=" * 60)
    print("Latin Hypercube Sampling - Control Experiments")
    print("=" * 60)
    
    # Step 1: Load experimental data
    try:
        exp_data = load_experimental_data()
        X_train, y_train, feature_names = prepare_training_data(exp_data)
    except Exception as e:

        print("\n" + "=" * 60)
        print("TROUBLESHOOTING:")
        print("=" * 60)
        print("1. Make sure your CSV filename is correct")
        print("2. Update FEATURE_COLS to match your actual column names")
        print("3. Update TARGET_COL to your output column name")
        print("4. Adjust param_ranges to match your experimental ranges")
        print("=" * 60)
        raise
# Step 2: Train Gaussian Process
    gp_model, scaler = train_gaussian_process(X_train, y_train)
    
    # Step 3: Generate LHS samples
    print("\n" + "=" * 60)
    print(f"Generating {N_SAMPLES} Latin Hypercube Samples")
    print("=" * 60)
    lhs_samples = generate_lhs_samples(N_SAMPLES, param_ranges)
    print(f" Generated {len(lhs_samples)} samples")
    print(f"\n Parameter ranges for LHS:")
    for param, (low, high) in param_ranges.items():
        print(f"  {param}: [{low}, {high}]")
    
    # Step 4: Predict responses using GP
    print("\n" + "=" * 60)
    print("Predicting Responses using Trained GP")
    print("=" * 60)
    
    X_lhs = lhs_samples[list(param_ranges.keys())].values
    X_lhs_scaled = scaler.transform(X_lhs)
    
    y_pred, sigma = gp_model.predict(X_lhs_scaled, return_std=True)
    
    lhs_samples['predicted_absorbance'] = y_pred
    lhs_samples['prediction_std'] = sigma
    lhs_samples['prediction_lower_95'] = y_pred - 1.96 * sigma
    lhs_samples['prediction_upper_95'] = y_pred + 1.96 * sigma
    
    print(f" Predictions complete!")
    print(f"  Mean prediction: {y_pred.mean():.4f}")
    print(f"  Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"  Mean uncertainty: {sigma.mean():.4f}")
    
    # Step 5: Save results
    output_file = f'lhs_control_predictions_{N_SAMPLES}samples.csv'
    lhs_samples.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")
    
    # Step 6: Display summary
    print("\n" + "=" * 60)
    print("First 15 Predicted Samples:")
    print("=" * 60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(lhs_samples.head(15).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Statistical Summary:")
    print("=" * 60)
    print(lhs_samples.describe().to_string())
    
    # Step 7: Visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # Plot 1: Parameter distributions
    n_params = len(param_ranges)
    fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 5))
    if n_params == 1:
        axes = [axes]
    
    for idx, (param, (low, high)) in enumerate(param_ranges.items()):
        axes[idx].hist(lhs_samples[param], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].set_xlabel(param.replace('_', ' ').title(), fontsize=12)
        axes[idx].set_ylabel('Frequency', fontsize=12)
        axes[idx].set_title(f'{param}\nRange: [{low}, {high}]', fontsize=13)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lhs_control_distributions.png', dpi=150)
    print(" Saved: lhs_control_distributions.png")
    plt.show()
    
    # Plot 2: LHS space coverage (2D scatter if 2 params)
    if n_params == 2:
        fig, ax = plt.subplots(figsize=(8, 8))
        param_names = list(param_ranges.keys())
        scatter = ax.scatter(lhs_samples[param_names[0]], 
                           lhs_samples[param_names[1]], 
                           c=y_pred, cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        ax.set_xlabel(param_names[0].replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(param_names[1].replace('_', ' ').title(), fontsize=12)
        ax.set_title('LHS Sample Distribution\n(colored by predicted absorbance)', fontsize=13)
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted Absorbance', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('lhs_control_2d_coverage.png', dpi=150)
        print(" Saved: lhs_control_2d_coverage.png")
        plt.show()
    
    # Plot 3: Predictions with uncertainty
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(y_pred, bins=20, edgecolor='black', alpha=0.7, color='green')
    ax1.axvline(y_train.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Training mean: {y_train.mean():.3f}')
    ax1.set_xlabel('Predicted Absorbance', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of GP Predictions', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    sorted_idx = np.argsort(y_pred)
    ax2.plot(y_pred[sorted_idx], 'b-', label='Prediction', linewidth=2)
    ax2.fill_between(range(len(y_pred)), 
                      lhs_samples['prediction_lower_95'].values[sorted_idx],
                      lhs_samples['prediction_upper_95'].values[sorted_idx],
                      alpha=0.3, color='blue', label='95% CI')
    ax2.set_xlabel('Sample (sorted by prediction)', fontsize=12)
    ax2.set_ylabel('Predicted Absorbance', fontsize=12)
    ax2.set_title('Predictions with Uncertainty', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lhs_control_predictions.png', dpi=150)
    
    print("Saved look in your folder: lhs_control_predictions.png")
    #remeber to change this or all runs look the same
    plt.show()
    
    print("\n" + "=" * 60)

    print("=" * 60)
    print(f"\n You now have {N_SAMPLES} LHS samples with GP-predicted absorbances!")
    print(f"Original training data: {len(X_train)} samples")
    print(f"Augmented with LHS: {N_SAMPLES} new samples")
    print(f"\nTotal available for retraining: {len(X_train) + N_SAMPLES} samples")
    print("=" * 60)