# gpr/model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, Matern

def standardize_gpr_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {
        "stock_concentration": "concentration",
        "stock_volume": "volume",
        "lambda_max_wavelength": "lambda max wavelength",
        "lambda_max_absorbance": "lambda max absorbance",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

def add_features(df):
    df = df.copy()
    df["rxn_concentration"] = df["concentration"].astype(float) * df["volume"].astype(float)
    return df

def build_features(
    df: pd.DataFrame,
    num_cols: list,
    cat_col: list,
    target_col: str,
):
    """
    Build X (scaled numeric + one-hot categorical) and y.
    Assumes feature columns already exist in df.
    """
    missing = [c for c in (num_cols + cat_col + [target_col]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for GPR: {missing}")

    encoder = OneHotEncoder(sparse_output=False)
    X_cat = encoder.fit_transform(df[cat_col])

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(df[num_cols])

    X = np.hstack([X_num_scaled, X_cat])
    y = df[target_col].to_numpy(dtype=float)

    return X, y, scaler, encoder

def build_kernel(n_features: int, nu: float = 1.5):
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(n_features),
            length_scale_bounds=(1e-2, 1e2),
            nu=nu
        )
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    )
    return kernel

def fit_gpr(
    X,
    y,
    test_size=0.2,
    random_state=0,
    n_restarts_optimizer=10,
    nu=1.5,
):
    """
    Fit GPR model and return model + split data + predictions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    n_features = X_train.shape[1]
    kernel = build_kernel(n_features=n_features, nu=nu)

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )

    gpr.fit(X_train, y_train)
    y_mean, y_std = gpr.predict(X_test, return_std=True)
    print("Learned kernel:", gpr.kernel_)
    return {
        "gpr": gpr,
        "kernel_learned": gpr.kernel_,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_test_mean": y_mean, "y_test_std": y_std,
    }
