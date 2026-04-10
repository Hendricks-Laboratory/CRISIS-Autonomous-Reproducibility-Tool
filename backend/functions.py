# =============================================================================
# backend/functions.py
#
# Core GPR functions: replicate grouping, GP fitting, and plot generation.
# These are the low-level building blocks called by supporting_functions.py.
#
# Functions:
#   group()              — aggregate replicates into mean/std/count per condition
#   fit_gp()             — fit a GaussianProcessRegressor and return metrics
#   get_color_palette()  — return color palette and colormap for a given scheme
#   descale_axis()       — inverse transform a 1D array using a StandardScaler
#   plot_gp()            — 1D GP plot with scatter, predicted mean, and 95% CI
#   plot_gp_2d()         — 2D side-by-side GP contour plots (mean + uncertainty)
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def group(data, controlVars, outputVar, blockVar=None, outlierSD=None):
    """
    Aggregate replicate rows into per-group summary statistics.

    Groups rows by controlVars (and optionally blockVar), then computes the
    mean, std, and count of outputVar within each group. Groups with fewer
    than 2 replicates are dropped.

    Args:
        data (pd.DataFrame): Input dataframe with raw replicate rows.
        controlVars (list[str]): Columns that define a unique experimental condition.
        outputVar (str): Column containing the measurement to summarize.
        blockVar (str | None): Optional blocking variable. When provided, the
            return value is a dict keyed by block level rather than a DataFrame.
        outlierSD (float | None): If set, replicates more than this many standard
            deviations from the group mean are removed before summarizing.

    Returns:
        pd.DataFrame | dict[Any, pd.DataFrame]: Summary DataFrame (or dict of
        DataFrames keyed by block) with columns: controlVars, mean, std, count.
    """
    groups = []
    dropped_groups = []

    group_cols = controlVars if blockVar is None else [blockVar] + controlVars

    for keys, g in data.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        y = g[outputVar].astype(float)

        # outlier filter
        if outlierSD is not None and len(y) > 2:
            m = y.mean()
            s = y.std()

            if s > 0:
                mask = np.abs(y - m) <= outlierSD * s
                g = g[mask]
                y = g[outputVar].astype(float)

        count = len(y)

        # drop if less than 2 points
        if count < 2:
            dropped_groups.append(keys)
            continue

        row = {
            **dict(zip(group_cols, keys)),
            "mean": y.mean(),
            "std": y.std(),
            "count": count
        }
        groups.append(row)

    summary = pd.DataFrame(groups)

    if dropped_groups:
        print(f"Warning: dropping {len(dropped_groups)} groups with <2 replicates")

    if blockVar is None:
        return summary.reset_index(drop=True)

    return {
        block: df_block.reset_index(drop=True)
        for block, df_block in summary.groupby(blockVar)
    }

def fit_gp(df, controlVars, kernel=None, outputVar=None, logVars=None, noise="std"):
    """
    Fit a GaussianProcessRegressor to the provided data.

    Args:
        df (pd.DataFrame): Grouped or raw dataframe; must contain controlVars and outputVar.
        controlVars (list[str]): Feature columns used as GP inputs.
        kernel: sklearn kernel object. Defaults to ConstantKernel × ARD-RBF.
        outputVar (str | None): Target column. Falls back to "mean" if None.
        logVars (list[str]): Columns to log10-transform before fitting.
        noise (str): Alpha strategy — "std" uses per-group variance, "sem" uses
            standard error of the mean, "constant" uses a fixed scalar (5e-2).

    Returns:
        tuple[GaussianProcessRegressor, dict]: Fitted GP and a metrics dict
        with keys r2, mae, rmse evaluated on the training set.
    """
    logVars = logVars or []

    X = df[controlVars].copy().astype(float)
    for col in logVars:
        X[col] = np.log10(X[col])

    if outputVar == None:
        y = df["mean"].to_numpy(dtype=float)
    else:
        y = df[outputVar].to_numpy(dtype=float)

    if noise == "sem":
        alpha = (df["std"].to_numpy(dtype=float) / np.sqrt(df["count"].to_numpy(dtype=float))) ** 2
    elif noise == "std":
        alpha = df["std"].to_numpy(dtype=float) ** 2
    elif noise == "constant":
        alpha = 5e-2
    else:
        raise ValueError("noise must be 'std', 'sem', or 'constant'")

    if kernel == None:
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
            length_scale=np.ones(len(controlVars)),
            length_scale_bounds=(0.5, 5)
        )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=5
    )

    gp.fit(X.to_numpy(), y)

    y_pred = gp.predict(X.to_numpy())

    metrics = {
        "r2": float(r2_score(y, y_pred)),
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
    }

    return gp, metrics


def get_color_palette(color_scheme="default"):
    """
    Return (palette_1d, cmap_2d) for the requested color scheme.

    Args:
        color_scheme (str): "default", "colorblind", or "high_contrast".

    Returns:
        tuple: (list[str], matplotlib colormap)
    """
    if color_scheme == "colorblind":
        palette_1d = ["#E69F00", "#56B4E9", "#009E73", "#F0E442"]
        cmap_2d = plt.cm.cividis      # perceptually uniform + colorblind-safe
    elif color_scheme == "high_contrast":
        palette_1d = ["#000000", "#0072B2", "#D55E00", "#CC79A7"]
        cmap_2d = plt.cm.binary
    else:
        palette_1d = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        cmap_2d = plt.cm.viridis
    return palette_1d, cmap_2d


def descale_axis(grid_1d, idx, scaler, n_features):
    """
    Inverse transform a 1D array for display using a StandardScaler.

    Tries the dummy-matrix approach first (for scalers fit on the full X matrix).
    Falls back to simple reshape (for per-column scalers fit on shape (n,1)).

    Args:
        grid_1d (np.ndarray): 1D array of scaled values to inverse transform.
        idx (int): Column index of this variable in the full feature matrix.
        scaler: StandardScaler instance, or None (returns grid_1d unchanged).
        n_features (int): Total number of features in the full feature matrix.

    Returns:
        np.ndarray: Inverse-transformed values in original units.
    """
    if scaler is None:
        return grid_1d
    try:
        dummy = np.zeros((len(grid_1d), n_features))
        dummy[:, idx] = grid_1d
        return scaler.inverse_transform(dummy)[:, idx]
    except ValueError:
        return scaler.inverse_transform(grid_1d.reshape(-1, 1)).ravel()


def plot_gp(df, gp, controlVars, xVar, yVar="mean", logVars=None, fixedVals=None, title=None,
            color_scheme="default", x_scaler=None, x_scaler_col=None, y_scaler=None, ylabel=None):
    """
    Generate a 1D GP plot: scatter of training data + predicted mean ± 95% CI.

    All data is plotted in scaled space. x_scaler and y_scaler inverse transform
    the axis values for display only — the GP and scatter positions are unchanged.

    Args:
        df (pd.DataFrame): Training dataframe (grouped summary in std mode, raw rows in mean mode).
        gp: Fitted GaussianProcessRegressor.
        controlVars (list[str]): GP input columns (scaled).
        xVar (str): Column to vary along the x-axis.
        yVar (str): Column to plot as the observed target (default "mean").
        logVars (list[str]): Columns displayed on a log scale.
        fixedVals (dict): Values to hold constant for non-xVar control vars.
            When empty, other vars are held at their median.
        title (str | None): Plot title.
        color_scheme (str): "default", "colorblind", or "high_contrast".
        x_scaler: StandardScaler for xVar. When provided, inverse transforms the
            x-axis grid and scatter x positions for display in original units.
        x_scaler_col (str): Name of the column x_scaler applies to. Must match
            xVar for the scaler to be applied.
        y_scaler: StandardScaler for the GP output. When provided, inverse
            transforms y_pred and y_std for display. Used in mean mode and in
            std mode when gp_target is replicate mean. None for std target.
        ylabel (str | None): Override label for the y-axis. Used when gp_target
            is "mean" in std mode to display the measurement column name instead
            of "mean".
    """
    logVars = logVars or []
    fixedVals = fixedVals or {}

    palette, _ = get_color_palette(color_scheme)
    color_line = palette[0]
    color_scatter = palette[1]

    x_raw_scaled = np.sort(df[xVar].astype(float).unique())
    x_grid_scaled = np.geomspace(x_raw_scaled.min(), x_raw_scaled.max(), 300) if xVar in logVars else np.linspace(x_raw_scaled.min(), x_raw_scaled.max(), 300)

    if x_scaler is not None and x_scaler_col == xVar:
        x_grid_display = x_scaler.inverse_transform(x_grid_scaled.reshape(-1, 1)).ravel()
    else:
        x_grid_display = x_grid_scaled

    X_pred_rows = []
    for x in x_grid_scaled:
        row = {}
        for col in controlVars:
            if col == xVar:
                row[col] = x
            elif col in fixedVals:
                row[col] = fixedVals[col]
            else:
                row[col] = df[col].median()
        X_pred_rows.append(row)

    X_pred = np.array([[row[col] for col in controlVars] for row in X_pred_rows], dtype=float)

    for i, col in enumerate(controlVars):
        if col in logVars:
            X_pred[:, i] = np.log10(X_pred[:, i])

    y_pred, y_std = gp.predict(X_pred, return_std=True)

    if y_scaler is not None:
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_std = y_std * y_scaler.scale_[0]

    plot_df = df.copy()
    for col, val in fixedVals.items():
        plot_df = plot_df[plot_df[col] == val]

    if x_scaler is not None and x_scaler_col == xVar:
        scatter_x = x_scaler.inverse_transform(plot_df[[xVar]]).ravel()
    else:
        scatter_x = plot_df[xVar].values

    if y_scaler is not None:
        scatter_y = y_scaler.inverse_transform(plot_df[[yVar]]).ravel()
    else:
        scatter_y = plot_df[yVar].values

    plt.figure(figsize=(8, 4))
    plt.scatter(scatter_x, scatter_y, color=color_scatter, label='Training Data')
    plt.plot(x_grid_display, y_pred, color=color_line, label='Predicted Mean')
    plt.fill_between(
        x_grid_display,
        y_pred - 1.96 * y_std,
        y_pred + 1.96 * y_std,
        color=color_line,
        alpha=0.35,
        label='95% Confidence Interval'
    )

    if xVar in logVars:
        plt.xscale('log')

    plt.xlabel(xVar)
    plt.ylabel(ylabel if ylabel is not None else yVar)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_gp_2d(df, gp, controlVars, xVar, yVar, zLabel="mean", zLabel_display=None,
               logVars=None, x_scaler=None, y_scaler=None, z_scaler_mean=None,
               z_scaler_unc=None, z_is_log=False, n_grid=100, color_scheme="default"):
    """
    Generate 2D GP contour plots: predicted mean and uncertainty side-by-side.

    Returns a single figure with two subplots (mean left, uncertainty right).
    Axes are inverse transformed for display using x_scaler and y_scaler.
    The predicted mean panel (left) is always unscaled via z_scaler_mean.
    The uncertainty panel (right) is unscaled only when z_scaler_unc is provided.

    Args:
        df (pd.DataFrame): Training dataframe.
        gp: Fitted GaussianProcessRegressor.
        controlVars (list[str]): GP input columns (scaled).
        xVar (str): Column mapped to the x-axis.
        yVar (str): Column mapped to the y-axis.
        zLabel (str): Column name in df used for scatter dot color on the mean
            plot (default "mean"). In std mode this is "mean" or "std";
            in mean mode this is the output column name.
        zLabel_display (str | None): Text label shown on the mean panel colorbar.
            When None, falls back to zLabel. Used in std mode to display the
            measurement column name instead of "mean"/"std".
        logVars (list[str]): Columns displayed on a log scale.
        x_scaler: StandardScaler for xVar. Inverse transforms x-axis for display.
        y_scaler: StandardScaler for yVar. Inverse transforms y-axis for display.
        z_scaler_mean: StandardScaler for the GP output. Always applied to the
            predicted mean panel (left subplot) for display in original units.
        z_scaler_unc: StandardScaler for the GP output uncertainty. Applied to
            the uncertainty panel (right subplot) only when provided (toggle-controlled).
        z_is_log (bool): If True, apply exp() after z_scaler_mean inverse transform
            on the scatter color values.
        n_grid (int): Resolution of the prediction grid per axis (default 100).
        color_scheme (str): "default", "colorblind", or "high_contrast".

    Returns:
        list[matplotlib.figure.Figure]: Single-element list containing the combined figure.
    """
    logVars = logVars or []

    palette, cmap_2d = get_color_palette(color_scheme)

    x_raw = df[xVar].astype(float).to_numpy()
    y_raw = df[yVar].astype(float).to_numpy()

    x_grid = np.geomspace(x_raw.min(), x_raw.max(), n_grid) if xVar in logVars else np.linspace(x_raw.min(), x_raw.max(), n_grid)
    y_grid = np.geomspace(y_raw.min(), y_raw.max(), n_grid) if yVar in logVars else np.linspace(y_raw.min(), y_raw.max(), n_grid)

    XX, YY = np.meshgrid(x_grid, y_grid)

    pred_df = {}
    for col in controlVars:
        if col == xVar:
            pred_df[col] = XX.ravel()
        elif col == yVar:
            pred_df[col] = YY.ravel()
        else:
            pred_df[col] = np.full(XX.size, df[col].median())

    X_pred = np.column_stack([pred_df[col] for col in controlVars]).astype(float)

    for i, col in enumerate(controlVars):
        if col in logVars:
            X_pred[:, i] = np.log10(X_pred[:, i])

    z_pred, z_std = gp.predict(X_pred, return_std=True)

    # Left panel (predicted mean): always unscale if z_scaler_mean provided
    if z_scaler_mean is not None:
        z_pred = z_scaler_mean.inverse_transform(z_pred.reshape(-1, 1)).ravel()
    z_pred = np.clip(z_pred, 0, None)
    Z_mean = z_pred.reshape(XX.shape)

    # Right panel (uncertainty): unscale only if z_scaler_unc provided (toggle-controlled)
    if z_scaler_unc is not None:
        z_std = z_std * z_scaler_unc.scale_[0]
    Z_unc = z_std.reshape(XX.shape)

    x_idx = controlVars.index(xVar)
    y_idx = controlVars.index(yVar)
    n_features = len(controlVars)

    x_display = descale_axis(x_grid, x_idx, x_scaler, n_features)
    y_display = descale_axis(y_grid, y_idx, y_scaler, n_features)
    XX_display, YY_display = np.meshgrid(x_display, y_display)

    scatter_x = descale_axis(df[xVar].to_numpy(), x_idx, x_scaler, n_features)
    scatter_y = descale_axis(df[yVar].to_numpy(), y_idx, y_scaler, n_features)

    if zLabel in df.columns:
        scatter_z = df[zLabel].to_numpy()
        if z_scaler_mean is not None:
            scatter_z = z_scaler_mean.inverse_transform(df[[zLabel]]).ravel()
            if z_is_log:
                scatter_z = np.exp(scatter_z)
    else:
        scatter_z = None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cf1 = axes[0].contourf(XX_display, YY_display, Z_mean, levels=40, cmap=cmap_2d)
    if scatter_z is not None:
        axes[0].scatter(scatter_x, scatter_y, c=scatter_z, cmap=cmap_2d, edgecolors='k')
    if xVar in logVars:
        axes[0].set_xscale('log')
    if yVar in logVars:
        axes[0].set_yscale('log')
    axes[0].set_xlabel(xVar)
    axes[0].set_ylabel(yVar)
    axes[0].set_title("GP Predicted Mean")
    fig.colorbar(cf1, ax=axes[0], label=zLabel_display if zLabel_display is not None else zLabel)

    cf2 = axes[1].contourf(XX_display, YY_display, Z_unc, levels=40, cmap=cmap_2d)
    axes[1].scatter(scatter_x, scatter_y, color=palette[1], edgecolors='k')
    if xVar in logVars:
        axes[1].set_xscale('log')
    if yVar in logVars:
        axes[1].set_yscale('log')
    axes[1].set_xlabel(xVar)
    axes[1].set_ylabel(yVar)
    axes[1].set_title("GP Uncertainty")
    fig.colorbar(cf2, ax=axes[1], label='std')

    plt.tight_layout()

    return [fig]
