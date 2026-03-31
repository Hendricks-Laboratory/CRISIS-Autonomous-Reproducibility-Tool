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
#   plot_gp()            — 1D GP plot with confidence interval
#   plot_gp_2d()         — 2D GP contour plots for mean and uncertainty
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
        raise ValueError("noise must be 'std', 'sem' and constant")

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
    print("kernel =", gp.kernel_)

    y_pred = gp.predict(X.to_numpy())

    metrics = {
        "r2": float(r2_score(y, y_pred)),
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
    }

    return gp, metrics


def get_color_palette(color_scheme="default"):
    """Return (palette_1d, cmap_2d) for the requested color scheme."""
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


def plot_gp(df, gp, controlVars, xVar, yVar="mean", logVars=None, fixedVals=None, title=None, color_scheme="default"):
    """
    Generate a 1D GP plot: scatter of training data + predicted mean ± 95% CI.

    Args:
        df (pd.DataFrame): Training dataframe (post-grouping in std mode).
        gp: Fitted GaussianProcessRegressor.
        controlVars (list[str]): GP input columns.
        xVar (str): Column to vary along the x-axis.
        yVar (str): Column to plot as the observed target (default "mean").
        logVars (list[str]): Columns displayed on a log scale.
        fixedVals (dict): Values to hold constant for non-xVar control vars.
            When empty, other vars are held at their median.
        title (str | None): Plot title.
        color_scheme (str): "default", "colorblind", or "high_contrast".
    """
    logVars = logVars or []
    fixedVals = fixedVals or {}

    palette, _ = get_color_palette(color_scheme)
    color_line = palette[0]
    color_scatter = palette[1]

    x_raw = np.sort(df[xVar].astype(float).unique())
    x_grid = np.geomspace(x_raw.min(), x_raw.max(), 300) if xVar in logVars else np.linspace(x_raw.min(), x_raw.max(), 300)

    X_pred_rows = []
    for x in x_grid:
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

    plot_df = df.copy()
    for col, val in fixedVals.items():
        plot_df = plot_df[plot_df[col] == val]

    plt.figure(figsize=(8, 4))
    plt.scatter(plot_df[xVar], plot_df[yVar], color=color_scatter, label='Training Data')
    plt.plot(x_grid, y_pred, color=color_line, label='Predicted Mean')
    plt.fill_between(
        x_grid,
        y_pred - 1.96 * y_std,
        y_pred + 1.96 * y_std,
        color=color_line,
        alpha=0.2,
        label='95% Confidence Interval'
    )

    if xVar in logVars:
        plt.xscale('log')

    plt.xlabel(xVar)
    plt.ylabel(yVar)
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def plot_gp_2d(df, gp, controlVars, xVar, yVar, logVars=None, n_grid=100, color_scheme="default"):
    """
    Generate 2D GP contour plots: predicted mean and uncertainty over a grid.

    Produces two figures — one for the predicted mean and one for the
    predictive standard deviation — with training points overlaid.

    Args:
        df (pd.DataFrame): Training dataframe.
        gp: Fitted GaussianProcessRegressor.
        controlVars (list[str]): GP input columns.
        xVar (str): Column mapped to the x-axis.
        yVar (str): Column mapped to the y-axis.
        logVars (list[str]): Columns displayed on a log scale.
        n_grid (int): Resolution of the prediction grid per axis (default 100).
        color_scheme (str): "default", "colorblind", or "high_contrast".
    """
    logVars = logVars or []

    _, cmap_2d = get_color_palette(color_scheme)

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

    y_pred, y_std = gp.predict(X_pred, return_std=True)

    Z_mean = y_pred.reshape(XX.shape)
    Z_std = y_std.reshape(XX.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(XX, YY, Z_mean, levels=40, cmap=cmap_2d)
    plt.scatter(df[xVar], df[yVar], c=df['mean'], edgecolors='k')
    if xVar in logVars:
        plt.xscale('log')
    if yVar in logVars:
        plt.yscale('log')
    plt.xlabel(xVar)
    plt.ylabel(yVar)
    plt.title("GP Predicted Mean")
    plt.colorbar(label='mean')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.contourf(XX, YY, Z_std, levels=40)
    plt.scatter(df[xVar], df[yVar], color='red', edgecolors='k')
    if xVar in logVars:
        plt.xscale('log')
    if yVar in logVars:
        plt.yscale('log')
    plt.xlabel(xVar)
    plt.ylabel(yVar)
    plt.title("GP Uncertainty")
    plt.colorbar(label='std')
    plt.show()
