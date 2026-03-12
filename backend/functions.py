import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


def group(data, controlVars, outputVar, blockVar=None, outlierSD=None):
    """
    Aggregate replicate observations by control variables and compute summary statistics.

    The function groups `data` by `controlVars`, optionally nested within `blockVar`,
    then computes the mean, standard deviation, and count of `outputVar` for each group.
    Groups with fewer than 2 retained observations are dropped. An optional outlier
    filter can remove points farther than `outlierSD` standard deviations from the
    group mean before aggregation.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing the grouping variables and output variable.
    controlVars : list[str]
        Column names used to define experimental conditions or grouping factors.
    outputVar : str
        Column name of the response variable to summarize.
    blockVar : str, optional
        Column name for a higher-level grouping variable. If provided, results are
        returned separately for each block.
    outlierSD : float, optional
        If provided, remove observations more than `outlierSD * std` from the group
        mean before computing summary statistics. Outlier filtering is only applied
        when a group has more than 2 observations.

    Returns
    -------
    pandas.DataFrame or dict
        If `blockVar` is None, returns a DataFrame with one row per group and columns
        for the grouping variables plus:
        - ``mean`` : mean of `outputVar`
        - ``std`` : standard deviation of `outputVar`
        - ``count`` : number of retained observations

        If `blockVar` is provided, returns a dictionary mapping each block value to
        its corresponding summary DataFrame.

    Notes
    -----
    Groups with fewer than 2 observations after optional outlier filtering are dropped.
    A warning is printed if any groups are excluded.
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

def fit_gp(df, controlVars, logVars=None, noise="std", kernel = None):
    """
    Fit a Gaussian process regressor to grouped summary data.

    The model is trained using the columns in `controlVars` as predictors and the
    ``mean`` column of `df` as the target. Predictors listed in `logVars` are
    transformed with base-10 logarithms before fitting. Observation noise is derived
    from either the group standard deviation or standard error of the mean.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing predictor columns and the required columns:
        ``mean``, ``std``, and ``count``.
    controlVars : list[str]
        Column names to use as model inputs.
    logVars : list[str], optional
        Predictor columns that should be log10-transformed before fitting.
    noise : {"std", "sem"}, default="std"
        Method used to define observation noise:
        - ``"std"`` uses variance based on the group standard deviation
        - ``"sem"`` uses variance based on the standard error of the mean

    Returns
    -------
    sklearn.gaussian_process.GaussianProcessRegressor
        The fitted Gaussian process model.

    Raises
    ------
    ValueError
        If `noise` is not one of ``"std"`` or ``"sem"``.

    Notes
    -----
    Noise variances are clipped to a minimum of ``1e-5`` for numerical stability.
    The function prints the final `alpha` vector and optimized kernel after fitting.
    """
    logVars = logVars or []

    X = df[controlVars].copy().astype(float)
    for col in logVars:
        X[col] = np.log10(X[col])

    y = df["mean"].to_numpy(dtype=float)

    if noise == "sem":
        alpha = (df["std"].to_numpy(dtype=float) / np.sqrt(df["count"].to_numpy(dtype=float))) ** 2
    elif noise == "std":
        alpha = df["std"].to_numpy(dtype=float) ** 2
    else:
        raise ValueError("noise must be 'std' or 'sem'")

    alpha = np.clip(alpha, 1e-5, None)
    print("alpha =", alpha)

    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
        length_scale=np.ones(len(controlVars)),
        length_scale_bounds=(0.5, 5)
    )

    if kernel is None:
        raise ValueError("Kernel must be provided")

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=5
    )

    gp.fit(X.to_numpy(), y)
    print("kernel =", gp.kernel_)
    return gp

def plot_gp(df, gp, controlVars, xVar, logVars=None, fixedVals=None):
    """
    Plot a 1D slice of Gaussian process predictions against observed group means.

    This function varies one predictor (`xVar`) over a grid while holding all other
    predictors fixed, then plots the GP predicted mean and a 95% confidence interval.
    Observed training data are overlaid as scatter points. If `fixedVals` is provided,
    only matching rows are shown in the scatter plot for contextual comparison.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing predictor columns and a ``mean`` column.
    gp : sklearn.gaussian_process.GaussianProcessRegressor
        A fitted Gaussian process model.
    controlVars : list[str]
        Predictor columns used when fitting the GP.
    xVar : str
        The predictor to vary along the x-axis.
    logVars : list[str], optional
        Predictor columns that are modeled on a log10 scale. If `xVar` is included,
        the x-grid is spaced geometrically and the x-axis is shown on a log scale.
    fixedVals : dict, optional
        Mapping from predictor names to fixed values for the non-varying dimensions.
        Any predictor not specified here and not equal to `xVar` is fixed at its
        median value in `df`.

    Returns
    -------
    None
        Displays a matplotlib plot.

    Notes
    -----
    The shaded band represents approximately a 95% confidence interval
    (``mean ± 1.96 * std``) from the GP posterior.
    """
    logVars = logVars or []
    fixedVals = fixedVals or {}

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

    # only show slice since using 2 variables for 1D plot. ex: if we want to use volume we have to use volume at a tixed level 
    plot_df = df.copy()
    for col, val in fixedVals.items():
        plot_df = plot_df[plot_df[col] == val]

    plt.figure(figsize=(8, 4))
    plt.scatter(plot_df[xVar], plot_df['mean'], color='red', label='Training Data')
    plt.plot(x_grid, y_pred, color='black', label='Predicted Mean')
    plt.fill_between(
        x_grid,
        y_pred - 1.96 * y_std,
        y_pred + 1.96 * y_std,
        color='gray',
        alpha=0.35,
        label='95% Confidence Interval'
    )

    if xVar in logVars:
        plt.xscale('log')

    plt.xlabel(xVar)
    plt.ylabel('mean')
    plt.legend()
    plt.show()

def plot_gp_2d(df, gp, controlVars, xVar, yVar, logVars=None, n_grid=100):
    """
    Plot 2D Gaussian process mean and uncertainty surfaces.

    This function evaluates the fitted GP over a 2D grid defined by `xVar` and `yVar`,
    while holding all remaining predictors at their median values. It produces two
    contour plots: one for the GP predicted mean and one for the predictive standard
    deviation.

    Parameters
    ----------
    df : pandas.DataFrame
        Summary DataFrame containing predictor columns and a ``mean`` column.
    gp : sklearn.gaussian_process.GaussianProcessRegressor
        A fitted Gaussian process model.
    controlVars : list[str]
        Predictor columns used when fitting the GP.
    xVar : str
        Predictor to use on the x-axis.
    yVar : str
        Predictor to use on the y-axis.
    logVars : list[str], optional
        Predictor columns that are modeled on a log10 scale. If included, the
        corresponding grid axis is spaced geometrically and displayed on a log scale.
    n_grid : int, default=100
        Number of grid points per axis for evaluating the GP surface.

    Returns
    -------
    None
        Displays two matplotlib plots:
        1. GP predicted mean
        2. GP predictive standard deviation

    Notes
    -----
    All predictors other than `xVar` and `yVar` are fixed at their median values in
    `df` when generating the prediction surface.
    """
    logVars = logVars or []

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
    plt.contourf(XX, YY, Z_mean, levels=40)
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

#==============================================================
# Example Usage 
#==============================================================

df = pd.read_csv("output.csv")

gp_data = group(
    data=df,
    controlVars=['concentration','volume'],
    outputVar='lambda max wavelength',
    blockVar='additive',
    outlierSD=1
)

# Select block
block_df = gp_data['C8']

gp = fit_gp(
    block_df,
    controlVars=['concentration', 'volume'],
    logVars=['concentration'],
    noise="sem"
)

plot_gp(
    block_df,
    gp,
    controlVars=['concentration','volume'],
    xVar='concentration',
    logVars=['concentration'],
    fixedVals={'volume': 10}
)

plot_gp_2d(
    block_df,
    gp,
    controlVars=['concentration', 'volume'],
    xVar='concentration',
    yVar='volume',
    logVars=['concentration']
)

print(gp.kernel_)