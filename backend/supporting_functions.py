# =============================================================================
# backend/supporting_functions.py
#
# Pipeline orchestration for the GPR web platform.
# Sits between the Flask routes and the core functions in functions.py.
#
# Functions:
#   load_dataset()              — load CSV into DataFrame
#   auto_detect_features()      — classify columns as numeric or categorical
#   apply_feature_engineering() — evaluate user-defined column expressions
#   preprocess()                — scale numerics, one-hot encode categoricals,
#                                 build control_vars and category_combos
#   build_kernel()              — construct an sklearn kernel from a config dict
#   run_gp_pipeline()           — end-to-end pipeline for std or mean mode
#   _build_combos()             — build full combo list for plot iteration
#   generate_plot()             — render 1D/2D plots and return base64 PNGs
#   generate_plot_both()        — side-by-side std/mean plot pairs for both mode
# =============================================================================

import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from backend.functions import group, fit_gp, plot_gp, plot_gp_2d
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

# ======================================================
# DATASET LOADING
# ======================================================
def load_dataset(file):
    """Load a CSV file and return it as a DataFrame."""
    return pd.read_csv(file)

# ======================================================
# AUTO DETECT NUMERIC / CATEGORICAL
# ======================================================
def auto_detect_features(df, target_col):
    """
    Classify columns as numerical or categorical, excluding the target.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Column to exclude from the feature lists.

    Returns:
        dict: {"num_cols": [...], "cat_cols": [...]}
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    return {"num_cols": num_cols, "cat_cols": cat_cols}

# ======================================================
# FEATURE ENGINEERING
# ======================================================
def apply_feature_engineering(df, expressions):
    """
    Evaluate user-defined column expressions and append them to the dataframe.

    Each expression must be in the form "new_col = formula", where formula can
    reference existing columns by name and numpy as "np".

    Args:
        df (pd.DataFrame): Input dataframe.
        expressions (list[str]): List of "name = formula" strings.

    Returns:
        pd.DataFrame: Dataframe with new columns added.

    Raises:
        ValueError: If an expression cannot be evaluated.
    """
    if not expressions:
        return df
    for expr in expressions:
        if "=" not in expr:
            continue
        new_col, formula = expr.split("=", 1)
        new_col = new_col.strip()
        formula = formula.strip()
        try:
            safe_dict = {"np": np, "df": df}
            safe_dict.update({col: df[col] for col in df.columns})
            df[new_col] = eval(formula, {"__builtins__": {}}, safe_dict)
        except Exception as e:
            raise ValueError(f"Error in expression '{expr}': {e}")
    return df

# ======================================================
# PREPROCESSING
# ======================================================
def preprocess(df, num_cols, cat_cols, scale_num=True):
    """
    Scale numeric columns and one-hot encode categoricals.

    Fits a StandardScaler per numeric column and stores each scaler in
    num_scalers so that predictions can be inverse-transformed for display.
    One-hot encodes categoricals via pd.get_dummies and builds a combo list
    used later to generate per-category plots.

    Args:
        df (pd.DataFrame): Input dataframe.
        num_cols (list[str]): Numeric feature columns to scale.
        cat_cols (list[str]): Categorical columns to one-hot encode.
        scale_num (bool): Whether to apply StandardScaler to num_cols.

    Returns:
        tuple:
            df (pd.DataFrame): Processed dataframe with scaled numerics and
                dummy columns replacing the originals.
            control_vars (list[str]): Ordered GP input columns
                (scaled numerics + dummy columns).
            category_combos (list[dict]): Each entry has "label" (str | None)
                and "fixedVals" (dict mapping dummy col → 0/1).
                Contains a single placeholder entry when cat_cols is empty.
            num_scalers (dict): Maps column name → fitted StandardScaler.
    """
    df = df.copy()
    num_scalers = {}
    dummy_cols = []
    category_combos = []

    if scale_num and num_cols:
        for col in num_cols:
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
            num_scalers[col] = scaler

    if cat_cols:
        unique_combos = df[cat_cols].drop_duplicates().to_dict(orient="records")
        cols_before = set(df.columns)
        df = pd.get_dummies(df, columns=cat_cols)
        dummy_cols = [c for c in df.columns if c not in cols_before]
        for col in dummy_cols:
            df[col] = df[col].astype(int)
        for combo in unique_combos:
            fixed = {col: 0 for col in dummy_cols}
            for cat_col, cat_val in combo.items():
                dummy_name = f"{cat_col}_{cat_val}"
                if dummy_name in dummy_cols:
                    fixed[dummy_name] = 1
            label = ", ".join(f"{k}={v}" for k, v in combo.items())
            category_combos.append({"label": label, "fixedVals": fixed})
    else:
        category_combos = [{"label": None, "fixedVals": {}}]

    control_vars = num_cols + dummy_cols
    return df, control_vars, category_combos, num_scalers

# ======================================================
# KERNEL CONFIGURATION
# ======================================================
def build_kernel(n_features, kernel_config):
    """
    Construct an sklearn kernel from a configuration dict.

    Args:
        n_features (int): Number of input dimensions; used for ARD length scales.
        kernel_config (dict | None): Configuration with optional keys:
            kernel_type (str): "rbf" | "matern" | "rq" (default "rbf")
            ard (bool): Use per-dimension length scales (default True)
            length_scale_init (float): Initial length scale value
            length_scale_bounds (tuple): (lower, upper) bounds for optimization
            nu (float): Matérn smoothness — 0.5, 1.5, or 2.5
            rq_alpha_init (float): RationalQuadratic alpha initial value
            rq_alpha_bounds (tuple): (lower, upper) for RQ alpha
            white_noise (bool): Use WhiteKernel instead of ConstantKernel wrapper
            constant_value / constant_bounds: ConstantKernel params
            noise_level / noise_lower / noise_upper: WhiteKernel params

    Returns:
        sklearn kernel | None: Constructed kernel, or None if kernel_config is falsy.
    """
    if not kernel_config:
        return None

    kernel_type = kernel_config.get("kernel_type", "rbf")
    ard = kernel_config.get("ard", True)
    length_scale_init = kernel_config.get("length_scale_init", 1.0)
    length_scale_bounds = kernel_config.get("length_scale_bounds", (0.5, 5))
    nu = kernel_config.get("nu", 1.5)

    constant_value = kernel_config.get("constant_value", 1.0)
    constant_bounds = kernel_config.get("constant_bounds", (1e-2, 1e2))
    use_white_noise = kernel_config.get("white_noise", False)
    noise_level = kernel_config.get("noise_level", 1e-4)
    noise_lower = kernel_config.get("noise_lower", 1e-8)
    noise_upper = kernel_config.get("noise_upper", 1e-2)

    if ard:
        length_scale = np.ones(n_features) * length_scale_init
    else:
        length_scale = length_scale_init

    if kernel_type == "rbf":
        base_kernel = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
    elif kernel_type == "matern":
        base_kernel = Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
    elif kernel_type == "rq":
        rq_alpha_init = kernel_config.get("rq_alpha_init", 1.0)
        rq_alpha_bounds = kernel_config.get("rq_alpha_bounds", (1e-2, 1e2))
        base_kernel = RationalQuadratic(length_scale=length_scale, alpha=rq_alpha_init, 
                                        alpha_bounds=rq_alpha_bounds)
    else:
        raise ValueError("Unsupported kernel type")

    if use_white_noise:
        kernel = base_kernel + WhiteKernel(noise_level=noise_level, 
                                           noise_level_bounds=(noise_lower, noise_upper))
    else:
        kernel = ConstantKernel(constant_value, constant_bounds) * base_kernel

    return kernel

# ======================================================
# GPR PIPELINE  (mode = "std" | "mean")
# ======================================================
def run_gp_pipeline(df, num_cols, cat_cols=None, output_col=None, measurement_col=None,
                    gp_target="mean", mode="std", logVars=None, noise="std", kernel_config=None):
    """
    End-to-end GP pipeline: preprocess → (optionally group) → fit → return results.

    Two modes are supported:
      - "std": groups replicate measurements by control variables, then fits a GP
        on either the per-group mean or std of measurement_col.
      - "mean": skips grouping and fits a GP directly on each row using output_col
        as the target. Noise is fixed to "constant".

    Args:
        df (pd.DataFrame): Raw input dataframe.
        num_cols (list[str]): Numeric feature columns (will be scaled).
        cat_cols (list[str] | None): Categorical feature columns (will be one-hot encoded).
        output_col (str | None): Target column for mean mode.
        measurement_col (str | None): Raw measurement column for std mode.
        gp_target (str): Which grouped statistic to fit: "mean" or "std" (std mode only).
        mode (str): "std" or "mean".
        logVars (list[str] | None): Columns to treat on log scale in fit_gp.
        noise (str): Noise model for fit_gp — "std" (from grouped std) or "constant".
        kernel_config (dict | None): Kernel configuration passed to build_kernel.

    Returns:
        dict with keys:
            model: Fitted GaussianProcessRegressor.
            gp_data (pd.DataFrame): Preprocessed (and grouped, in std mode) data.
            control_vars (list[str]): GP input columns.
            category_combos (list[dict]): Per-category plot combos.
            num_scalers (dict): Per-column StandardScalers.
            metrics (dict): R², MAE, RMSE on training data.
            gp_target (str): The GP output column ("mean"/"std" in std mode, output_col in mean mode).
            measurement_col (str | None): Original measurement column (std mode) or None.
            mode (str): The mode used ("std" or "mean").
    """
    cat_cols = cat_cols or []
    logVars = logVars or []

    df_processed, control_vars, category_combos, num_scalers = preprocess(df, num_cols, cat_cols, 
                                                                          scale_num=True)

    n_features = len(control_vars)
    kernel = build_kernel(n_features, kernel_config)

    if mode == "std":
        # scale measurement_col before grouping
        if measurement_col and measurement_col in df_processed.columns:
            m_scaler = StandardScaler()
            df_processed[measurement_col] = m_scaler.fit_transform(df_processed[[measurement_col]])
            num_scalers[measurement_col] = m_scaler

        gp_data = group(df_processed, controlVars=control_vars, outputVar=measurement_col)
        gp, metrics = fit_gp(gp_data, controlVars=control_vars, kernel=kernel, 
                             outputVar=gp_target, logVars=logVars, noise=noise)
    else:
        # mean mode: fit directly on raw rows
        gp_data = df_processed.copy()
        # Scale output_col (matching notebook behavior)
        if output_col and output_col in gp_data.columns:
            o_scaler = StandardScaler()
            gp_data[output_col] = o_scaler.fit_transform(gp_data[[output_col]])
            num_scalers[output_col] = o_scaler
        gp, metrics = fit_gp(gp_data, controlVars=control_vars, kernel=kernel, 
                             outputVar=output_col, logVars=logVars, noise="constant")

    return {
        "model": gp,
        "gp_data": gp_data,
        "control_vars": control_vars,
        "category_combos": category_combos,
        "num_scalers": num_scalers,
        "metrics": metrics,
        "gp_target": gp_target if mode == "std" else output_col,
        "measurement_col": measurement_col if mode == "std" else None,
        "mode": mode,
    }

# ======================================================
# PLOT GENERATION
# ======================================================
def _fig_to_b64(fig):
    """Save a matplotlib figure to a base64-encoded PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    result = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return result


def _build_combos(category_combos):
    """
    Build the full combo list for plot iteration.
    Prepends an 'All' combo only when real categorical combos exist.
    Avoids duplicating the placeholder combo used when there are no categoricals.
    """
    real_cats = [c for c in category_combos if c.get("label") is not None]
    if real_cats:
        return [{"label": "All", "fixedVals": {}}] + real_cats
    return [{"label": None, "fixedVals": {}}]


def generate_plot(gp_data, gp, control_vars, dimension, xVar, yVar="mean", logVars=None,
                  category_combos=None, color_scheme="default", num_scalers=None,
                  y_scaler=None, z_scaler_mean=None, z_scaler_unc=None,
                  gp_target="mean", zLabel=None, zLabel_display=None, ylabel=None):
    """
    Render GP plots for all category combos and return them as base64 PNG strings.

    For 1D plots, iterates over all combos from _build_combos (one image per combo).
    For 2D plots, calls plot_gp_2d which returns one figure (two panels: mean + uncertainty).

    Unscaling rules:
      - x-axis: always unscaled via num_scalers[xVar] when available.
      - y-axis (1D): unscaled when y_scaler is provided. Auto-applied for std mode
        replicate-mean target and mean mode; toggle-controlled for std uncertainty.
      - 2D left panel (mean): always unscaled via z_scaler_mean.
      - 2D right panel (uncertainty): unscaled only when z_scaler_unc is provided
        (controlled by the "Unscale std axis" checkbox in the UI).

    Args:
        gp_data (pd.DataFrame): Training dataframe passed to plot_gp / plot_gp_2d.
        gp: Fitted GaussianProcessRegressor.
        control_vars (list[str]): GP input columns.
        dimension (str): "1d" or "2d".
        xVar (str): Column for the x-axis (and left axis in 2D).
        yVar (str): Target column for 1D scatter; right axis variable in 2D.
        logVars (list[str] | None): Columns to display on log scale.
        category_combos (list[dict]): Each entry has "label" and "fixedVals".
        color_scheme (str): "default", "colorblind", or "high_contrast".
        num_scalers (dict): Per-column StandardScalers keyed by column name.
        y_scaler: Scaler for 1D y-axis (None disables unscaling).
        z_scaler_mean: Scaler for 2D predicted mean panel (always applied when set).
        z_scaler_unc: Scaler for 2D uncertainty panel (None disables unscaling).
        gp_target (str): GP output column name used for scatter lookup.
        zLabel (str | None): Column name used for 2D colorbar lookup. Defaults to gp_target.
        zLabel_display (str | None): Display text for the 2D colorbar label.
            If None, falls back to zLabel.
        ylabel (str | None): Override for 1D y-axis label. Used in std mode when
            gp_target is "mean" so the label shows the original measurement column name.

    Returns:
        list[str]: Base64-encoded PNG images, one per category combo (1D) or
            one per figure returned by plot_gp_2d (2D).
    """
    category_combos = category_combos or [{"label": None, "fixedVals": {}}]
    num_scalers = num_scalers or {}
    zLabel = zLabel or gp_target
    images = []

    if dimension == "2d":
        for fig in plot_gp_2d(gp_data, gp, control_vars, xVar, yVar,
                               zLabel=zLabel,
                               zLabel_display=zLabel_display,
                               logVars=logVars, color_scheme=color_scheme,
                               x_scaler=num_scalers.get(xVar),
                               y_scaler=num_scalers.get(yVar),
                               z_scaler_mean=z_scaler_mean,
                               z_scaler_unc=z_scaler_unc):
            images.append(_fig_to_b64(fig))
        return images

    for combo in _build_combos(category_combos):
        fixedVals = combo.get("fixedVals", {})
        label = combo.get("label")
        plot_gp(gp_data, gp, control_vars, xVar, yVar=yVar, logVars=logVars,
                fixedVals=fixedVals, title=label, color_scheme=color_scheme,
                x_scaler=num_scalers.get(xVar), x_scaler_col=xVar,
                y_scaler=y_scaler, ylabel=ylabel)
        images.append(_fig_to_b64(plt.gcf()))

    return images

def generate_plot_both(std_result, mean_result,
                       dimension, xVar,
                       logVars=None, color_scheme="default", unscale_std=False):
    """
    Render side-by-side std/mean plot pairs for "both" mode.

    Iterates over the longer of the two combo lists and renders one std plot and
    one mean plot per iteration. plt.close("all") is called before each side to
    prevent matplotlib figure state from bleeding between renders.

    Unscaling rules:
      - std side 1D: y-axis auto-unscaled when gp_target is "mean" (replicate mean);
        unscaled only when unscale_std=True for "std" gp_target.
      - std side 2D: left panel always unscaled; right panel unscaled when unscale_std=True.
      - mean side 1D: y-axis always unscaled (output column scaler).
      - mean side 2D: left panel always unscaled; right panel unscaled when unscale_std=True.

    Args:
        std_result (dict): Result dict from run_gp_pipeline with mode="std".
        mean_result (dict): Result dict from run_gp_pipeline with mode="mean".
        dimension (str): "1d" or "2d".
        xVar (str): Column for the x-axis.
        logVars (list[str] | None): Columns to display on log scale.
        color_scheme (str): "default", "colorblind", or "high_contrast".
        unscale_std (bool): Whether to unscale the uncertainty (right) panels.

    Returns:
        list[dict]: Each entry has "std" and/or "mean" keys mapped to base64 PNG strings.
            One dict per category combo iteration.
    """
    std_combos = _build_combos(std_result["category_combos"])
    mean_combos = _build_combos(mean_result["category_combos"])

    pairs = []
    n = max(len(std_combos), len(mean_combos))

    for i in range(n):
        pair = {}

        if i < len(std_combos):
            combo = std_combos[i]
            std_scalers = std_result.get("num_scalers", {})
            plt.close("all")  # clear any leftover state before plotting std side
            measurement_col = std_result.get("measurement_col")
            if dimension == "1d":
                plot_gp(
                    std_result["gp_data"], std_result["model"],
                    std_result["control_vars"], xVar,
                    yVar=std_result["gp_target"],
                    logVars=logVars,
                    fixedVals=combo.get("fixedVals", {}),
                    title=combo.get("label"),
                    color_scheme=color_scheme,
                    x_scaler=std_scalers.get(xVar),
                    x_scaler_col=xVar,
                    y_scaler=std_scalers.get(measurement_col) if unscale_std else None,
                    ylabel=measurement_col if std_result["gp_target"] == "mean" and measurement_col else None
                )
                fig = plt.gcf()
            else:
                yVar2 = std_result["control_vars"][1] if len(std_result["control_vars"]) > 1 else xVar
                fig = plot_gp_2d(
                    std_result["gp_data"], std_result["model"],
                    std_result["control_vars"], xVar, yVar2,
                    zLabel=std_result["gp_target"],
                    zLabel_display=measurement_col,
                    logVars=logVars, color_scheme=color_scheme,
                    x_scaler=std_scalers.get(xVar),
                    y_scaler=std_scalers.get(yVar2),
                    z_scaler_mean=std_scalers.get(measurement_col),
                    z_scaler_unc=std_scalers.get(measurement_col) if unscale_std else None
                )[0]
            pair["std"] = _fig_to_b64(fig)

        if i < len(mean_combos):
            combo = mean_combos[i]
            mean_scalers = mean_result.get("num_scalers", {})
            plt.close("all")  # clear any leftover state before plotting mean side
            if dimension == "1d":
                plot_gp(
                    mean_result["gp_data"], mean_result["model"],
                    mean_result["control_vars"], xVar,
                    yVar=mean_result["gp_target"],
                    logVars=logVars,
                    fixedVals=combo.get("fixedVals", {}),
                    title=combo.get("label"),
                    color_scheme=color_scheme,
                    x_scaler=mean_scalers.get(xVar),
                    x_scaler_col=xVar,
                    y_scaler=mean_scalers.get(mean_result["gp_target"])
                )
                fig = plt.gcf()
            else:
                yVar2 = mean_result["control_vars"][1] if len(mean_result["control_vars"]) > 1 else xVar
                fig = plot_gp_2d(
                    mean_result["gp_data"], mean_result["model"],
                    mean_result["control_vars"], xVar, yVar2,
                    zLabel=mean_result["gp_target"],
                    logVars=logVars, color_scheme=color_scheme,
                    x_scaler=mean_scalers.get(xVar),
                    y_scaler=mean_scalers.get(yVar2),
                    z_scaler_mean=mean_scalers.get(mean_result["gp_target"]),
                    z_scaler_unc=mean_scalers.get(mean_result["gp_target"]) if unscale_std else None
                )[0]
            pair["mean"] = _fig_to_b64(fig)

        pairs.append(pair)

    return pairs
