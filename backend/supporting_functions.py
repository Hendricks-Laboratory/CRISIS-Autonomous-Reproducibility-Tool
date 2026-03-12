import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel

from backend.functions import group, fit_gp, plot_gp, plot_gp_2d


# ======================================================
# DATASET LOADING
# ======================================================

def load_dataset(file):

    df = pd.read_csv(file)

    return df


# ======================================================
# AUTO DETECT NUMERIC / CATEGORICAL
# ======================================================

def auto_detect_features(df, target_col):

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if target_col in num_cols:
        num_cols.remove(target_col)

    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    return {
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }


# ======================================================
# FEATURE ENGINEERING
# ======================================================

def apply_feature_engineering(df, expressions):

    if not expressions:
        return df

    for expr in expressions:

        if "=" not in expr:
            continue

        new_col, formula = expr.split("=", 1)

        new_col = new_col.strip()
        formula = formula.strip()

        try:

            safe_dict = {
                "np": np,
                "df": df
            }

            safe_dict.update({col: df[col] for col in df.columns})

            df[new_col] = eval(formula, {"__builtins__": {}}, safe_dict)

        except Exception as e:

            raise ValueError(f"Error in expression '{expr}': {e}")

    return df


# ======================================================
# KERNEL CONFIGURATION
# ======================================================

def build_kernel(n_features, kernel_config=None):

    # DEFAULT RESEARCH VALUES

    kernel_type = "rbf"

    length_scale_init = 1.0
    length_scale_bounds = (0.5, 5)

    constant_value = 1.0
    constant_bounds = (1e-2, 1e2)

    ard = True


    if kernel_config:

        kernel_type = kernel_config.get("kernel_type", "rbf")

        if kernel_config.get("advanced", False):

            ard = kernel_config.get("ard", True)

            length_scale_init = kernel_config.get(
                "length_scale_init",
                length_scale_init
            )

            length_scale_bounds = kernel_config.get(
                "length_scale_bounds",
                length_scale_bounds
            )

            constant_value = kernel_config.get(
                "constant_value",
                constant_value
            )

            constant_bounds = kernel_config.get(
                "constant_bounds",
                constant_bounds
            )


    # ARD

    if ard:
        length_scale = np.ones(n_features) * length_scale_init
    else:
        length_scale = length_scale_init


    # KERNEL SELECTION

    if kernel_type == "rbf":

        base_kernel = RBF(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds
        )

    elif kernel_type == "matern":

        base_kernel = Matern(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            nu=1.5
        )

    elif kernel_type == "rq":

        base_kernel = RationalQuadratic(
            length_scale=length_scale,
            alpha=1.0
        )

    else:
        raise ValueError("Unsupported kernel type")


    # FINAL KERNEL

    kernel = ConstantKernel(constant_value, constant_bounds) * base_kernel


    return kernel


def group_data(df,
                    controlVars,
                    outputVar,
                    blockVar=None,
                    logVars=None,
                    outlierSD=None,
                    noise="std"):

    grouped = group(
        data=df,
        controlVars=controlVars,
        outputVar=outputVar,
        blockVar=blockVar,
        outlierSD=outlierSD
    )

    return grouped

def run_gp_block(block_df,
                 controlVars,
                 logVars=None,
                 noise="std"):

    gp = fit_gp(
        block_df,
        controlVars=controlVars,
        logVars=logVars,
        noise=noise
    )

    return gp

def run_random_forest(X, y):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=0
    )

    model.fit(X, y)

    return model

def generate_plot(df,
                  gp,
                  controlVars,
                  dimension,
                  xVar,
                  yVar=None,
                  logVars=None,
                  fixedVals=None):

    if dimension == "1d":

        plot_gp(
            df,
            gp,
            controlVars,
            xVar,
            logVars=logVars,
            fixedVals=fixedVals
        )

    elif dimension == "2d":

        plot_gp_2d(
            df,
            gp,
            controlVars,
            xVar,
            yVar,
            logVars=logVars
        )