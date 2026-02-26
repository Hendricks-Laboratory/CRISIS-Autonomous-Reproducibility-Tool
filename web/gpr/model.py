# gpr/model.py

import numpy as np
from itertools import combinations
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, DotProduct
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def auto_detect_features(df, target_col):
    import numpy as np

    # Automatic detection
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    return {
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }

def apply_feature_engineering(df, expressions):
    """
    expressions: list of strings like:
        ["new_col = col1 * col2", "log_col = np.log1p(col1)"]
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
            safe_dict = {
                "np": np,
                "df": df
            }

            # Allow direct column reference
            safe_dict.update({col: df[col] for col in df.columns})

            df[new_col] = eval(formula, {"__builtins__": {}}, safe_dict)

        except Exception as e:
            raise ValueError(f"Error in expression '{expr}': {e}")

    return df

def run_gp_pipeline(
    df,
    target_col,
    mode="auto",
    num_cols=None,
    cat_cols=None,
    kernel_config=None
):

    if target_col not in df.columns:
        raise ValueError("Target column not found.")

    # ---------- FEATURE SELECTION ----------

    if num_cols is None:
        num_cols = []

    if cat_cols is None:
        cat_cols = []

    if len(num_cols) == 0 and len(cat_cols) == 0:
        raise ValueError("No feature columns selected.")

    # ---------- SCALING ----------
    scaler = None
    encoder = None

    if num_cols:
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(df[num_cols])
    else:
        X_num_scaled = None

    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = encoder.fit_transform(df[cat_cols])
    else:
        X_cat = None

    # ---------- FEATURE MATRIX ----------
    if X_num_scaled is not None and X_cat is not None:
        X = np.hstack([X_num_scaled, X_cat])
    elif X_num_scaled is not None:
        X = X_num_scaled
    else:
        X = X_cat

    y = df[target_col].to_numpy(dtype=float)

    # ---------- TRAIN / TEST SPLIT ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # ---------- MODEL ----------
    n_features = X_train.shape[1]

    # ------------------------
    # DEFAULT RESEARCH VALUES
    # ------------------------
    kernel_type = "rbf"
    length_scale_init = 1.0
    length_lower = 1e-2
    length_upper = 1e2
    noise_level = 1e-4
    alpha = 1e-6
    restarts = 5
    ard = True  

    if kernel_config:

        kernel_type = kernel_config.get("kernel_type", "rbf")

        if kernel_config.get("advanced", False):

            length_scale_init = kernel_config.get("length_scale_init", 1.0)
            length_lower = kernel_config.get("length_scale_lower", 1e-2)
            length_upper = kernel_config.get("length_scale_upper", 1e2)
            noise_level = kernel_config.get("noise_level", 1e-4)
            alpha = kernel_config.get("alpha", 1e-6)
            restarts = kernel_config.get("restarts", 5)

    # ------------------------
    # ARD LENGTH SCALE
    # ------------------------
    if ard:
        length_scale = np.ones(n_features) * length_scale_init
    else:
        length_scale = length_scale_init

    # ------------------------
    # KERNEL CONSTRUCTION
    # ------------------------
    if kernel_type == "rbf":

        kernel = RBF(
            length_scale=length_scale,
            length_scale_bounds=(length_lower, length_upper)
        )

    elif kernel_type == "matern":

        kernel = Matern(
            length_scale=length_scale,
            length_scale_bounds=(length_lower, length_upper),
            nu=1.5
        )

    elif kernel_type == "rq":

        kernel = RationalQuadratic(
            length_scale=length_scale,
            alpha=1.0
        )

    else:
        raise ValueError("Unsupported kernel type")

    kernel += WhiteKernel(noise_level=noise_level)

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=restarts,
        random_state=0
    )

    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

    # ---------- FEATURE IMPORTANCE ----------
    ranked_features = []
    top_pairs_2d = []

    if len(num_cols) > 0:

        length_scales = gpr.kernel_.k1.length_scale
        numeric_length_scales = length_scales[:len(num_cols)]

        importance = 1.0 / numeric_length_scales
        importance_dict = dict(zip(num_cols, importance))
        importance_sorted = sorted(
            importance_dict.items(),
            key=lambda x: -x[1]
        )

        ranked_features = [k for k, _ in importance_sorted]
        # Save full importance with values
        feature_importance = [
            {"feature": k, "importance": float(v)}
            for k, v in importance_sorted
        ]
        top_pairs_2d = list(combinations(ranked_features[:3], 2))

    # ---------- METRICS ----------
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    metrics = {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
    }

    # ---------- RETURN ----------
    return {
        # Core artifacts (kept for backend usage)
        "model": gpr,
        "scaler": scaler,
        "encoder": encoder,

        # Feature metadata
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_count": int(X.shape[1]),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),

        # Plot-ready predictions
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),

        "ranked_features": ranked_features,
        "top_pairs_2d": top_pairs_2d,
        "feature_importance": feature_importance,

        # Performance metrics
        "metrics": metrics,
    }