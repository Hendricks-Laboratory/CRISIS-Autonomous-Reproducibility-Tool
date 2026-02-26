# gpr/model.py

import numpy as np
from itertools import combinations
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, DotProduct
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .visualize import plot_1d, plot_2d, plot_3d

def interactive_feature_engineering(df):
    print("\n|| FEATURE ENGINEERING STEP ||")
    print("\nYou may create new columns using full expressions.")
    print("Example:")
    print("  rxn_conc = concentration * volume")
    print("  rxn_conc_log = np.log1p(rxn_conc)")
    print("Type 'done' when finished.\n")

    while True:
        expr = input("Enter expression (or 'done'): ").strip()
        if expr.lower() == "done":
            break
        if "=" not in expr:
            print("Invalid format. Use: new_column = expression")
            continue

        new_col, formula = expr.split("=", 1)
        new_col = new_col.strip()
        formula = formula.strip()

        try:
            # Safe evaluation context
            safe_dict = {
                "np": np,
                "df": df
            }
            # Allow column names directly as variables
            safe_dict.update({col: df[col] for col in df.columns})

            # Evaluate the expression safely
            df[new_col] = eval(formula, {"__builtins__": {}}, safe_dict)
            print(f"Created column: {new_col}")

        except Exception as e:
            print("Error evaluating expression:", e)

    return df

def run_gp_pipeline(df, target_col):
    if target_col not in df.columns:
        raise ValueError("Target column not found in dataframe.")

    df = interactive_feature_engineering(df)
    mode = input(
        "\nType 'a' for automatic feature detection\n"
        "Type 'm' to manually choose numeric/categorical columns\n"
        "Selection: "
    ).strip().lower()

    if mode == "m":
        print("\nEnter numeric feature columns separated by commas:")
        num_input = input("Numeric columns: ")
        num_cols = [c.strip() for c in num_input.split(",") if c.strip()]

        print("\nEnter categorical feature columns separated by commas:")
        cat_input = input("Categorical columns (or leave blank): ")
        cat_cols = [c.strip() for c in cat_input.split(",") if c.strip()]

    else:
        # Automatic detection
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)

        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        print("\nAuto-detected numeric columns:")
        for i, col in enumerate(num_cols):
            print(f"{i}: {col}")

        drop_num = input(
            "\nEnter indices of numeric columns to DROP (comma-separated) or press Enter to keep all: "
        ).strip()
        if drop_num:
            drop_indices = [int(i.strip()) for i in drop_num.split(",") if i.strip().isdigit()]
            num_cols = [col for i, col in enumerate(num_cols) if i not in drop_indices]

        print("\nAuto-detected categorical columns:")
        for i, col in enumerate(cat_cols):
            print(f"{i}: {col}")

        drop_cat = input(
            "\nEnter indices of categorical columns to DROP (comma-separated) or press Enter to keep all: "
        ).strip()
        if drop_cat:
            drop_indices = [int(i.strip()) for i in drop_cat.split(",") if i.strip().isdigit()]
            cat_cols = [col for i, col in enumerate(cat_cols) if i not in drop_indices]

    if len(num_cols) == 0 and len(cat_cols) == 0:
        raise ValueError("No feature columns selected.")

    print("\nNumeric features:", num_cols)
    print("Categorical features:", cat_cols)

    scaler = None
    encoder = None

    if len(num_cols) > 0:
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(df[num_cols])
    else:
        X_num_scaled = None

    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = encoder.fit_transform(df[cat_cols])
    else:
        X_cat = None

    if X_num_scaled is not None and X_cat is not None:
        X = np.hstack([X_num_scaled, X_cat])
    elif X_num_scaled is not None:
        X = X_num_scaled
    elif X_cat is not None:
        X = X_cat
    else:
        raise ValueError("No valid features available.")

    y = df[target_col].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0    )

    n_features = X_train.shape[1]
    kernel = (
        RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-2))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=0,
    )

    gpr.fit(X_train, y_train)
    y_mean, y_std = gpr.predict(X_test, return_std=True)

    print("\nLearned kernel:", gpr.kernel_)
    print("Test R2:", r2_score(y_test, y_mean))
    print("Test MAE:", mean_absolute_error(y_test, y_mean))

    if len(num_cols) > 0:
        length_scales = gpr.kernel_.k1.length_scale
        numeric_length_scales = length_scales[:len(num_cols)]
        importance = 1.0 / numeric_length_scales
        importance_dict = dict(zip(num_cols, importance))
        importance_sorted = dict(sorted(importance_dict.items(), key=lambda x: -x[1]))
        print("\nFeature importance (numeric only):")

        for k, v in importance_sorted.items():
            print(f"{k:25s} {v:.4f}")
        ranked_features = list(importance_sorted.keys())
    else:
        ranked_features = []

    if len(ranked_features) == 0:
        print("\nNo numeric features available for plotting.")
        return

    top_k_1d = min(3, len(ranked_features))
    top_features_1d = ranked_features[:top_k_1d]

    pair_candidates = list(combinations(top_features_1d, 2))
    top_pairs_2d = pair_candidates[:min(3, len(pair_candidates))]

    print("\nTop 1D features:", top_features_1d)
    print("Top 2D pairs:", top_pairs_2d)

    plot_choice = input("\nDo you want to generate plots? (y/n): ").strip().lower()
    if plot_choice == "y":
        color_scheme = input("\nChoose color scheme: default / colorblind / high_contrast: ").strip().lower()
        if color_scheme not in ["default", "colorblind", "high_contrast"]:
            color_scheme = "default"

        if len(cat_cols) > 0:
            unique_combinations = df[cat_cols].drop_duplicates().to_dict(orient="records")
        else:
            unique_combinations = [{}]

        for fixed_cat_dict in unique_combinations:
            print("\nCategory slice:", fixed_cat_dict)

            plot_1d_toggle = input("Plot 1D features? (y/n): ").strip().lower()
            if plot_1d_toggle == "y":
                for feature in top_features_1d:
                    plot_1d(gpr, df, num_cols, scaler, encoder,
                            feature, fixed_cat_dict, target_col, color_scheme=color_scheme)

            plot_2d_toggle = input("Plot 2D features? (y/n): ").strip().lower()
            if plot_2d_toggle == "y":
                for fx, fy in top_pairs_2d:
                    plot_2d(gpr, df, num_cols, scaler, encoder,
                            fx, fy, fixed_cat_dict, target_col, color_scheme=color_scheme)

            plot_3d_toggle = input("Plot 3D features? (y/n): ").strip().lower()
            if plot_3d_toggle == "y":
                for fx, fy in top_pairs_2d:
                    plot_3d(gpr, df, num_cols, scaler, encoder,
                            fx, fy, fixed_cat_dict, target_col, color_scheme=color_scheme)
    else:
        print("\nSkipping plotting as per user choice.")