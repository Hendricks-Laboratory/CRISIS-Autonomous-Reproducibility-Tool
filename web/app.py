# =============================================================================
# web/app.py
#
# Flask application entry point for the GPR Research Platform.
# Serves the multi-page UI and exposes the following API routes:
#
#   GET  /                      — home page
#   GET  /tool                  — analysis tool page
#   GET  /methodology           — methodology page
#   GET  /creators              — creators page
#   POST /upload                — upload CSV, returns column list
#   POST /auto_detect_features  — classify columns as numeric/categorical
#   POST /apply_feature         — add a derived column via expression
#   POST /run_gpr               — fit GP model(s) for std | mean | both mode;
#                                 returns metrics and category combo metadata
#   POST /generate_plot         — render 1D or 2D GP plots for the stored
#                                 model(s); returns base64 PNGs
#   GET  /load_example          — load a bundled example dataset into GLOBAL_STATE;
#                                 type=1d uses 1d_data.csv,
#                                 type=2d uses 2d_data.csv
#
# GLOBAL_STATE holds the active DataFrame and trained model results for the
# duration of the session (single-user, in-memory). Models are keyed as
# "std_model" and "mean_model" and cleared on each new CSV upload.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
from flask import Flask, render_template, request, jsonify
from backend.supporting_functions import (load_dataset, auto_detect_features, apply_feature_engineering, run_gp_pipeline, generate_plot, generate_plot_both)

app = Flask(__name__)

GLOBAL_STATE = {
    "df": None,
    "std_model": None,
    "mean_model": None,
}

# -------------------------------------------------------
# PAGES
# -------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/tool")
def tool():
    return render_template("tool.html")

@app.route("/methodology")
def methodology():
    return render_template("methodology.html")

@app.route("/creators")
def creators():
    return render_template("creators.html")

# -------------------------------------------------------
# UPLOAD
# -------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    """Accept a CSV upload, store the DataFrame in GLOBAL_STATE, return column names."""
    file = request.files["file"]
    df = load_dataset(file)
    GLOBAL_STATE["df"] = df
    GLOBAL_STATE["std_model"] = None
    GLOBAL_STATE["mean_model"] = None
    return jsonify({"columns": df.columns.tolist()})

# -------------------------------------------------------
# AUTO DETECT FEATURES
# -------------------------------------------------------
@app.route("/auto_detect_features", methods=["POST"])
def auto_detect_features_route():
    """Classify dataset columns as numerical or categorical, excluding the given target."""
    data = request.json
    target_col = data.get("target_col", "")
    df = GLOBAL_STATE["df"]
    result = auto_detect_features(df, target_col)
    return jsonify(result)

# -------------------------------------------------------
# APPLY FEATURE ENGINEERING
# -------------------------------------------------------
@app.route("/apply_feature", methods=["POST"])
def apply_feature():
    """Evaluate a feature engineering expression and update the stored DataFrame."""
    data = request.json
    df = GLOBAL_STATE["df"]
    try:
        df = apply_feature_engineering(df, [data["expression"]])
        GLOBAL_STATE["df"] = df
        return jsonify({"columns": df.columns.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------
# RUN GPR  (mode = std | mean | both)
# -------------------------------------------------------
@app.route("/run_gpr", methods=["POST"])
def run_gpr():
    """
    Fit GP model(s) and store results in GLOBAL_STATE.

    Accepts JSON body with mode = "std" | "mean" | "both".
    In "std" mode, expects std_num_cols, std_cat_cols, measurement_col,
    std_gp_target ("mean" or "std"), std_log_vars, std_noise, and
    std_kernel_config. In "mean" mode, expects the mean_* equivalents
    plus mean_output_col. "both" runs both pipelines in sequence.

    Returns:
        JSON with a key per mode run ("std", "mean"), each containing
        r2, mae, rmse, control_vars, category_combos (label only), and
        gp_target. Also includes "mode" at the top level.
    """
    data = request.json
    df = GLOBAL_STATE["df"].copy()
    mode = data.get("mode", "std")

    try:
        response = {}

        if mode in ("std", "both"):
            std_result = run_gp_pipeline(
                df,
                num_cols=data.get("std_num_cols", []),
                cat_cols=data.get("std_cat_cols", []),
                measurement_col=data.get("measurement_col"),
                gp_target=data.get("std_gp_target", "mean"),
                mode="std",
                logVars=data.get("std_log_vars", []),
                noise=data.get("std_noise", "std"),
                kernel_config=data.get("std_kernel_config", None),
            )
            GLOBAL_STATE["std_model"] = std_result
            response["std"] = {
                "r2": round(std_result["metrics"]["r2"], 4),
                "mae": round(std_result["metrics"]["mae"], 4),
                "rmse": round(std_result["metrics"]["rmse"], 4),
                "control_vars": std_result["control_vars"],
                "category_combos": [{"label": c["label"]} for c in std_result["category_combos"]],
                "gp_target": std_result["gp_target"],
            }

        if mode in ("mean", "both"):
            mean_result = run_gp_pipeline(
                df,
                num_cols=data.get("mean_num_cols", []),
                cat_cols=data.get("mean_cat_cols", []),
                output_col=data.get("mean_output_col"),
                mode="mean",
                logVars=data.get("mean_log_vars", []),
                noise="constant",
                kernel_config=data.get("mean_kernel_config", None),
            )
            GLOBAL_STATE["mean_model"] = mean_result
            response["mean"] = {
                "r2": round(mean_result["metrics"]["r2"], 4),
                "mae": round(mean_result["metrics"]["mae"], 4),
                "rmse": round(mean_result["metrics"]["rmse"], 4),
                "control_vars": mean_result["control_vars"],
                "category_combos": [{"label": c["label"]} for c in mean_result["category_combos"]],
                "gp_target": mean_result["gp_target"],
            }

        response["mode"] = mode
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------
# GENERATE PLOT
# -------------------------------------------------------
@app.route("/generate_plot", methods=["POST"])
def generate_plot_route():
    """
    Render GP plots for the stored model(s) and return base64 PNG images.

    Accepts JSON body with:
        mode (str): "std" | "mean" | "both"
        plot_type (str): "1d" | "2d"
        x_var (str): Column for the x-axis.
        y_var (str | None): 2D y-axis column. In 1D, falls back to gp_target when None.
        log_vars (list[str]): Columns to display on log scale.
        color_scheme (str): "default" | "colorblind" | "high_contrast"
        unscale_std (bool): Whether to unscale the uncertainty (right) panels.

    Unscaling logic applied per mode:
        std, gp_target="mean": y_scaler and ylabel are both set to the measurement
            column automatically (replicate mean is always shown in original units).
        std, gp_target="std":  y_scaler is set only when unscale_std=True.
        mean: y-axis and 2D left panel always unscaled; uncertainty panel
            unscaled only when unscale_std=True.
        both: delegates entirely to generate_plot_both with the same unscale_std flag.

    Returns:
        JSON with {"images": [...], "mode", "plot_type", "gp_target"} for std/mean,
        or {"pairs": [...], "mode": "both", "plot_type"} for both mode.
    """
    data = request.json
    mode = data.get("mode", "std")
    dimension = data.get("plot_type", "1d")
    xVar = data.get("x_var")
    yVar = data.get("y_var")
    logVars = data.get("log_vars", [])
    color_scheme = data.get("color_scheme", "default")
    unscale_std = data.get("unscale_std", False)

    try:
        if mode == "both":
            std_result = GLOBAL_STATE.get("std_model")
            mean_result = GLOBAL_STATE.get("mean_model")
            if not std_result or not mean_result:
                return jsonify({"error": "Both models required. Run GPR first."}), 400

            pairs = generate_plot_both(
                std_result, mean_result,
                dimension=dimension,
                xVar=xVar,
                logVars=logVars,
                color_scheme=color_scheme,
                unscale_std=unscale_std
            )
            return jsonify({"pairs": pairs, "mode": "both", "plot_type": dimension})

        elif mode == "std":
            model_data = GLOBAL_STATE.get("std_model")
            if not model_data:
                return jsonify({"error": "Std model not found. Run GPR first."}), 400
            yVar = yVar or model_data["gp_target"]
            num_scalers = model_data.get("num_scalers", {})
            measurement_col = model_data.get("measurement_col")
            measurement_scaler = num_scalers.get(measurement_col)
            z_scaler_mean = measurement_scaler   # always unscale left panel
            z_scaler_unc = measurement_scaler if unscale_std else None
            zLabel = model_data["gp_target"]           # column lookup (mean or std)
            zLabel_display = measurement_col or model_data["gp_target"]  # colorbar text
            if model_data["gp_target"] == "mean" and measurement_col:
                y_scaler = measurement_scaler  # auto-unscale 1D replicate mean
                ylabel = measurement_col
            else:
                y_scaler = measurement_scaler if unscale_std else None
                ylabel = None

        else:
            model_data = GLOBAL_STATE.get("mean_model")
            if not model_data:
                return jsonify({"error": "Mean model not found. Run GPR first."}), 400
            yVar = yVar or model_data["gp_target"]
            num_scalers = model_data.get("num_scalers", {})
            output_col = model_data["gp_target"]
            output_scaler = num_scalers.get(output_col)
            y_scaler = output_scaler           # 1D y-axis unscale
            z_scaler_mean = output_scaler      # 2D left panel — always unscale
            z_scaler_unc = output_scaler if unscale_std else None
            zLabel = output_col
            zLabel_display = None
            ylabel = None

        images = generate_plot(
            gp_data=model_data["gp_data"],
            gp=model_data["model"],
            control_vars=model_data["control_vars"],
            dimension=dimension,
            xVar=xVar,
            yVar=yVar,
            logVars=logVars,
            category_combos=model_data["category_combos"],
            color_scheme=color_scheme,
            num_scalers=num_scalers,
            y_scaler=y_scaler,
            z_scaler_mean=z_scaler_mean,
            z_scaler_unc=z_scaler_unc,
            gp_target=model_data["gp_target"],
            zLabel=zLabel,
            zLabel_display=zLabel_display,
            ylabel=ylabel
        )
        return jsonify({"images": images, "mode": mode, "plot_type": dimension, "gp_target": model_data["gp_target"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------
# LOAD EXAMPLE
# -------------------------------------------------------
@app.route("/load_example", methods=["GET"])
def load_example():
    """
    Load a bundled example dataset into GLOBAL_STATE.

    Query params:
        type (str): "1d" (1d_data.csv) or "2d" (2d_data.csv).
            Defaults to "1d".

    The 1D example uses mean mode on nanocrystalline materials data.
    The 2D example uses both mode (std + mean) on nanocrystalline materials data.
    Feature engineering (e.g. rxn_concentration) is applied client-side via
    /apply_feature after this endpoint returns the base column list.

    Returns:
        JSON {"columns": [...]} mirroring the /upload response.
    """
    example_type = request.args.get("type", "1d")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    filename = "2d_data.csv" if example_type == "2d" else "1d_data.csv"
    path = os.path.join(project_root, "example_data", filename)

    try:
        df = load_dataset(path)
        GLOBAL_STATE["df"] = df
        GLOBAL_STATE["std_model"] = None
        GLOBAL_STATE["mean_model"] = None
        return jsonify({"columns": df.columns.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
