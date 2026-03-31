import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")

import pandas as pd
from flask import Flask, render_template, request, jsonify

from backend.supporting_functions import (
    load_dataset,
    auto_detect_features,
    apply_feature_engineering,
    run_gp_pipeline,
    generate_plot,
    generate_plot_both,
)

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
    data = request.json
    mode = data.get("mode", "std")
    dimension = data.get("plot_type", "1d")
    xVar = data.get("x_var")
    yVar = data.get("y_var")
    logVars = data.get("log_vars", [])
    color_scheme = data.get("color_scheme", "default")

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
                color_scheme=color_scheme
            )
            return jsonify({"pairs": pairs, "mode": "both"})

        elif mode == "std":
            model_data = GLOBAL_STATE.get("std_model")
            if not model_data:
                return jsonify({"error": "Std model not found. Run GPR first."}), 400
            yVar = yVar or model_data["gp_target"]

        else:
            model_data = GLOBAL_STATE.get("mean_model")
            if not model_data:
                return jsonify({"error": "Mean model not found. Run GPR first."}), 400
            yVar = yVar or model_data["gp_target"]

        images = generate_plot(
            gp_data=model_data["gp_data"],
            gp=model_data["model"],
            control_vars=model_data["control_vars"],
            dimension=dimension,
            xVar=xVar,
            yVar=yVar,
            logVars=logVars,
            category_combos=model_data["category_combos"],
            color_scheme=color_scheme
        )
        return jsonify({"images": images, "mode": mode})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
