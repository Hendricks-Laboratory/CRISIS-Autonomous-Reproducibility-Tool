import io
import base64
import pandas as pd
from flask import Flask, render_template, request, jsonify
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from helpers import group
from detect import detectOutliers
from gpr.model import run_gp_pipeline, apply_feature_engineering, auto_detect_features
from gpr.visualize import plot_1d, plot_2d, plot_3d

app = Flask(__name__)

GLOBAL_STATE = {
    "df": None,
    "grouped": None,
    "gpr_model": None
}

# -------------------------------
# HOME
# -------------------------------
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

# -------------------------------
# UPLOAD CSV
# -------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    df = pd.read_csv(file)
    GLOBAL_STATE["df"] = df

    return jsonify({"columns": df.columns.tolist()})

# -------------------------------
# GROUP + DETECT OUTLIERS
# -------------------------------
@app.route("/detect", methods=["POST"])
def detect():
    data = request.json
    df = GLOBAL_STATE["df"]

    grouped = group(
        df,
        control_cols=data["control_cols"],
        metadata_cols=data["metadata_cols"],
        output_cols=data["output_cols"]
    )

    grouped = detectOutliers(grouped)
    GLOBAL_STATE["grouped"] = grouped

    outliers_exist = any(
        rep["is_outlier"].any()
        for rep in grouped["replicates"]
    )

    return jsonify({
        "outliers_detected": outliers_exist
    })

# -------------------------------
# RUN GPR
# -------------------------------
@app.route("/run_gpr", methods=["POST"])
def run_gpr():
    data = request.json
    df = GLOBAL_STATE["df"].copy()

    try:
        # ---------- FEATURE ENGINEERING ----------
        expressions = data.get("expressions", [])
        if expressions:
            df = apply_feature_engineering(df, expressions)

        GLOBAL_STATE["df_trained"] = df.copy()
        mode = data.get("mode", "auto")
        target_col = data.get("target_col")

        # ---------- MANUAL MODE ----------
        if mode == "manual":
            result = run_gp_pipeline(
                df,
                target_col=target_col,
                mode="manual",
                num_cols=data.get("num_cols", []),
                cat_cols=data.get("cat_cols", []),
                kernel_config=data.get("kernel_config", None)
            )

        # ---------- AUTOMATIC MODE ----------
        else:
            result = run_gp_pipeline(
                df,
                target_col=target_col,
                mode="auto",
                num_cols=data.get("num_cols", []),
                cat_cols=data.get("cat_cols", []),
                kernel_config=data.get("kernel_config", None)
            )

        GLOBAL_STATE["gpr_model"] = result

        return jsonify({
            "r2": round(result["metrics"]["r2"], 4),
            "mae": round(result["metrics"]["mae"], 4),
            "rmse": round(result["metrics"]["rmse"], 4),
            "feature_count": result["feature_count"],
            "train_size": result["train_size"],
            "test_size": result["test_size"],
            "feature_importance": result["feature_importance"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# AUTO DETECT FEATURES
# -------------------------------
@app.route("/auto_detect_features", methods=["POST"])
def auto_detect_features_route():

    data = request.json
    target_col = data.get("target_col")

    if not target_col:
        return jsonify({"error": "Target column missing"}), 400

    df = GLOBAL_STATE["df"]

    import numpy as np

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Remove target from numeric
    if target_col in num_cols:
        num_cols.remove(target_col)

    return jsonify({
        "num_cols": num_cols,
        "cat_cols": cat_cols
    })

# -------------------------------
# APPLY FEATURE ENGINEERING
# -------------------------------
@app.route("/apply_feature", methods=["POST"])
def apply_feature():
    data = request.json
    df = GLOBAL_STATE["df"]

    try:
        df = apply_feature_engineering(df, [data["expression"]])
        GLOBAL_STATE["df"] = df  # Update global state

        return jsonify({
            "columns": df.columns.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_category_slices", methods=["POST"])
def get_category_slices():

    model_data = GLOBAL_STATE["gpr_model"]
    df = GLOBAL_STATE["df"]

    cat_cols = model_data["cat_cols"]

    if len(cat_cols) > 0:
        slices = df[cat_cols].drop_duplicates().to_dict(orient="records")
    else:
        slices = [{}]

    return jsonify({"slices": slices})

@app.route("/generate_plot", methods=["POST"])
def generate_plot():

    data = request.json

    model_data = GLOBAL_STATE.get("gpr_model")
    if not model_data:
        return jsonify({"error": "Model not found. Run GPR first."})

    gpr = model_data["model"]
    scaler = model_data["scaler"]
    encoder = model_data["encoder"]
    num_cols = model_data["num_cols"]

    df = GLOBAL_STATE["df_trained"]

    target_col = data.get("target_col")
    if not target_col:
        return jsonify({"error": "Target column missing."})

    fixed_slice = data.get("slice", {})
    plot_type = data.get("plot_type")
    color_scheme = data.get("color_scheme", "default")

    if not num_cols:
        return jsonify({"error": "No numeric features available for plotting."})

    figures = []

    try:

        # ==========================
        # 1D PLOTS (ALL NUMERIC)
        # ==========================
        if plot_type == "1d":

            for feature in num_cols:
                fig = plot_1d(
                    gpr, df, num_cols, scaler, encoder,
                    feature,
                    fixed_slice,
                    target_col,
                    color_scheme=color_scheme
                )
                figures.append(fig)

        # ==========================
        # 2D PLOTS (ALL PAIRS)
        # ==========================
        elif plot_type == "2d":

            if len(num_cols) < 2:
                return jsonify({"error": "Need at least 2 numeric features for 2D plot."})

            for fx, fy in combinations(num_cols, 2):
                fig = plot_2d(
                    gpr, df, num_cols, scaler, encoder,
                    fx, fy,
                    fixed_slice,
                    target_col,
                    color_scheme=color_scheme
                )
                figures.append(fig)

        # ==========================
        # 3D PLOTS (ALL PAIRS)
        # ==========================
        elif plot_type == "3d":

            if len(num_cols) < 2:
                return jsonify({"error": "Need at least 2 numeric features for 3D plot."})

            for fx, fy in combinations(num_cols, 2):
                fig = plot_3d(
                    gpr, df, num_cols, scaler, encoder,
                    fx, fy,
                    fixed_slice,
                    target_col,
                    color_scheme=color_scheme
                )
                figures.append(fig)

        else:
            return jsonify({"error": "Invalid plot type."})

        # ==========================
        # Convert All Figures
        # ==========================
        images = []

        for fig in figures:
            img = io.BytesIO()
            fig.savefig(img, format="png", bbox_inches="tight")
            img.seek(0)
            encoded = base64.b64encode(img.getvalue()).decode()
            images.append(encoded)
            plt.close(fig)

        return jsonify({"images": images})

    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == "__main__":
    app.run(debug=False)