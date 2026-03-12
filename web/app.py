from flask import Flask, render_template, request, jsonify
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt

from backend.functions import group, fit_gp, plot_gp, plot_gp_2d

app = Flask(__name__)

STATE = {
    "df": None,
    "grouped": None,
    "gp": None
}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/tool")
def tool():
    return render_template("tool.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]

    df = pd.read_csv(file)

    STATE["df"] = df

    return jsonify({
        "columns": df.columns.tolist()
    })


@app.route("/group", methods=["POST"])
def group_route():

    data = request.json

    df = STATE["df"]

    grouped = group(
        data=df,
        controlVars=data["control_vars"],
        outputVar=data["output_var"],
        blockVar=data["block_var"],
        outlierSD=data["outlier_sd"]
    )

    STATE["grouped"] = grouped

    blocks = list(grouped.keys())

    return jsonify({
        "blocks": blocks
    })


@app.route("/run_gp", methods=["POST"])
def run_gp():

    data = request.json

    block = data["block"]

    block_df = STATE["grouped"][block]

    gp = fit_gp(
        block_df,
        controlVars=data["control_vars"],
        logVars=data["log_vars"]
    )

    STATE["gp"] = gp
    STATE["block_df"] = block_df
    STATE["control_vars"] = data["control_vars"]

    return jsonify({"status": "trained"})


@app.route("/generate_plot", methods=["POST"])
def generate_plot():

    data = request.json

    gp = STATE["gp"]
    df = STATE["block_df"]
    controlVars = STATE["control_vars"]

    plt.figure()

    if data["dimension"] == "1d":

        plot_gp(
            df,
            gp,
            controlVars,
            xVar=data["xVar"]
        )

    else:

        plot_gp_2d(
            df,
            gp,
            controlVars,
            xVar=data["xVar"],
            yVar=data["yVar"]
        )

    img = io.BytesIO()

    plt.savefig(img, format="png")

    img.seek(0)

    encoded = base64.b64encode(img.getvalue()).decode()

    return jsonify({
        "image": encoded
    })


if __name__ == "__main__":
    app.run(debug=True)