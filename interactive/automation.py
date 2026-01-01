# automation.py

"""
Automates process to kickstart science vs engineering track
"""

import sys
from typing import Dict
import pandas as pd

from .helpers import group
from .detect import detectOutliers
from .tracks.science import runScienceTrack 
from .tracks.engineering import runEngineeringTrack
from .gpr import add_features, standardize_gpr_schema, build_features, fit_gpr, plot_gp_1d_slice_scaled

def anyOutliers(data: Dict) -> bool:
    for rep in data.get("replicates", []):
        if "is_outlier" in rep.columns and rep["is_outlier"].any():
            return True
    return False

def chooseTrack() -> str:
    """
    Ask user which track to run once outliers are detected.
    Returns 'science', 'engineering', or '' if invalid/none.
    """
    print("\n|| ADVANCING TO TRACK SELECTION ||")
    print("Available tracks:")
    print("  1: Science track")
    print("  2: Engineering track")
    choice = input("Choose track: ").strip()

    if choice.startswith("1"):
        return "science"
    elif choice.startswith("2"):
        return "engineering"
    else:
        print("Invalid choice; no track will be run.")
        return ""


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m interactive.automation <csvfile>")
        sys.exit(1)
    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    data = group(df)
    data = detectOutliers(data, method=0, maxSDEV=1.0)

    if anyOutliers(data):
        print("\nOutliers detected in at least one replicate.")

        try:
            df_gpr = standardize_gpr_schema(df)
            df_gpr = add_features(df_gpr)

            num_cols = ["rxn_concentration"]
            cat_col = ["additive"]
            target_col = "lambda max wavelength"

            X, y, scaler, encoder = build_features(df_gpr, num_cols, cat_col, target_col)
            gpr_pack = fit_gpr(X, y, n_restarts_optimizer=10, nu=1.5)
            print("Learned kernel:", gpr_pack["kernel_learned"])

            data["gpr_results"] = {
                "gpr": gpr_pack["gpr"],
                "scaler": scaler,
                "encoder": encoder,
                "df": df_gpr,
                "num_cols": num_cols,
                "cat_col": cat_col, 
                "target_col": target_col,
            }

            plot_gp_1d_slice_scaled(
                gpr=gpr_pack["gpr"],
                df=df_gpr,
                num_cols=num_cols,
                encoder=encoder,
                scaler=scaler,
                feature="rxn_concentration",
                fixed_additive="C8",
                target_col=target_col
            )

        except Exception as e:
            print(f"GPR kickstart failed: {e}")

        track = chooseTrack()
        if track == "science":
            runScienceTrack(data)
        elif track == "engineering":
            runEngineeringTrack(data)

    else:
        print("\nNo outliers detected in any replicate.")
    print("\n|| THANK YOU ||")
        
if __name__ == "__main__":
    main()
