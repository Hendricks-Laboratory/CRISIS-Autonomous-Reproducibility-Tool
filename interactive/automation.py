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
from .gpr.model import run_gp_pipeline

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
            print("\n|| STARTING GPR PROCESS ||")
            target_col = data["output"][0]
            run_gp_pipeline(df, target_col)

        except Exception as e:
            print(f"GPR process failed: {e}")

        track = chooseTrack()
        if track == "science":
            runScienceTrack(data)
        elif track == "engineering":
            runEngineeringTrack(data)

    else:
        print("\nNo outliers detected in any replicate.")
    print("\n|| THANK YOU FOR USING OUR TOOL ||")
        
if __name__ == "__main__":
    main()
