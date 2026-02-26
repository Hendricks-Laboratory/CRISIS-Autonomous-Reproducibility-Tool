# automation.py

from typing import Dict
import pandas as pd

from .helpers import group
from .detect import detectOutliers


def process_file(df: pd.DataFrame) -> Dict:
    """
    Groups data and runs outlier detection.
    Returns structured result for frontend.
    """

    data = group(df)
    data = detectOutliers(data, method=0, maxSDEV=1.0)

    outliers_exist = anyOutliers(data)

    return {
        "data": data,
        "outliers_detected": outliers_exist
    }


def anyOutliers(data: Dict) -> bool:
    for rep in data.get("replicates", []):
        if "is_outlier" in rep.columns and rep["is_outlier"].any():
            return True
    return False