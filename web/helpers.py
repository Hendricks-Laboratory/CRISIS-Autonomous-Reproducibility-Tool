# helpers.py

import pandas as pd
from typing import Dict, List


def group(
    data: pd.DataFrame,
    control_cols: List[str],
    metadata_cols: List[str],
    output_cols: List[str],
) -> Dict:
    """
    Groups dataframe into replicates based on control columns.
    """

    if not control_cols:
        raise ValueError("Control columns must be provided.")

    if not output_cols:
        raise ValueError("At least one output column must be provided.")

    replicates = [
        rep_df.copy()
        for _, rep_df in data.groupby(control_cols)
    ]

    return {
        "controls": control_cols,
        "metadata": metadata_cols or [],
        "output": output_cols,
        "replicates": replicates,
    }