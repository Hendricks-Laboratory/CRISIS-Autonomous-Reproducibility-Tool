# detect.py
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detectOutliers(data, method = 0, maxSDEV = 1):
    # Standard deviation
    if method == 0:
        for replicate in data["replicates"]:
            vals = replicate[data["output"][0]]
            mean = vals.mean()
            sdev = np.std(vals, 0)
            minVal, maxVal = mean - (maxSDEV * sdev), mean + (maxSDEV * sdev)
            replicate["is_outlier"] = ~vals.between(minVal, maxVal)
    return data
