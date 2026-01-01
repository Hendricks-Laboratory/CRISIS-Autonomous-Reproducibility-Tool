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

def controlSensitizingGraph(controlData, outputCol, generateGraph = True):
    data = controlData[outputCol].dropna()
    mean = data.mean()
    std = data.std()
    zScores = []
    problemPoints = set()

    for i, x in enumerate(data):
        z = (x - mean) / std
        zScores.append(z)

        # Rule 1
        # One Point outside of control limit of 3
        if abs(z) >= 3:
            print(f"[Rule 1] Point {i} is outside control limits: z = {z: .2f}")
            problemPoints.add(i)

        # Rule 2 
        # Two of three consecutive points outside +/- 2 on the same side
        if i >= 2:
            window = zScores[i - 2 : i + 1]
            pos = sum(v >=  2 for v in window)
            neg = sum(v <= -2 for v in window)
            if pos >= 2 or neg >= 2:
                print(f"[Rule 2] Two of three points outside +/-2 between {i - 2} and {i}")
                problemPoints.update(range(i - 2, i + 1))

        # Rule 3 
        # Four of five consecutive points beyond +/-1 on the same side
        if i >= 4:
            window = zScores[i - 4 : i + 1]
            pos = sum(v >=  1 for v in window)
            neg = sum(v <= -1 for v in window)
            if pos >= 4 or neg >= 4:
                print(f"[Rule 3] Four of five points beyond +/- 1 between {i - 4} and {i}")
                problemPoints.update(range(i - 4, i + 1))

        # Rule 4 
        # Eight consecutive points on the same side of the center line
        if i >= 7:
            window = zScores[i - 7 : i + 1]
            all_pos = all(v > 0 for v in window)
            all_neg = all(v < 0 for v in window)
            if all_pos or all_neg:
                print(f"[Rule 4] Eight points on one side of center between {i - 7} and {i}")
                problemPoints.update(range(i - 7, i + 1))

        # Rule 5 
        # Six points in a row steadily increasing or decreasing
        if i >= 5:
            w = list(data.iloc[i - 5 : i + 1])
            inc = all(w[j] < w[j + 1] for j in range(5))
            dec = all(w[j] > w[j + 1] for j in range(5))
            if inc or dec:
                print(f"[Rule 5] Six points steadily {'increasing' if inc else 'decreasing'} "
                      f"between {i - 5} and {i}")
                problemPoints.update(range(i - 5, i + 1))

        # Rule 6 
        # Fifteen points in a row within +/- 1
        if i >= 14:
            window = zScores[i - 14 : i + 1]
            if all(abs(v) < 1 for v in window):
                print(f"[Rule 6] Fifteen points within +/- 1 between {i - 14} and {i}")
                problemPoints.update(range(i - 14, i + 1))

        # Rule 7 
        # Fourteen points in a row alternating up and down
        if i >= 13:
            window = zScores[i - 13 : i + 1]
            if all(v != 0 for v in window):
                alternating = all(window[j] * window[j + 1] < 0 for j in range(13))
                if alternating:
                    print(f"[Rule 7] Fourteen points alternating up/down between {i - 13} and {i}")
                    problemPoints.update(range(i - 13, i + 1))

        # Rule 8 
        # Eight points in a row on both sides of center, none within +/- 1
        if i >= 7:
            window = zScores[i - 7 : i + 1]
            if all(abs(v) >= 1 for v in window) and any(v > 0 for v in window) and any(v < 0 for v in window):
                print(f"[Rule 8] Eight points outside +/- 1 on both sides between {i - 7} and {i}")
                problemPoints.update(range(i - 7, i + 1))

    if generateGraph:
        # Create plot
        x = range(len(zScores))
        plt.figure(figsize=(10, 5))
        plt.plot(x, zScores, marker='o')

        # Problem points
        prob = sorted(problemPoints)
        plt.scatter(prob, [zScores[i] for i in prob], color='red', s=60, zorder=10)

        for level in range(-3, 4):
            linestyle = '--' if level != 0 else '-'  
            alpha = 0.7 if level != 0 else 1
            plt.axhline(level, color='red' if abs(level) >= 2 else 'black',
                        linestyle=linestyle, linewidth=1, alpha=alpha)

        plt.title(f"Std Deviations from Mean for '{outputCol}'")
        plt.xlabel("Trial Number")
        plt.ylabel("Number of Standard Deviations")
        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.show()
        return problemPoints

    if problemPoints:
        print(f"Potential control issues at indices: {sorted(set(problemPoints))}")
    else:
        print("No control rule violations detected.")
