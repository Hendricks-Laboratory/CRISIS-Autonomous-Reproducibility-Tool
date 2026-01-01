# helpers.py
import pandas as pd

def group(data):
    print("\nColumns:")
    for i, col in enumerate(data.columns):
        print(f"{i}: {col}")
    controlVars = [data.columns[int(i.strip())] # Stores columns for control data
               for i in input("\nControl columns (e.g. 0,1,8): ").split(",") 
               if i.strip().isdigit()]
    metaDataVars = [data.columns[int(i.strip())] # Stores columns for meta data
               for i in input("Meta data columns (e.g. 9,6,7): ").split(",") 
               if i.strip().isdigit()]
    outputVars = [data.columns[int(i.strip())] # Stores columns for output data
               for i in input("Output column (e.g. 2): ").split(",") 
               if i.strip().isdigit()]
    replicates = [ # Only group based on control columns:
        repDf.copy()
        for _, repDf in data.groupby(controlVars)
    ]
    data = ({
        "controls": controlVars,
        "metadata": metaDataVars,
        "output": outputVars,
        "replicates": replicates,})
    return data