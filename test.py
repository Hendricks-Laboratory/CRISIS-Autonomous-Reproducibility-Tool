from base_functions import *


data = pd.read_csv("output.csv")
data = group(data)
data = detectOutliers(data)
print(data["replicates"][0])

controlData = pd.read_csv("control_only.csv")
controlSensitizing(controlData, "wavelength")