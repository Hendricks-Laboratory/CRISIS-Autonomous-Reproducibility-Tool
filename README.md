# GPR Research Platform
### Whitman College Computer Science Capstone

## Overview

An interactive web-based tool for fitting **Gaussian Process Regression (GPR)** models to experimental data. Researchers can upload a dataset, engineer features, configure a kernel, train a GP model, and generate 1D and 2D visualizations — all through a guided four-step interface.

The platform is designed for experimental science workflows where measurements may be replicated across conditions and relationships between variables are nonlinear or uncertain.

---

## Installation

Clone the repository:
```bash
git clone https://github.com/Hendricks-Laboratory/CRISIS-Autonomous-Reproducibility-Tool
cd outlier_tool
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python web/app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Key Features

### Four-Step Guided Workflow
1. **Upload** — load a CSV dataset or use a built-in example
2. **Feature Engineering** — create derived columns from existing ones (e.g. `rxn_concentration = concentration * volume`)
3. **Analysis Setup** — select analysis mode, classify columns as numeric or categorical, choose log-scale variables and GP target
4. **Kernel Building & Training** — configure the kernel, train the GP, and generate plots

### Three Analysis Modes
- **Experimental Uncertainty (Std mode)** — groups replicate rows by condition, summarizes each group into mean/std, and fits a GP on either the replicate mean or the standard deviation
- **Experimental Data (Mean mode)** — fits the GP directly on each row without grouping; suited for datasets without replicates
- **Both** — runs both pipelines simultaneously for side-by-side comparison

### Kernel Options
- **RBF** — infinitely smooth; good baseline
- **Matérn** (ν = 1/2, 3/2, or 5/2) — controls smoothness; preferred for physical data
- **Rational Quadratic** — captures multi-scale variation

Each kernel can be wrapped in a **ConstantKernel** (scales output variance) or combined with a **WhiteKernel** (explicit noise term). All kernels use **ARD** (Automatic Relevance Determination) — one length scale per input dimension.

### Hyperparameter Tuning
- Length scale initialization, lower and upper bounds
- Constant kernel value and bounds
- White noise level and bounds
- Matérn ν and Rational Quadratic α parameters
- 5 optimizer restarts to avoid local optima

### Feature Engineering
- Add derived columns via Python-style expressions (`new_col = col1 * col2`)
- Expressions support NumPy functions via `np`
- New columns are immediately available for use in the analysis setup

### Categorical Variable Support
- Categorical columns are one-hot encoded before fitting
- Plots are generated for every unique category combination, plus an **All** plot

### Unscaling
- Axis values are inverse-transformed to original units for display
- Left panel (predicted mean) is always unscaled; uncertainty panel can be toggled

### Visualization
- **1D plots** — GP mean ± 95% confidence interval vs. a chosen input variable
- **2D plots** — contour maps of predicted mean and uncertainty over two input variables
- Three color schemes: **Default**, **Colorblind-safe**, **High Contrast**
- All plots are downloadable as PNG

### Example Mode
Two pre-loaded example datasets demonstrate the full workflow:
- **1D Example** — spectroscopy data in Mean mode; includes a feature engineering step showing `rxn_concentration = concentration * volume`
- **2D Example** — nanoparticle synthesis data in Both mode; demonstrates side-by-side Std and Mean GP contour plots

---

## Model Outputs

After training, the platform reports:

| Metric | Description |
|--------|-------------|
| **R²** | Coefficient of determination on the training set |
| **MAE** | Mean absolute error (standardized units) |
| **RMSE** | Root mean squared error (standardized units) |

---

## Authors

Built by the Whitman College CRISIS Capstone Team — Software Engineering & Machine Learning:

- **Audrey Marthin** — Whitman College
- **Carl Odegard** — Whitman College
- **Beatrice Archer** — Whitman College

## Acknowledgements

The following researchers originally developed the CRISIS methodology this platform is built upon:

- **Gabe Gomes** — Carnegie Mellon University
- **Mark Hendricks** — Whitman College
- **Martin Seifrid** — NC State University
- **Jessica Sampson** — University of Delaware
