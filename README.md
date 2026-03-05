# Capstone
# ML Outlier Detection & Gaussian Process Regression Platform

## Overview
This platform is an interactive web-based tool designed to help researchers explore experimental datasets, detect outliers, and model nonlinear relationships using **Gaussian Process Regression (GPR)**.

The tool enables users to upload datasets, select relevant features, tune model hyperparameters, and visualize predictions through automatically generated plots.

It is designed to support **data-driven analysis of experimental processes**, particularly where relationships between variables may be nonlinear or uncertain.

---

## Key Features

### Data Upload & Exploration
- Upload experimental datasets via **CSV**
- Automatically detects **numeric and categorical features**
- Allows **manual or automatic feature selection**

### Feature Engineering
- Create derived features from existing columns
- Evaluate model performance with new feature combinations

### Gaussian Process Regression
Supports multiple kernels:

- **RBF (Radial Basis Function)**
- **Matern**
- **Rational Quadratic**

Hyperparameter tuning options include:

- Length scale initialization and bounds  
- Noise level and noise bounds  
- Number of optimizer restarts  
- **ARD (Automatic Relevance Determination) toggle**

---

## Model Outputs

After training the model, the platform provides:

### Model Performance Metrics
- **R²**
- **MAE**
- **RMSE**

### Feature Importance
- Calculated via **ARD length scales**

### Visualization Tools
- **1D response plots**
- **2D surface plots**
- **3D response plots**

Plots can be generated across **categorical variable combinations** to explore how different experimental conditions influence outcomes.

---

## Applications

This tool is intended for researchers and engineers who want to:

- Detect **outliers in experimental data**
- Model **nonlinear relationships between variables**
- Understand **feature importance**
- Explore **system behavior across experimental conditions**
- Generate **interpretable predictive visualizations**

Typical use cases include:

- Experimental chemistry  
- Process monitoring  
- Scientific modeling  
- Research data exploration  

---

## Installation

Clone the repository:
- git clone https://github.com/Hendricks-Laboratory/CRISIS-Autonomous-Reproducibility-Tool

Install the required dependencies:
- pip install -r requirements.txt

---

## Running the Application

Start the local server:
- python app.py
The application will run on a local development server.

Open your browser and navigate to:
- http://127.0.0.1:5000 or http://localhost:5000

## Authors

This tool was developed by Whitman College Computer Science Capstone Group:

- **Audrey Marthin**  
- **Beatrice Archer** 
- **Carl Odegard** 

Adviced by: 
- **Professor Mark Hendricks**
- **Professor William Bares**

Additional contributions and feedback provided by research collaborators.

---