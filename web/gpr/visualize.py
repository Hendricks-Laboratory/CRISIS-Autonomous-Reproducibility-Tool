# gpr/visualize.py
# on colorblind option --> legend color dont match heatmap color
# make options for how the legends are showed on 2d --> have the atual points layer ornot as option because scale might be issue

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def get_color_palette(color_scheme="default"):
    if color_scheme == "colorblind":
        palette_1d = ["#E69F00", "#56B4E9", "#009E73", "#F0E442"]
        cmap_2d = plt.cm.Set2
    elif color_scheme == "high_contrast":
        palette_1d = ["black", "red", "blue", "green"]
        cmap_2d = plt.cm.binary
    else:  # default
        palette_1d = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        cmap_2d = plt.cm.viridis
    return palette_1d, cmap_2d

def plot_1d(gpr, df, num_cols, scaler, encoder, feature, fixed_cat_dict, target_col,
            grid_size=200, color_scheme="default"):

    palette_1d, _ = get_color_palette(color_scheme)
    color = palette_1d[0]

    f_idx = num_cols.index(feature)
    x_vals = np.linspace(df[feature].min(), df[feature].max(), grid_size)

    mean_vals = df[num_cols].mean().to_numpy()
    X_num = np.tile(mean_vals, (grid_size, 1))
    X_num[:, f_idx] = x_vals

    X_num_scaled = scaler.transform(X_num)

    if encoder is not None:
        cat_df = pd.DataFrame([fixed_cat_dict])
        add_vec = encoder.transform(cat_df)[0]
        X_cat_block = np.tile(add_vec, (grid_size, 1))
        X_gp = np.hstack([X_num_scaled, X_cat_block])
    else:
        X_gp = X_num_scaled

    mean, std = gpr.predict(X_gp, return_std=True)

    mask = np.ones(len(df), dtype=bool)
    for k, v in fixed_cat_dict.items():
        mask &= (df[k] == v)

    plt.figure(figsize=(7,5))
    plt.scatter(df.loc[mask, feature], df.loc[mask, target_col], s=20, color="black")
    plt.plot(x_vals, mean, color=color)
    plt.fill_between(x_vals, mean - 1.96*std, mean + 1.96*std, alpha=0.3, color=color)
    plt.xlabel(feature)
    plt.ylabel(target_col)
    plt.title(f"GP for {feature} | {fixed_cat_dict}")
    return plt.gcf()

def plot_2d(gpr, df, num_cols, scaler, encoder, feature_x, feature_y, fixed_cat_dict,
            target_col, grid_size=60, color_scheme="default"):

    _, cmap_2d = get_color_palette(color_scheme)

    ix = num_cols.index(feature_x)
    iy = num_cols.index(feature_y)

    x_vals = np.linspace(df[feature_x].min(), df[feature_x].max(), grid_size)
    y_vals = np.linspace(df[feature_y].min(), df[feature_y].max(), grid_size)
    Xg, Yg = np.meshgrid(x_vals, y_vals)

    mean_vals = df[num_cols].mean().to_numpy()
    X_num = np.tile(mean_vals, (grid_size*grid_size, 1))
    X_num[:, ix] = Xg.ravel()
    X_num[:, iy] = Yg.ravel()

    X_num_scaled = scaler.transform(X_num)

    if encoder is not None:
        cat_df = pd.DataFrame([fixed_cat_dict])
        add_vec = encoder.transform(cat_df)[0]
        X_cat_block = np.tile(add_vec, (grid_size*grid_size, 1))
        X_gp = np.hstack([X_num_scaled, X_cat_block])
    else:
        X_gp = X_num_scaled

    mean_pred, _ = gpr.predict(X_gp, return_std=True)
    Z = mean_pred.reshape(grid_size, grid_size)

    mask = np.ones(len(df), dtype=bool)
    for k, v in fixed_cat_dict.items():
        mask &= (df[k] == v)

    plt.figure(figsize=(8,6))
    plt.contourf(Xg, Yg, Z, levels=30, cmap=cmap_2d)
    plt.scatter(df.loc[mask, feature_x], df.loc[mask, feature_y], c=df.loc[mask, target_col], edgecolor="k")
    plt.colorbar(label="Predicted " + target_col)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f"GP mean: {feature_x} vs {feature_y} | {fixed_cat_dict}")
    return plt.gcf()

def plot_3d(gpr, df, num_cols, scaler, encoder, feature_x, feature_y, fixed_cat_dict, 
            target_col, grid_size=60, color_scheme="default"):

    _, cmap_2d = get_color_palette(color_scheme)

    ix = num_cols.index(feature_x)
    iy = num_cols.index(feature_y)

    x_vals = np.linspace(df[feature_x].min(), df[feature_x].max(), grid_size)
    y_vals = np.linspace(df[feature_y].min(), df[feature_y].max(), grid_size)
    Xg, Yg = np.meshgrid(x_vals, y_vals)

    mean_vals = df[num_cols].mean().to_numpy()
    X_num = np.tile(mean_vals, (grid_size*grid_size, 1))
    X_num[:, ix] = Xg.ravel()
    X_num[:, iy] = Yg.ravel()

    X_num_scaled = scaler.transform(X_num)

    if encoder is not None:
        cat_df = pd.DataFrame([fixed_cat_dict])
        add_vec = encoder.transform(cat_df)[0]
        X_cat_block = np.tile(add_vec, (grid_size*grid_size, 1))
        X_gp = np.hstack([X_num_scaled, X_cat_block])
    else:
        X_gp = X_num_scaled

    mean_pred, _ = gpr.predict(X_gp, return_std=True)
    Z = mean_pred.reshape(grid_size, grid_size)

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xg, Yg, Z, cmap=cmap_2d, alpha=0.85)

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_zlabel(target_col)
    ax.set_title(f"3D GP Surface: {feature_x} vs {feature_y}")
    return plt.gcf()