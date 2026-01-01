# gpr/visualize.py
import numpy as np
import matplotlib.pyplot as plt

def plot_gp_1d_slice_scaled(
    gpr,
    df,
    num_cols,
    encoder,
    scaler,
    feature,
    fixed_additive,
    target_col="lambda max wavelength",
    grid_size=200
):

    f_idx = num_cols.index(feature)
    x_vals = np.linspace(df[feature].min(), df[feature].max(), grid_size)
    mean_vals = df[num_cols].mean().to_numpy()
    X_full = np.zeros((grid_size, len(num_cols) + len(encoder.categories_[0])))

    # fill all numeric values with mean, then overwrite one column
    for i in range(len(num_cols)):
        X_full[:, i] = mean_vals[i]

    X_full[:, f_idx] = x_vals

    X_num_scaled = scaler.transform(X_full[:, :len(num_cols)])
    add_vec = encoder.transform([[fixed_additive]])[0]
    X_gp = np.hstack([X_num_scaled, np.tile(add_vec, (grid_size, 1))])
    mean, std = gpr.predict(X_gp, return_std=True)
    mask = df["additive"] == fixed_additive

    plt.figure(figsize=(7,5))
    plt.scatter(
        df.loc[mask, feature],
        df.loc[mask, target_col],
        s=18,
        color="black",
        label="Observations"
    )
    plt.plot(x_vals, mean, color="blue", label="Mean prediction")
    plt.fill_between(
        x_vals,
        mean - 1.96 * std,
        mean + 1.96 * std,
        alpha=0.3,
        color="skyblue",
        label="95% CI"
    )

    plt.xlabel(feature)
    plt.ylabel(target_col)
    plt.title(f"GP for {feature} (additive={fixed_additive})")
    plt.legend()
    plt.show()
