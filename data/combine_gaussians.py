#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# === INPUT FILES ===
apo_bin_file = 'gaussians.dat'
bdbs_file = 'binned_dist_lat6_0.08dex.dat'  # Make sure this exists

# === Load APOGEE Gaussian MDF fits ===
df = pd.read_csv(apo_bin_file, delimiter=",", dtype={'Latitude_Band': str})
mu1, sigma1, w1 = df['mu1'].to_numpy(), df['sigma1'].to_numpy(), df['w1'].to_numpy()
mu2, sigma2, w2 = df['mu2'].to_numpy(), df['sigma2'].to_numpy(), df['w2'].to_numpy()
mu3, sigma3, w3 = df['mu3'].to_numpy(), df['sigma3'].to_numpy(), df['w3'].to_numpy()
labels = df['Latitude_Band'].to_numpy()

# === Define x-axis and normalization ===
x = np.linspace(-2, 1, 1000)
dx = x[1] - x[0]
composite = np.zeros_like(x)
curves = []

# === APOGEE latitude weights ===
weight_table = {
    "-9.5": 0.011157, "-8.5": 0.017961, "-7.5": 0.028916, "-6.5": 0.046553,
    "-5.5": 0.074948, "-4.5": 0.120660, "-3.5": 0.194255, "-2.5": 0.312736,
    "-1.5": 0.503483, "-0.5": 0.810573, "0.5": 0.810573, "1.5": 0.503483,
    "2.5": 0.312736, "3.5": 0.194255, "4.5": 0.120660, "5.5": 0.074948,
    "6.5": 0.046553, "7.5": 0.028916, "8.5": 0.017961, "9.5": 0.011157
}
norm_factor = weight_table["-0.5"]
for k in weight_table:
    weight_table[k] /= norm_factor

# === Label to center mapping (actual bins used) ===
label_to_bcenter = {
    "6<=|b|<=10": "8.5",
    "4<=|b|<=6": "5.5",
    "2.5<=|b|<=4": "3.5",
    "1.7<=|b|<=2.5": "2.5",
    "|b|<=1.7": "0.5"
}

# === Build APOGEE composite ===
for i in range(len(mu1)):
    mu = [mu1[i], mu2[i], mu3[i]]
    sigma = [sigma1[i], sigma2[i], sigma3[i]]
    weights = np.array([w1[i], w2[i], w3[i]])
    weights /= np.sum(weights)

    y_total = sum(w * norm.pdf(x, loc=m, scale=s) for m, s, w in zip(mu, sigma, weights))
    y_total /= np.sum(y_total * dx)
    curves.append((labels[i], y_total))

    band_label = labels[i]
    b_center_str = label_to_bcenter.get(band_label, None)
    if b_center_str is None:
        print(f"Skipping unmatched band label: {band_label}")
        continue
    lat_weight = weight_table[b_center_str]
    composite += lat_weight * y_total

# Final normalize
composite /= np.sum(composite * dx)

# Save APOGEE composite
with open("composite_" + apo_bin_file, "w") as f:
    for xi, yi in zip(x, composite):
        f.write(f"{xi:.2f} {yi:.6e}\n")

# === Load BDBS MDF ===
bdbs_x, bdbs_y = np.loadtxt(bdbs_file, unpack=True)

# Interpolate BDBS to same grid
bdbs_interp = np.interp(x, bdbs_x, bdbs_y, left=0, right=0)
bdbs_interp /= np.sum(bdbs_interp * dx)

# === Combine into BEST MDF ===
best_y = 0.5 * composite + 0.5 * bdbs_interp
best_y /= np.sum(best_y * dx)

# Save BEST MDF
np.savetxt("best_mdf.dat", np.column_stack((x, best_y)), fmt="%.3f %.6e")

# === Plot all ===
plt.figure(figsize=(10, 6))
for label, y in curves:
    plt.plot(x, y, '--', label=label)
plt.plot(x, composite, color='black', linewidth=2, label="APOGEE Composite")
plt.plot(x, bdbs_interp, color='blue', linestyle='-', linewidth=1.5, label="BDBS MDF")
plt.plot(x, best_y, color='red', linestyle='-', linewidth=2.5, label="BEST MDF")
plt.xlabel("[Fe/H]")
plt.ylabel("Probability Density")
plt.title("MDF Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("best_mdf.png")
plt.close()
