#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

bin_file = 'gaussians.dat'

# Read the CSV file with labeled columns
df = pd.read_csv(bin_file, delimiter=",", dtype={'Latitude_Band': str})

# Extract individual arrays if needed
mu1 = df['mu1'].to_numpy()
sigma1 = df['sigma1'].to_numpy()
w1 = df['w1'].to_numpy()

mu2 = df['mu2'].to_numpy()
sigma2 = df['sigma2'].to_numpy()
w2 = df['w2'].to_numpy()

mu3 = df['mu3'].to_numpy()
sigma3 = df['sigma3'].to_numpy()
w3 = df['w3'].to_numpy()

labels = df['Latitude_Band'].to_numpy()

# Prepare x range and initialize list of curves
x = np.linspace(-2, 1, 1000)
dx = x[1] - x[0]
composite = np.zeros_like(x)
curves = []

for i in range(len(mu1)):
    mu = [mu1[i], mu2[i], mu3[i]]
    sigma = [sigma1[i], sigma2[i], sigma3[i]]
    weights = [w1[i], w2[i], w3[i]]

    # Normalize the weights
    weights = np.array(weights)
    weights /= np.sum(weights)

    # Compute the total mixture distribution
    y_total = np.zeros_like(x)
    for m, s, w in zip(mu, sigma, weights):
        y_total += w * norm.pdf(x, loc=m, scale=s)

    # Normalize y_total to area = 1
    y_total /= np.sum(y_total * dx)

    # Store the curve
    curves.append((labels[i], y_total))
    composite += y_total

    # Plot individual mixture
    plt.figure(figsize=(8, 4))
    for m, s, w in zip(mu, sigma, weights):
        plt.plot(x, w * norm.pdf(x, loc=m, scale=s), '--', label=f'N({m:.2f}, {s:.2f}) × {w:.2f}')
    plt.plot(x, y_total, label='Normalized Mixture', color='black')
    plt.title("Latitude band: " + labels[i])
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('row_'+str(i)+'.png')
    plt.close()

# Normalize the composite
composite /= np.sum(composite * dx)




# --- Write composite distribution out in the same two-column format ---
output_file = 'composite_' + bin_file
with open(output_file, 'w') as f:
    for xi, yi in zip(x, composite):
        # bin center and composite density
        f.write(f"{xi:.2f} {yi:.6e}\n")
print(f"Composite distribution saved to {output_file}")





# Plot composite
plt.figure(figsize=(10, 5))
for label, y in curves:
    plt.plot(x, y, '--', label=label)
plt.plot(x, composite, color='black', linewidth=2, label='Composite')
plt.title("Composite of All Normalized Mixtures")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('composite.png')
plt.close()