#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from scipy.interpolate import interp1d

def create_corrected_mdf_combination(
    apogee_file='gaussians.dat',
    bdbs_file='binned_dist_lat6_0.08dex.dat',
    feh_range=(-2.0, 1.0),
    dx=0.08,
    apogee_weight=0.5,
    bdbs_weight=0.5,
    output_file='corrected_mdf.dat',
    plot_comparison=True
):
    """
    Fixed MDF combination using the working approach from your original code,
    but with cleaner implementation and better diagnostics.
    """
    
    print("Creating corrected MDF combination...")
    print("=====================================")
    
    # Create x-axis
    x = np.linspace(feh_range[0], feh_range[1], int((feh_range[1] - feh_range[0])/dx) + 1)
    dx_actual = x[1] - x[0]
    print(f"Grid: {len(x)} points, dx = {dx_actual:.4f}")
    
    # ===============================
    # APOGEE COMPONENT (FIXED)
    # ===============================
    
    print("\n1. Processing APOGEE data...")
    
    # Load APOGEE data
    df = pd.read_csv(apogee_file, delimiter=",", dtype={'Latitude_Band': str})
    mu1, sigma1, w1 = df['mu1'].values, df['sigma1'].values, df['w1'].values
    mu2, sigma2, w2 = df['mu2'].values, df['sigma2'].values, df['w2'].values
    mu3, sigma3, w3 = df['mu3'].values, df['sigma3'].values, df['w3'].values
    labels = df['Latitude_Band'].values
    
    print(f"  Loaded {len(labels)} APOGEE latitude bands")
    
    # ORIGINAL latitude weights from your working code
    weight_table = {
        "-9.5": 0.011157, "-8.5": 0.017961, "-7.5": 0.028916, "-6.5": 0.046553,
        "-5.5": 0.074948, "-4.5": 0.120660, "-3.5": 0.194255, "-2.5": 0.312736,
        "-1.5": 0.503483, "-0.5": 0.810573, "0.5": 0.810573, "1.5": 0.503483,
        "2.5": 0.312736, "3.5": 0.194255, "4.5": 0.120660, "5.5": 0.074948,
        "6.5": 0.046553, "7.5": 0.028916, "8.5": 0.017961, "9.5": 0.011157
    }
    
    # Normalize weights to peak value (your original approach)
    norm_factor = weight_table["-0.5"]
    for k in weight_table:
        weight_table[k] /= norm_factor
    
    # ORIGINAL label mapping from your working code
    label_to_bcenter = {
        "6<=|b|<=10": "8.5",
        "4<=|b|<=6": "5.5",
        "2.5<=|b|<=4": "3.5", 
        "1.7<=|b|<=2.5": "2.5",
        "|b|<=1.7": "0.5"
    }
    
    # Build APOGEE composite exactly like your working version
    apogee_composite = np.zeros_like(x)
    curves = []
    
    for i in range(len(mu1)):
        # Build 3-component Gaussian mixture for this latitude band
        mu = [mu1[i], mu2[i], mu3[i]]
        sigma = [sigma1[i], sigma2[i], sigma3[i]]
        weights = np.array([w1[i], w2[i], w3[i]])
        
        # Normalize component weights
        weights /= np.sum(weights)
        
        # Build mixture
        y_total = np.zeros_like(x)
        for m, s, w in zip(mu, sigma, weights):
            y_total += w * norm.pdf(x, loc=m, scale=s)
        
        # Normalize to unit area
        y_total /= np.sum(y_total * dx_actual)
        
        # Store curve
        curves.append((labels[i], y_total))
        
        # Apply latitude weighting
        band_label = labels[i]
        b_center_str = label_to_bcenter.get(band_label, None)
        if b_center_str is None:
            print(f"  Warning: No mapping for band {band_label}")
            continue
            
        if b_center_str not in weight_table:
            print(f"  Warning: No weight for b_center {b_center_str}")
            continue
            
        lat_weight = weight_table[b_center_str]
        apogee_composite += lat_weight * y_total
        
        print(f"  {band_label} -> b={b_center_str}, weight={lat_weight:.3f}")
    
    # Final normalization for APOGEE
    if np.sum(apogee_composite) > 0:
        apogee_composite /= np.sum(apogee_composite * dx_actual)
    
    # ===============================
    # BDBS COMPONENT (FIXED)
    # ===============================
    
    print("\n2. Processing BDBS data...")
    
    try:
        bdbs_x, bdbs_y = np.loadtxt(bdbs_file, unpack=True)
        print(f"  Loaded BDBS: {len(bdbs_x)} points from {bdbs_x.min():.2f} to {bdbs_x.max():.2f}")
        
        # Interpolate BDBS to common grid
        f_interp = interp1d(bdbs_x, bdbs_y, kind='linear', 
                           bounds_error=False, fill_value=0.0)
        bdbs_interp = f_interp(x)
        
        # Normalize BDBS to unit area
        if np.sum(bdbs_interp) > 0:
            bdbs_interp /= np.sum(bdbs_interp * dx_actual)
            
    except Exception as e:
        print(f"  Error loading BDBS: {e}")
        bdbs_interp = np.zeros_like(x)
    
    # ===============================
    # COMBINATION (CLEAN APPROACH)
    # ===============================
    
    print(f"\n3. Combining surveys...")
    
    # Normalize weights
    total_weight = apogee_weight + bdbs_weight
    w_apogee = apogee_weight / total_weight
    w_bdbs = bdbs_weight / total_weight
    
    # Simple weighted combination
    combined_mdf = w_apogee * apogee_composite + w_bdbs * bdbs_interp
    
    # Final normalization
    if np.sum(combined_mdf) > 0:
        combined_mdf /= np.sum(combined_mdf * dx_actual)
    
    print(f"  Final weights: APOGEE={w_apogee:.3f}, BDBS={w_bdbs:.3f}")
    
    # ===============================
    # QUALITY CHECKS
    # ===============================
    
    print(f"\n4. Quality checks...")
    
    # Check normalization
    integral = np.sum(combined_mdf * dx_actual)
    print(f"  Normalization: ∫ P(x) dx = {integral:.6f}")
    
    # Check for negative values
    neg_count = np.sum(combined_mdf < 0)
    print(f"  Negative values: {neg_count}")
    
    # Peak locations
    apogee_peak = x[np.argmax(apogee_composite)]
    bdbs_peak = x[np.argmax(bdbs_interp)]
    combined_peak = x[np.argmax(combined_mdf)]
    
    print(f"  Peak locations:")
    print(f"    APOGEE: [Fe/H] = {apogee_peak:.2f}")
    print(f"    BDBS:   [Fe/H] = {bdbs_peak:.2f}")
    print(f"    Combined: [Fe/H] = {combined_peak:.2f}")
    
    # Overlap consistency (where both surveys have significant signal)
    apogee_mask = apogee_composite > 0.1 * np.max(apogee_composite)
    bdbs_mask = bdbs_interp > 0.1 * np.max(bdbs_interp)
    overlap_mask = apogee_mask & bdbs_mask
    
    if np.any(overlap_mask):
        x_overlap = x[overlap_mask]
        rms_diff = np.sqrt(np.mean((apogee_composite[overlap_mask] - bdbs_interp[overlap_mask])**2))
        print(f"  RMS difference in overlap ([Fe/H] = {x_overlap.min():.1f} to {x_overlap.max():.1f}): {rms_diff:.4f}")
    
    # ===============================
    # SAVE OUTPUT
    # ===============================
    
    # Save main result
    np.savetxt(output_file, np.column_stack((x, combined_mdf)), 
               fmt="%.4f %.6e", header="[Fe/H] Normalized_Count")
    print(f"\nSaved MDF to {output_file}")
    
    # ===============================
    # COMPARISON PLOTS
    # ===============================
    
    if plot_comparison:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Panel 1: Individual APOGEE latitude bands
        ax1 = axes[0, 0]
        for label, y in curves:
            ax1.plot(x, y, '--', alpha=0.7, linewidth=1, label=label)
        ax1.plot(x, apogee_composite, 'k-', linewidth=3, label='APOGEE Composite')
        ax1.set_xlabel('[Fe/H]')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('APOGEE Components')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(feh_range)
        
        # Panel 2: Survey comparison
        ax2 = axes[0, 1]
        ax2.plot(x, apogee_composite, 'b-', linewidth=2, label='APOGEE Composite')
        ax2.plot(x, bdbs_interp, 'r-', linewidth=2, label='BDBS MDF')
        ax2.plot(x, combined_mdf, 'k-', linewidth=3, label='Combined MDF')
        ax2.set_xlabel('[Fe/H]')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Survey Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(feh_range)
        
        # Panel 3: Final result
        ax3 = axes[1, 0]
        ax3.plot(x, combined_mdf, 'k-', linewidth=3)
        ax3.fill_between(x, 0, combined_mdf, alpha=0.3, color='gray')
        ax3.set_xlabel('[Fe/H]')
        ax3.set_ylabel('Probability Density')
        ax3.set_title(f'Final Combined MDF\n(APOGEE: {w_apogee:.1%}, BDBS: {w_bdbs:.1%})')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(feh_range)
        
        # Panel 4: Residuals
        ax4 = axes[1, 1]
        if np.any(overlap_mask):
            residuals = apogee_composite - bdbs_interp
            ax4.plot(x, residuals, 'ko-', markersize=2, alpha=0.7)
            ax4.axhline(0, color='red', linestyle='--')
            ax4.set_xlabel('[Fe/H]')
            ax4.set_ylabel('APOGEE - BDBS')
            ax4.set_title('Survey Differences')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(feh_range)
        
        plt.tight_layout()
        plt.savefig(output_file.replace('.dat', '_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    return x, combined_mdf

# ===============================
# ALTERNATIVE: OVERLAP-WEIGHTED
# ===============================

def create_overlap_weighted_mdf(
    apogee_file='gaussians.dat',
    bdbs_file='binned_dist_lat6_0.08dex.dat',
    output_file='overlap_weighted_mdf.dat'
):
    """
    Alternative approach: weight based on survey overlap and consistency.
    Use APOGEE where it's reliable, BDBS where APOGEE is weak.
    """
    
    # First create the basic combination
    x, combined_basic = create_corrected_mdf_combination(
        apogee_file, bdbs_file, plot_comparison=False
    )
    
    # Load components separately for analysis
    df = pd.read_csv(apogee_file, delimiter=",")
    # ... [build APOGEE composite as above] ...
    
    # Determine reliability regions
    # Use APOGEE where it has good statistics (near peak)
    # Use BDBS in metal-poor tail where APOGEE is sparse
    
    # This is a more sophisticated approach for future development
    return x, combined_basic

# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    print("CORRECTED MDF COMBINATION")
    print("=========================")
    print("Using your working approach with better implementation")
    
    # Test different weightings
    combinations = [
        (0.5, 0.5, "equal_weight"),
    ]
    
    for apogee_w, bdbs_w, name in combinations:
        print(f"\n--- Testing {name} (APOGEE: {apogee_w}, BDBS: {bdbs_w}) ---")
        
        x, mdf = create_corrected_mdf_combination(
            apogee_weight=apogee_w,
            bdbs_weight=bdbs_w,
            output_file=f'{name}_mdf.dat',
            plot_comparison=(name == "equal_weight")  # Only plot the first one
        )
    