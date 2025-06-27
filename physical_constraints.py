import numpy as np

def check_simple_alpha_constraints(alpha_arrs, liberal=False):
    """
    Simple three-bin check for alpha element abundances.
    
    Parameters:
    -----------
    alpha_arrs : list of [x_data, y_data] pairs
        Alpha element abundances vs [Fe/H] for [Mg/Fe], [Si/Fe], [Ca/Fe], [Ti/Fe]
    liberal : bool
        If True, use penalties instead of hard rejection
        
    Returns:
    --------
    is_physical : bool
        True if model passes all checks
    penalty_factor : float
        Multiplier for loss function (1.0 = no penalty, >1.0 = penalty)
    """
    
    penalty_factor = 1.0
    is_physical = True
    
    if len(alpha_arrs) < 4:  # Need all 4 alpha elements
        return True, 1.0
    
    element_names = ['Mg', 'Si', 'Ca', 'Ti']
    
    for i, (alpha_x, alpha_y) in enumerate(alpha_arrs[:3]):
        alpha_x = np.array(alpha_x)
        alpha_y = np.array(alpha_y)
        
        # Skip if no data
        if len(alpha_x) == 0 or len(alpha_y) == 0:
            continue
            
        # Skip if all NaN or infinite
        valid_mask = np.isfinite(alpha_x) & np.isfinite(alpha_y)
        if np.sum(valid_mask) == 0:
            continue
            
        alpha_x = alpha_x[valid_mask]
        alpha_y = alpha_y[valid_mask]
        
        # Bin 1: [Fe/H] < -1.0 → alpha should be > 0.15
        bin1_mask = alpha_x < -1.0
        if np.sum(bin1_mask) > 0:
            bin1_alpha = alpha_y[bin1_mask]
            violations = np.sum(bin1_alpha <= 0.15)
            violation_fraction = violations / len(bin1_alpha)
            
            #print(f"  {element_names[i]} Bin1 ([Fe/H] < -1.0): {violations}/{len(bin1_alpha)} violations ({violation_fraction:.2%})")
            
            if violation_fraction > 0.05:  # More than 5% violations
                if liberal:
                    penalty_factor *= (1 + 50 * violation_fraction)
                else:
                    #print(f"REJECTED: {element_names[i]} has {violations}/{len(bin1_alpha)} points <= 0.15 for [Fe/H] < -1.0")
                    is_physical = False
                    return is_physical, penalty_factor
            elif violations > 0:
                penalty_factor *= (1 + 10 * violation_fraction)
        
        # Bin 2: -1.0 <= [Fe/H] < -0.5 → alpha should be between 0 and 0.4
        bin2_mask = (alpha_x >= -1.0) & (alpha_x < -0.5)
        if np.sum(bin2_mask) > 0:
            bin2_alpha = alpha_y[bin2_mask]
            violations = np.sum((bin2_alpha < 0.05) | (bin2_alpha > 0.6))
            violation_fraction = violations / len(bin2_alpha)
            
            #print(f"  {element_names[i]} Bin2 (-1.0 to -0.5): {violations}/{len(bin2_alpha)} violations ({violation_fraction:.2%})")
            
            if violation_fraction > 0.10:  # More than 10% violations
                if liberal:
                    penalty_factor *= (1 + 20 * violation_fraction)
                else:
                    #print(f"REJECTED: {element_names[i]} has {violations}/{len(bin2_alpha)} points outside [0, 0.4] for -1.0 <= [Fe/H] < -0.5")
                    is_physical = False
                    return is_physical, penalty_factor
            elif violations > 0:
                penalty_factor *= (1 + 5 * violation_fraction)
        
        # Bin 3: [Fe/H] > 0.0 → alpha should be between -0.25 and 0.25
        bin3_mask = alpha_x > 0.0
        if np.sum(bin3_mask) > 0:
            bin3_alpha = alpha_y[bin3_mask]
            violations = np.sum((bin3_alpha < -0.2) | (bin3_alpha > 0.2))
            violation_fraction = violations / len(bin3_alpha)
            
            #print(f"  {element_names[i]} Bin3 ([Fe/H] > 0.0): {violations}/{len(bin3_alpha)} violations ({violation_fraction:.2%})")
            #print(f"    Min: {np.min(bin3_alpha):.3f}, Max: {np.max(bin3_alpha):.3f}")
            
            if violation_fraction > 0.10:  # More than 10% violations
                if liberal:
                    penalty_factor *= (1 + 20 * violation_fraction)
                else:
                    #print(f"REJECTED: {element_names[i]} has {violations}/{len(bin3_alpha)} points outside [-0.25, 0.25] for [Fe/H] > 0.0")
                    is_physical = False
                    return is_physical, penalty_factor
            elif violations > 0:
                penalty_factor *= (1 + 5 * violation_fraction)
    
    #print(f"Alpha constraints penalty factor: {penalty_factor:.2f}")
    return is_physical, penalty_factor


def check_physical_plausibility(MDF_x_data, MDF_y_data, alpha_arrs, age_x_data, age_y_data, liberal=False, age_meta_check=False):
    """
    Check if model outputs are physically plausible with simple alpha constraints.
    """
    
    penalty_factor = 1.0
    is_physical = True
    
    # Convert to numpy arrays for safety
    MDF_x = np.array(MDF_x_data)
    MDF_y = np.array(MDF_y_data)
    age_x = np.array(age_x_data)
    age_y = np.array(age_y_data)
    
    # ===============================
    # 1. BASIC MDF CHECKS
    # ===============================
    
    # Check for negative MDF values
    if np.any(MDF_y < 0):
        if liberal:
            penalty_factor *= 20.0
        else:
            #print("REJECTED: Negative MDF values")
            is_physical = False
            return is_physical, penalty_factor
    
    # Check MDF peak location (should be reasonable)
    if len(MDF_y) > 0 and np.max(MDF_y) > 0:
        peak_idx = np.argmax(MDF_y)
        peak_feh = MDF_x[peak_idx]
        
        if not (-1.0 <= peak_feh <= 1.0):
            if liberal:
                penalty_factor *= 10.0
            else:
                #print(f"REJECTED: MDF peak at [Fe/H] = {peak_feh:.2f}")
                is_physical = False
                return is_physical, penalty_factor

    # Check MDF peak location (should be reasonable)
    if len(MDF_y) > 0 and np.max(MDF_y) > 0:
        peak_idx = np.argmax(MDF_y)
        peak_feh = MDF_x[peak_idx]
        
        if not (-1.0 <= peak_feh <= 1.0):
            if liberal:
                penalty_factor *= 10.0
            else:
                #print(f"REJECTED: MDF peak at [Fe/H] = {peak_feh:.2f}")
                is_physical = False
                return is_physical, penalty_factor
    

    # ===============================
    # 2. LOW [Fe/H] TAIL CHECK  
    # ===============================

    # Check that very metal-poor stars ([Fe/H] < -1.0) have low number counts
    very_metal_poor_mask = MDF_x < -1.0
    if np.sum(very_metal_poor_mask) > 0:
        low_feh_counts = MDF_y[very_metal_poor_mask]
        
        # Check maximum value in the tail
        max_tail_count = np.max(low_feh_counts)
        if max_tail_count > 0.1:  # Threshold for maximum allowed count in tail
            if liberal:
                penalty_factor *= 5.0
            else:
                #print(f"REJECTED: Low [Fe/H] tail too high (max = {max_tail_count:.3f})")
                is_physical = False
                return is_physical, penalty_factor
        
        # Check mean value in the tail  
        mean_tail_count = np.mean(low_feh_counts)
        if mean_tail_count > 0.05:  # Threshold for mean count in tail
            if liberal:
                penalty_factor *= 3.0
            else:
                #print(f"REJECTED: Low [Fe/H] tail mean too high (mean = {mean_tail_count:.3f})")
                is_physical = False
                return is_physical, penalty_factor

    # Even stricter check for extremely metal-poor stars ([Fe/H] < -1.5)
    extremely_metal_poor_mask = MDF_x < -1.5
    if np.sum(extremely_metal_poor_mask) > 0:
        extreme_low_feh_counts = MDF_y[extremely_metal_poor_mask]
        max_extreme_tail = np.max(extreme_low_feh_counts)
        
        if max_extreme_tail > 0.03:  # Very strict threshold for extreme tail
            if liberal:
                penalty_factor *= 10.0
            else:
                #print(f"REJECTED: Extreme low [Fe/H] tail too high (max = {max_extreme_tail:.3f})")
                is_physical = False
                return is_physical, penalty_factor
    # ===============================
    # 2. SIMPLE ALPHA ELEMENT CONSTRAINTS
    # ===============================
    
    #print("Checking alpha constraints:")
    alpha_is_physical, alpha_penalty = check_simple_alpha_constraints(alpha_arrs, liberal=liberal)
    
    if not alpha_is_physical:
        return False, penalty_factor
    
    penalty_factor *= alpha_penalty
    
    # ===============================
    # 3. BASIC AGE-METALLICITY CHECKS
    # ===============================
    
    if age_meta_check and len(age_x) > 0 and len(age_y) > 0:
        
        # Convert age from years to Gyr if needed
        if np.max(age_x) > 100:
            age_gyr = age_x / 1e9
        else:
            age_gyr = age_x
            
        # Check for reasonable age range
        if np.any(age_gyr < 0) or np.any(age_gyr > 15):
            if liberal:
                penalty_factor *= 1.3
            else:
                #print("REJECTED: Unreasonable age range")
                is_physical = False
                return is_physical, penalty_factor
    
    # ===============================
    # 4. GLOBAL SANITY CHECKS
    # ===============================
    
    # Check for NaN or inf values anywhere
    all_arrays = [MDF_x, MDF_y, age_x, age_y]
    for alpha_x, alpha_y in alpha_arrs:
        all_arrays.extend([np.array(alpha_x), np.array(alpha_y)])
    
    for arr in all_arrays:
        if len(arr) > 0 and (np.any(np.isnan(arr)) or np.any(np.isinf(arr))):
            #print("REJECTED: NaN or inf values found")
            is_physical = False
            penalty_factor *= 10.0
            return is_physical, penalty_factor
    
    #print(f"Model PASSED with total penalty factor: {penalty_factor:.2f}")
    return is_physical, penalty_factor


def apply_physics_penalty(loss_value, MDF_x_data, MDF_y_data, alpha_arrs, age_x_data, age_y_data):
    """
    Convenience function to apply physics penalty to a loss value.
    """
    
    is_physical, penalty_factor = check_physical_plausibility(MDF_x_data, MDF_y_data, alpha_arrs, age_x_data, age_y_data, liberal=False, age_meta_check=True)
    
    if not is_physical:
        # Return a very high loss for unphysical models
        return 1000.0
    else:
        # Apply penalty factor
        return loss_value * penalty_factor