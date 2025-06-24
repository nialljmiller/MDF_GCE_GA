import numpy as np



def check_physical_plausibility(MDF_x_data, MDF_y_data, alpha_arrs, age_x_data, age_y_data, liberal=False, age_meta_check=False):
    """
    Check if model outputs are physically plausible.
    
    Parameters:
    -----------
    MDF_x_data, MDF_y_data : arrays
        Metallicity distribution function
    alpha_arrs : list of [x_data, y_data] pairs
        Alpha element abundances vs [Fe/H] for [Mg/Fe], [Si/Fe], [Ca/Fe], [Ti/Fe]
    age_x_data, age_y_data : arrays
        Age vs [Fe/H] data
    liberal : bool
        If True, use very loose constraints
        
    Returns:
    --------
    is_physical : bool
        True if model passes all physical checks
    penalty_factor : float
        Multiplier for loss function (1.0 = no penalty, >1.0 = penalty)
    """
    
    penalty_factor = 1.0
    is_physical = True
    
    # Convert to numpy arrays for safety
    MDF_x = np.array(MDF_x_data)
    MDF_y = np.array(MDF_y_data)
    age_x = np.array(age_x_data)
    age_y = np.array(age_y_data)
    
    # ===============================
    # 1. MDF CHECKS
    # ===============================
    
    # Check for negative MDF values
    if np.any(MDF_y < 0):
        if liberal:
            penalty_factor *= 20.0
        else:
            is_physical = False
            return is_physical, penalty_factor
    
    # Check MDF peak location (should be reasonable)
    if len(MDF_y) > 0 and np.max(MDF_y) > 0:
        peak_idx = np.argmax(MDF_y)
        peak_feh = MDF_x[peak_idx]
        
        # Very liberal: peak should be between -2.5 and +1.0
        if not (-1.0 <= peak_feh <= 1.0):
            if liberal:
                penalty_factor *= 10.5
            else:
                is_physical = False
                return is_physical, penalty_factor
    
    # Check MDF isn't too narrow (avoid delta functions)
    if len(MDF_y) > 2:
        # Find width at half maximum
        half_max = np.max(MDF_y) / 20.0
        above_half = MDF_y >= half_max
        if np.sum(above_half) < 3:  # Less than 3 points above half max
            if liberal:
                penalty_factor *= 10.3
            else:
                is_physical = False
                return is_physical, penalty_factor
    
    # ===============================
    # 2. ALPHA ELEMENT CHECKS
    # ===============================
    
    if len(alpha_arrs) >= 4:  # We expect [Mg/Fe], [Si/Fe], [Ca/Fe], [Ti/Fe]
        
        for i, (alpha_x, alpha_y) in enumerate(alpha_arrs[:4]):
            alpha_x = np.array(alpha_x)
            alpha_y = np.array(alpha_y)
            
            # Skip if no data
            if len(alpha_x) == 0 or len(alpha_y) == 0:
                continue
                
            # === CLIP alpha_y to physical range before checks
            # Drop values outside ±2.0 (or whatever you define as "insane")
            physical_mask = (alpha_y >= -2.0) & (alpha_y <= 2.0)
            alpha_x = alpha_x[physical_mask]
            alpha_y = alpha_y[physical_mask]

            # Check for extreme alpha values
            if np.any(alpha_y > 1.0) or np.any(alpha_y < -1.0):
                if liberal:
                    penalty_factor *= 1.2
                else:
                    is_physical = False
                    return is_physical, penalty_factor
            
            # Check for alpha enhancement at low [Fe/H] (very liberal check)
            # Look for points with [Fe/H] < -1.0
            low_feh_mask = alpha_x < -1.0
            if np.sum(low_feh_mask) > 0:
                low_feh_alpha = alpha_y[low_feh_mask]
                # At least some points should show enhancement (>-0.2)
                if np.any(low_feh_alpha < 0.1):
                    if liberal:
                        penalty_factor *= 1.2
                    else:
                        is_physical = False
                        return is_physical, penalty_factor


            high_feh_mask = alpha_x > 0.0
            if np.sum(high_feh_mask) > 0:
                high_feh_alpha = alpha_y[high_feh_mask]
                # At least some points should show enhancement (>-0.2)
                if np.any(high_feh_alpha > 0.2):
                    if liberal:
                        penalty_factor *= 1.2
                    else:
                        is_physical = False
                        return is_physical, penalty_factor
    
            if np.median(alpha_y) < 0.0:
                penalty_factor *= 1.2
                if np.median(alpha_y) < -0.1:
                    is_physical = False
                    return is_physical, penalty_factor
            


    # ===============================
    # 3. AGE-METALLICITY CHECKS
    # ===============================
    
    if age_meta_check:

        if len(age_x) > 0 and len(age_y) > 0:
            
            # Convert age from years to Gyr if needed
            if np.max(age_x) > 100:  # Likely in years
                age_gyr = age_x / 1e9
            else:
                age_gyr = age_x
                
            # Check for reasonable age range
            if np.any(age_gyr < 0) or np.any(age_gyr > 15):
                if liberal:
                    penalty_factor *= 1.3
                else:
                    is_physical = False
                    return is_physical, penalty_factor
            
            # Very loose check: old stars shouldn't all be super metal-rich
            if len(age_gyr) > 5:
                old_stars = age_gyr > 10  # Stars older than 10 Gyr
                if np.sum(old_stars) > 0:
                    old_feh = age_y[old_stars]
                    # If ALL old stars have [Fe/H] > 0, that's suspicious
                    if np.all(old_feh > 0.2):
                        if liberal:
                            penalty_factor *= 1.2
                        else:
                            is_physical = False
                            return is_physical, penalty_factor
                    
                    if np.all(old_feh < 0.0):
                        if liberal:
                            penalty_factor *= 1.2
                        else:
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
            is_physical = False
            penalty_factor *= 10.0
            return is_physical, penalty_factor
    
    return is_physical, penalty_factor


def apply_physics_penalty(loss_value, MDF_x_data, MDF_y_data, alpha_arrs, age_x_data, age_y_data):
    """
    Convenience function to apply physics penalty to a loss value.
    
    Parameters:
    -----------
    loss_value : float
        Original loss value
    ... : model outputs (same as check_physical_plausibility)
        
    Returns:
    --------
    penalized_loss : float
        Loss value with physics penalty applied
    """
    
    is_physical, penalty_factor = check_physical_plausibility(
        MDF_x_data, MDF_y_data, alpha_arrs, age_x_data, age_y_data, liberal=False
    )
    
    if not is_physical:
        # Return a very high loss for unphysical models
        return 100.0
    else:
        # Apply penalty factor
        return loss_value * penalty_factor


