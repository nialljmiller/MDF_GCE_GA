# Authors: N Miller


import numpy as np
from scipy.stats import gaussian_kde

def check_alpha_distribution_properties(alpha_arrs, liberal=False):
    """
    Check alpha abundance distribution properties: peak location and FWHM.
    
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
    
    element_names = ['[Si/Fe]','[Ca/Fe]','[Mg/Fe]','[Ti/Fe]']
    do_element_names = ['[Si/Fe]','[Ca/Fe]','[Mg/Fe]']

    for i, (alpha_x, alpha_y) in enumerate(alpha_arrs):#[:4]):
        alpha_x = np.array(alpha_x)
        alpha_y = np.array(alpha_y)
        
        if element_names[i] in do_element_names:

            # Skip if no data
            if len(alpha_x) == 0 or len(alpha_y) == 0:
                continue
                
            # Skip if all NaN or infinite
            valid_mask = np.isfinite(alpha_x) & np.isfinite(alpha_y)
            if np.sum(valid_mask) < 10:  # Need at least 10 points for distribution analysis
                continue
                
            alpha_values = alpha_y[valid_mask]
            
            # Remove extreme outliers for distribution analysis
            Q1, Q3 = np.percentile(alpha_values, [25, 75])
            IQR = Q3 - Q1
            outlier_mask = (alpha_values >= Q1 - 3*IQR) & (alpha_values <= Q3 + 3*IQR)
            alpha_clean = alpha_values[outlier_mask]
            
            if len(alpha_clean) < 5:
                continue
                
            # =====================================
            # 1. PEAK LOCATION CHECK
            # =====================================
            
            # Use kernel density estimation to find peak
            try:
                kde = gaussian_kde(alpha_clean)
                test_points = np.linspace(alpha_clean.min(), alpha_clean.max(), 200)
                density = kde(test_points)
                peak_idx = np.argmax(density)
                peak_location = test_points[peak_idx]
                
                # Check if peak is between -0.3 and +0.3
                if not (-0.3 <= peak_location <= 0.3):
                    violation_severity = abs(peak_location) - 0.3
                    if liberal:
                        penalty_factor *= (3 * violation_severity)
                    else:
                        #print(f"REJECTED: {element_names[i]} peak at {peak_location:.3f} (outside [-0.3, 0.3])")
                        is_physical = False
                        return is_physical, penalty_factor
                        
            except Exception:
                # Fallback to simple median if KDE fails
                peak_location = np.median(alpha_clean)
                if not (-0.3 <= peak_location <= 0.3):
                    if liberal:
                        penalty_factor *= 2.0
                    else:
                        is_physical = False
                        return is_physical, penalty_factor
            
            # =====================================
            # 2. FWHM CHECK
            # =====================================
            
            try:
                # Calculate FWHM from KDE
                max_density = np.max(density)
                half_max = max_density / 2.0
                
                # Find points where density crosses half maximum
                above_half_max = density >= half_max
                if np.any(above_half_max):
                    indices_above = np.where(above_half_max)[0]
                    left_idx = indices_above[0]
                    right_idx = indices_above[-1]
                    
                    fwhm = test_points[right_idx] - test_points[left_idx]
                    
                    # Check if FWHM is less than 1.0
                    if fwhm >= 1.0:
                        violation_severity = fwhm - 1.0
                        if liberal:
                            penalty_factor *= (1 + 1 * violation_severity)
                        else:
                            #print(f"REJECTED: {element_names[i]} FWHM = {fwhm:.3f} (>= 1.0)")
                            is_physical = False
                            return is_physical, penalty_factor
                            
            except Exception:
                # Fallback to standard deviation-based width estimate
                std_dev = np.std(alpha_clean)
                fwhm_approx = 2.355 * std_dev  # FWHM ≈ 2.355 * σ for Gaussian
                
                if fwhm_approx >= 1.0:
                    if liberal:
                        penalty_factor *= 3.0
                    else:
                        is_physical = False
                        return is_physical, penalty_factor
            

    
    return is_physical, penalty_factor


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
    
    element_names = ['[Si/Fe]','[Ca/Fe]','[Mg/Fe]','[Ti/Fe]']
    do_element_names = ['[Si/Fe]','[Ca/Fe]','[Mg/Fe]']

    for i, (alpha_x, alpha_y) in enumerate(alpha_arrs):
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
        
        if element_names[i] in do_element_names:



            # Bin 1: [Fe/H] < -1.0 → alpha should be > 0.15
            bin1_mask = alpha_x < -1.0
            if np.sum(bin1_mask) > 0:
                bin1_alpha = alpha_y[bin1_mask]
                violations = np.sum(bin1_alpha <= 0.15)
                violation_fraction = violations / len(bin1_alpha)
                
                if violation_fraction > 0.05:  # More than 5% violations
                    if liberal:
                        penalty_factor *= (3 * violation_fraction)
                    else:
                        is_physical = False
                        return is_physical, penalty_factor
                elif violations > 0:
                    penalty_factor *= (3 * violation_fraction)
            


            # Bin 2: -1.0 <= [Fe/H] < -0.5 → alpha should be between 0 and 0.4
            bin2_mask = (alpha_x >= -1.0) & (alpha_x < -0.5)
            if np.sum(bin2_mask) > 0:
                bin2_alpha = alpha_y[bin2_mask]
                violations = np.sum((bin2_alpha < 0.05) | (bin2_alpha > 0.4))
                violation_fraction = violations / len(bin2_alpha)
                
                if violation_fraction > 0.10:  # More than 10% violations
                    if liberal:
                        penalty_factor *= (2 * violation_fraction)
                    else:
                        is_physical = False
                        return is_physical, penalty_factor
                elif violations > 0:
                    penalty_factor *= (2 * violation_fraction)
            
            # Bin 3: [Fe/H] > 0.0 → alpha should be between -0.25 and 0.25
            bin3_mask = alpha_x > 0.0
            if np.sum(bin3_mask) > 0:
                bin3_alpha = alpha_y[bin3_mask]
                violations = np.sum((bin3_alpha < -0.2) | (bin3_alpha > 0.2))
                violation_fraction = violations / len(bin3_alpha)
                
                if violation_fraction > 0.10:  # More than 10% violations
                    if liberal:
                        penalty_factor *= (2 * violation_fraction)
                    else:
                        is_physical = False
                        return is_physical, penalty_factor
                elif violations > 0:
                    penalty_factor *= (10 * violation_fraction)
    
    return is_physical, penalty_factor


def check_physical_plausibility(MDF_x_data, MDF_y_data, alpha_arrs, age_x_data, age_y_data, liberal=False, age_meta_check=False):
    """
    Check if model outputs are physically plausible with both binned and distribution-based alpha constraints.
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
            penalty_factor *= 4.0
        else:
            is_physical = False
            return is_physical, penalty_factor
    
    # Check MDF peak location (should be reasonable)
    if len(MDF_y) > 0 and np.max(MDF_y) > 0:
        peak_idx = np.argmax(MDF_y)
        peak_feh = MDF_x[peak_idx]
        
        if not (-1.0 <= peak_feh <= 1.0):
            if liberal:
                penalty_factor *= 3.0
            else:
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
                penalty_factor *= 2.0
            else:
                is_physical = False
                return is_physical, penalty_factor
        
        # Check mean value in the tail  
        mean_tail_count = np.mean(low_feh_counts)
        if mean_tail_count > 0.05:  # Threshold for mean count in tail
            if liberal:
                penalty_factor *= 2.0
            else:
                is_physical = False
                return is_physical, penalty_factor

    # Even stricter check for extremely metal-poor stars ([Fe/H] < -1.5)
    extremely_metal_poor_mask = MDF_x < -1.5
    if np.sum(extremely_metal_poor_mask) > 0:
        extreme_low_feh_counts = MDF_y[extremely_metal_poor_mask]
        max_extreme_tail = np.max(extreme_low_feh_counts)
        
        if max_extreme_tail > 0.03:  # Very strict threshold for extreme tail
            if liberal:
                penalty_factor *= 3.0
            else:
                is_physical = False
                return is_physical, penalty_factor

    # ===============================
    # 3. ALPHA ELEMENT CONSTRAINTS (BINNED)
    # ===============================
    
    alpha_is_physical, alpha_penalty = check_simple_alpha_constraints(alpha_arrs, liberal=liberal)
    
    if not alpha_is_physical:
        return False, penalty_factor
    
    penalty_factor *= alpha_penalty
    
    # ===============================
    # 4. ALPHA DISTRIBUTION PROPERTIES (NEW)
    # ===============================
    
    alpha_dist_is_physical, alpha_dist_penalty = check_alpha_distribution_properties(alpha_arrs, liberal=liberal)
    
    if not alpha_dist_is_physical:
        return False, penalty_factor
    
    penalty_factor *= alpha_dist_penalty
    
    # ===============================
    # 5. BASIC AGE-METALLICITY CHECKS
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
                penalty_factor *= 3.0
            else:
                is_physical = False
                return is_physical, penalty_factor


        if np.any(age_y > 0.7):
            if liberal:
                penalty_factor *= 3.0
            else:
                is_physical = False
                return is_physical, penalty_factor
    

        
        # Check that young stars (age < 8 Gyr) have reasonable median metallicity
        young_stars_mask = age_gyr < 8.0
        if np.sum(young_stars_mask) > 0:
            young_feh = age_y[young_stars_mask]
            
            # Remove any extreme outliers or invalid values
            valid_young_feh = young_feh[np.isfinite(young_feh)]
            
            if len(valid_young_feh) > 0:
                median_young_feh = np.median(valid_young_feh)
                
                # Check if median is between -0.5 and 0.6
                if not (-0.5 <= median_young_feh <= 0.6):
                    violation_severity = max(abs(median_young_feh + 0.5), abs(median_young_feh - 0.6)) - 0.5
                    if violation_severity > 0:
                        if liberal:
                            penalty_factor *= (1 + 3 * violation_severity)
                        else:
                            #print(f"REJECTED: Young stars median [Fe/H] = {median_young_feh:.3f} (outside [-0.5, 0.6])")
                            is_physical = False
                            return is_physical, penalty_factor
    



    # ===============================
    # 6. GLOBAL SANITY CHECKS
    # ===============================
    
    # Check for NaN or inf values anywhere
    all_arrays = [MDF_x, MDF_y, age_x, age_y]
    for alpha_x, alpha_y in alpha_arrs:
        all_arrays.extend([np.array(alpha_x), np.array(alpha_y)])
    
    for arr in all_arrays:
        if len(arr) > 0 and (np.any(np.isnan(arr)) or np.any(np.isinf(arr))):
            is_physical = False
            penalty_factor *= 3.0
            return is_physical, penalty_factor
    
    return is_physical, penalty_factor


def apply_physics_penalty(loss_value, MDF_x_data, MDF_y_data, alpha_arrs, age_x_data, age_y_data):
    """
    Convenience function to apply physics penalty to a loss value.
    """
    
    is_physical, penalty_factor = check_physical_plausibility(MDF_x_data, MDF_y_data, alpha_arrs, age_x_data, age_y_data, liberal=True, age_meta_check=True)

    return penalty_factor