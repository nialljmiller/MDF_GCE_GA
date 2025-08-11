# Authors: N Miller

import numpy as np


def compute_ks_distance(GA_class, theory_count_array):
    """
    1D Kolmogorov–Smirnov distance between the model distribution
    and the observed distribution (GA_class.normalized_count).
    Lower is better.
    """
    model_cdf = np.cumsum(theory_count_array)
    model_cdf /= model_cdf[-1]  # normalize

    data_cdf = np.cumsum(GA_class.normalized_count)
    data_cdf /= data_cdf[-1]

    return np.max(np.abs(model_cdf - data_cdf))




def huber_loss(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean()



# Function to compute WRMSE
def wrmse_compute(predicted, observed, sigma):
    return np.sqrt(np.mean(((predicted - observed) / sigma) ** 2))

def loss_compute(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean()



def compute_ensemble_metric(GA_class, theory_count_array):
    """
    Weighted ensemble of WRMSE, Huber, Cosine Similarity, with KS penalty.
    All components are minimized.
    """
    # Core loss terms
    wrmse_val = compute_wrmse(GA_class, theory_count_array)
    huber_val = compute_huber(GA_class, theory_count_array)
    cosine_val = compute_cosine_similarity(GA_class, theory_count_array)

    # KS distance: used as binary or continuous penalty
    ks_val = compute_ks_distance(GA_class, theory_count_array)
    
    # ---- Weighted sum ----
    alpha = 0.9  # WRMSE
    beta = 0.05   # Cosine
    gamma = 0.05  # Huber

    base_loss = alpha * wrmse_val + beta * cosine_val + gamma * huber_val

    # ---- KS penalty ----
    ks_threshold = 0.2  # If KS > this, trigger penalty
    if ks_val > ks_threshold:
        penalty_scale = 1.0 + 2.5 * (ks_val - ks_threshold)
        base_loss *= penalty_scale

    return base_loss


def compute_wrmse(GA_class, theory_count_array):
    return wrmse_compute(theory_count_array, GA_class.normalized_count, GA_class.placeholder_sigma_array)

def compute_mae(GA_class, theory_count_array):
    return np.mean(np.abs(np.array(theory_count_array) - np.array(GA_class.normalized_count)))

def compute_mape(GA_class, theory_count_array):
    return np.mean(np.abs((np.array(theory_count_array) - np.array(GA_class.normalized_count)) / np.array(GA_class.normalized_count))) * 100

def compute_huber(GA_class, theory_count_array):
    return np.mean(huber_loss(GA_class.normalized_count, theory_count_array))

def compute_cosine_similarity(GA_class, theory_count_array):
    return 1 - np.dot(GA_class.normalized_count, theory_count_array) / (np.linalg.norm(GA_class.normalized_count) * np.linalg.norm(theory_count_array))

def compute_log_cosh(GA_class, theory_count_array):
    return np.mean(np.log(np.cosh(theory_count_array - GA_class.normalized_count)))

def calculate_all_metrics(GA_class, theory_count_array):
    # Calculate all metrics
    wrmse = compute_wrmse(GA_class, theory_count_array)
    mae = compute_mae(GA_class, theory_count_array)
    mape = compute_mape(GA_class, theory_count_array)
    huber = compute_huber(GA_class, theory_count_array)
    cos_similarity = compute_cosine_similarity(GA_class, theory_count_array)
    log_cosh = compute_log_cosh(GA_class, theory_count_array)
    ensemble = compute_ensemble_metric(GA_class, theory_count_array)
    ks = compute_ks_distance(GA_class, theory_count_array)
    return ks, ensemble, wrmse, mae, mape, huber, cos_similarity, log_cosh







def compute_age_metallicity_loss(GA_class, age_x_data, age_y_data, loss_metric='log_likelihood'):
    """
    Compute age-metallicity loss using Joyce data and specified metric.
    
    Parameters:
    -----------
    GA_class : GalacticEvolutionGA
        GA class instance with Joyce observational data
    age_x_data : array
        Model age data (in years, will be converted to Gyr)
    age_y_data : array 
        Model [Fe/H] data
    loss_metric : str
        Metric to use for age-metallicity comparison
        
    Returns:
    --------
    float
        Age-metallicity loss value (lower is better for most metrics)
    """
    
    # Load Joyce observational data from GA class
    if not hasattr(GA_class, '_joyce_age_data'):
        # Load data on first call (should be loaded in GA class initialization)
        GA_class._load_joyce_age_data()
    
    joyce_ages = GA_class.joyce_ages
    joyce_feh = GA_class.joyce_feh
    joyce_feh_err = GA_class.joyce_feh_err
    
    # Convert model ages to Gyr
    model_ages_gyr = (age_x_data[-1] / 1e9) - np.array(age_x_data) / 1e9
    model_feh = np.array(age_y_data)
    
    # Filter for valid data
    valid_joyce = np.isfinite(joyce_ages) & np.isfinite(joyce_feh) & np.isfinite(joyce_feh_err)
    valid_model = np.isfinite(model_ages_gyr) & np.isfinite(model_feh)
    
    if np.sum(valid_joyce) < 5 or np.sum(valid_model) < 5:
        return 1000.0  # Large penalty if insufficient data
    
    joyce_ages_clean = joyce_ages[valid_joyce]
    joyce_feh_clean = joyce_feh[valid_joyce]
    joyce_feh_err_clean = joyce_feh_err[valid_joyce]
    
    model_ages_clean = model_ages_gyr[valid_model]
    model_feh_clean = model_feh[valid_model]
    
    # Find common age range
    min_age = max(np.min(joyce_ages_clean), np.min(model_ages_clean))
    max_age = min(np.max(joyce_ages_clean), np.max(model_ages_clean))
    
    if max_age <= min_age:
        return 1000.0  # No overlap
    
    # Filter data to common age range
    joyce_mask = (joyce_ages_clean >= min_age) & (joyce_ages_clean <= max_age)
    model_mask = (model_ages_clean >= min_age) & (model_ages_clean <= max_age)
    
    if np.sum(joyce_mask) < 3 or np.sum(model_mask) < 3:
        return 1000.0  # Insufficient overlap
    
    joyce_ages_final = joyce_ages_clean[joyce_mask]
    joyce_feh_final = joyce_feh_clean[joyce_mask]
    joyce_err_final = joyce_feh_err_clean[joyce_mask]
    
    model_ages_final = model_ages_clean[model_mask]
    model_feh_final = model_feh_clean[model_mask]
    
    try:
        # Calculate metrics using age_meta functions
        metrics = calculate_all_metrics(
            model_ages_final, model_feh_final,
            joyce_ages_final, joyce_feh_final, joyce_err_final,
            'Joyce'
        )
        
        # Extract the requested metric
        if loss_metric in metrics:
            loss_value = metrics[loss_metric]
            
            # Convert metrics where higher is better to lower is better
            if loss_metric in ['log_likelihood', 'correlation', 'spearman_correlation']:
                # For these metrics, higher values are better, so return negative
                return -loss_value
            else:
                # For these metrics, lower values are better
                return loss_value
        else:
            # Fallback to MAE if requested metric not available
            return metrics.get('mae', 1000.0)
            
    except Exception as e:
        # Return large penalty if calculation fails
        return 1000.0


def compute_combined_loss(GA_class, theory_count_array, age_x_data, age_y_data, 
                         mdf_vs_age_weight=0.8, age_meta_loss_metric='log_likelihood'):
    """
    Compute combined MDF + age-metallicity loss.
    
    Parameters:
    -----------
    GA_class : GalacticEvolutionGA
        GA class instance
    theory_count_array : array
        Model MDF data
    age_x_data : array
        Model age data
    age_y_data : array
        Model [Fe/H] data  
    mdf_vs_age_weight : float
        Weight for MDF vs age-metallicity (0.8 = 80% MDF, 20% age)
    age_meta_loss_metric : str
        Metric for age-metallicity loss
        
    Returns:
    --------
    float
        Combined loss value
    """
    
    # Calculate MDF loss using the selected loss function
    mdf_loss = GA_class.selected_loss_function(GA_class, theory_count_array)
    
    # Calculate age-metallicity loss
    age_loss = compute_age_metallicity_loss(GA_class, age_x_data, age_y_data, age_meta_loss_metric)
    
    # Normalize losses to similar scales (optional - may need tuning)
    # This helps ensure the weighting works as expected
    if mdf_loss > 0:
        mdf_loss_norm = mdf_loss
    else:
        mdf_loss_norm = 0.001
        
    if age_loss > 0:
        age_loss_norm = age_loss  
    else:
        age_loss_norm = 0.001
    
    # Combine with user-specified weighting
    combined_loss = (mdf_vs_age_weight * mdf_loss_norm + 
                    (1.0 - mdf_vs_age_weight) * age_loss_norm)
    
    return combined_loss
