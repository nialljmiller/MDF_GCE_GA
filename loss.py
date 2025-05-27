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
    Weighted combination of Huber loss and (1 - cosine_similarity).
    """
    alpha = 0.7  # Adjust weighting as you like
    beta = 0.3
    
    # Evaluate the existing metrics
    huber_val = compute_huber(GA_class, theory_count_array)            # lower = better
    cos_val   = compute_cosine_similarity(GA_class, theory_count_array) # higher = better
    
    # Combine them so lower is better overall
    return alpha * huber_val + beta * (1.0 - cos_val)

def compute_wrmse(GA_class, theory_count_array):
    return wrmse_compute(theory_count_array, GA_class.normalized_count, GA_class.placeholder_sigma_array)

def compute_mae(GA_class, theory_count_array):
    return np.mean(np.abs(np.array(theory_count_array) - np.array(GA_class.normalized_count)))

def compute_mape(GA_class, theory_count_array):
    return np.mean(np.abs((np.array(theory_count_array) - np.array(GA_class.normalized_count)) / np.array(GA_class.normalized_count))) * 100

def compute_huber(GA_class, theory_count_array):
    return np.mean(huber_loss(GA_class.normalized_count, theory_count_array))

def compute_cosine_similarity(GA_class, theory_count_array):
    return np.dot(GA_class.normalized_count, theory_count_array) / (np.linalg.norm(GA_class.normalized_count) * np.linalg.norm(theory_count_array))

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
