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
    alpha = 0.7  # WRMSE
    beta = 0.2   # Cosine
    gamma = 0.1  # Huber

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
