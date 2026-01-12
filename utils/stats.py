import numpy as np
from scipy.stats import pearsonr

def compute_symmetry_metrics(primary_y, comparison_y, weight_y=None, is_log=False):
    """
    Computes universal 'Bias' and 'Match' metrics for comparing primary and 
    symmetric datasets.
    
    Parameters:
    -----------
    primary_y : np.ndarray
        The reference (primary) curve data.
    comparison_y : np.ndarray
        The curve data to compare against the primary.
    weight_y : np.ndarray, optional
        Weights for the bias calculation (usually S11 intensity).
    is_log : bool, optional
        If True, computes metrics in log-space (recommended for S11).
        
    Returns:
    --------
    dict
        A dictionary containing the 'bias' and 'shape_match' metrics.
    """
    # Ensure no NaNs or Infs
    mask = np.isfinite(primary_y) & np.isfinite(comparison_y)
    if weight_y is not None:
        mask &= np.isfinite(weight_y)
    
    p = primary_y[mask]
    c = comparison_y[mask]
    w = weight_y[mask] if weight_y is not None else np.ones_like(p)
    
    if len(p) < 2:
        return {"bias": 0.0, "match": 0.0}

    # 1. BIAS (Scale Metric)
    if is_log:
        # For Log-Intensity: Measuring average magnitude shift
        # Delta = mean(log10(Comp) - log10(Pri))
        # We clamp to avoid log(<=0)
        p_log = np.log10(np.maximum(p, 1e-9))
        c_log = np.log10(np.maximum(c, 1e-9))
        # Weighted mean difference in log space
        bias_val = np.sum((c_log - p_log) * w) / np.sum(w)
    else:
        # For Ratios: Measuring absolute shift
        # Delta = mean(Comp - Pri) weighted by intensity
        bias_val = np.sum((c - p) * w) / np.sum(w)

    # 2. MATCH (Shape Metric)
    if is_log:
        p_corr = np.log10(np.maximum(p, 1e-9))
        c_corr = np.log10(np.maximum(c, 1e-9))
    else:
        p_corr, c_corr = p, c
        
    try:
        corr_coef, _ = pearsonr(p_corr, c_corr)
        match_val = corr_coef * 100.0 # Percentage
    except:
        match_val = 0.0

    return {
        "bias": bias_val,
        "match": match_val
    }

def get_metrics_description():
    """Returns a brief description of the metrics."""
    return """
### Quantitative Symmetry Metrics

To evaluate how well different experimental states (e.g., Extended Symmetry vs. Primary) match, we use two key metrics:

1. **Bias (Scale Metric)**:
   - **For Intensity (S11)**: Measures the average systematic power offset. Computed as the weighted mean difference in log-space. A value of +0.02 means the comparison is consistently higher (roughly +5% power).
   - **For Ratios (S12/S11, Depol)**: Measures the absolute vertical shift between curves, weighted by signal intensity (S11) to ensure high-noise, low-signal regions don't skew the results.

2. **Match (Shape Metric)**:
   - Measures the angular consistency of features (peaks and valleys) using the Pearson Correlation Coefficient.
   - **100%**: The curves have identical angular features, regardless of absolute scale.
   - **<90%**: Indicates physical differences in scattering behavior or flow instability.
"""
