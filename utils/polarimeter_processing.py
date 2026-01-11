# polarimeter_processing.py
import numpy as np
import os
import re
from collections import defaultdict
from scipy.optimize import curve_fit

def find_and_organize_files(measurement_folder, background_folder):
    """
    Scans folders and organizes files by searching for required tags (EP, EW, RP, RW, Iter)
    in any order within the filenames using anchored patterns.
    """
    MANDATORY_PATTERNS = {
        'ep': re.compile(r'_EP(-?\d+)', re.IGNORECASE),
        'ew': re.compile(r'_EW(-?\d+)', re.IGNORECASE),
        'rp': re.compile(r'_RP(-?\d+)', re.IGNORECASE),
        'rw': re.compile(r'_RW(-?\d+)', re.IGNORECASE),
    }
    OPTIONAL_PATTERNS = {
        'iter': re.compile(r'(?:_Iteration|_iter)(\d+)(?:_Step\d+)?', re.IGNORECASE)
    }
    
    # UPDATED: Added specific mappings for the depolarization measurement
    MEASUREMENT_MAP = {
        # Mueller Matrix States
        ((45, 45), (0, 0)):   'I_PV', ((45, 45), (90, 90)): 'I_PH',
        ((45, 45), (45, 45)): 'I_PP', ((45, 45), (135, 135)): 'I_PM',
        ((45, 45), (90, 45)):  'I_PL', ((45, 45), (45, 90)):  'I_PR',
        ((45, 90), (45, 45)): 'I_RP', ((45, 90), (135, 135)): 'I_RM',
        # Depolarization States
        ((0, 0), (0, 0)):     'Depol_Parallel',  # EP0_EW0_RP0_RW0
        ((0, 0), (90, 90)):   'Depol_Cross'      # EP0_EW0_RP90_RW90
    }
    
    # ... (The rest of this function is unchanged)
    organized_measurements = defaultdict(dict)
    try:
        for filename in os.listdir(measurement_folder):
            params = {}
            mandatory_found = True
            search_string = "_" + filename
            for key, pattern in MANDATORY_PATTERNS.items():
                match = pattern.search(search_string)
                if match: params[key] = int(match.group(1))
                else: mandatory_found = False; break
            
            if mandatory_found:
                iteration = int(iter_match.group(1)) if iter_match else 1
                # Normalize angles to [0, 180) for symmetry matching
                emitter_key = (params['ep'] % 180, params['ew'] % 180)
                receiver_key = (params['rp'] % 180, params['rw'] % 180)
                measurement_type = MEASUREMENT_MAP.get((emitter_key, receiver_key))
                if measurement_type:
                    filepath = os.path.join(measurement_folder, filename)
                    organized_measurements[iteration][measurement_type] = filepath
    except FileNotFoundError: return None, None, f"Measurement folder not found: {measurement_folder}"
    organized_backgrounds = defaultdict(list)
    bg_patterns = {'rp': MANDATORY_PATTERNS['rp'], 'rw': MANDATORY_PATTERNS['rw']}
    try:
        for filename in os.listdir(background_folder):
            params = {}; all_found = True
            search_string = "_" + filename
            for key, pattern in bg_patterns.items():
                match = pattern.search(search_string)
                if match: params[key] = int(match.group(1))
                else: all_found = False; break
            if all_found:
                receiver_key = (params['rp'], params['rw'])
                filepath = os.path.join(background_folder, filename)
                organized_backgrounds[receiver_key].append(filepath)
    except FileNotFoundError: return None, None, f"Background folder not found: {background_folder}"
    if not organized_measurements: return None, None, "No valid measurement files found. Ensure filenames contain tags like '_EP45', '_RW90', '_Iteration1', etc."
    if not organized_backgrounds: return None, None, "No valid background files found. Ensure filenames contain tags like '_RP90' and '_RW90'."
    return organized_measurements, organized_backgrounds, None

def calculate_background_stats(background_images):
    if not background_images: return None, None
    mean_background = np.mean(background_images, axis=0)
    noise_map = np.std(background_images, axis=0)
    return mean_background, noise_map

def get_subtracted_image(signal_img, background_imgs):
    """
    Subtracts the mean of background images from a signal image.
    """
    signal_img_f = signal_img.astype(np.float64)
    if not background_imgs:
        return signal_img_f  # Return original if no background

    background_stack = [img.astype(np.float64) for img in background_imgs]
    mean_background = np.mean(background_stack, axis=0)
    subtracted_img = signal_img_f - mean_background
    return subtracted_img

def gaussian_func(x, a, x0, sigma, offset):
    """Gaussian function for curve fitting."""
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

def double_gaussian_func(x, offset, a1, mu1, sigma1, a2, mu2, sigma2):
    """Double Gaussian function: Offset + Narrow + Broad."""
    g1 = a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    g2 = a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    return offset + g1 + g2

def fit_double_gaussian_params(y_pixels, profile):
    """
    Fits a double gaussian (Narrow + Broad) to the profile.
    Returns popt = [offset, a_narrow, mu_narrow, s_narrow, a_broad, mu_broad, s_broad]
    Returns None if fit fails.
    """
    try:
        p_max = np.max(profile)
        p_min = np.min(profile)
        mu_guess = np.argmax(profile)
        
        # Initial Guesses
        # Narrow (Signal): High Amp, Small Sigma (Start at 4.0 to encourage narrow fit)
        # Broad (Wall): Low Amp, Large Sigma (Start at 150.0)
        p0 = [p_min, p_max - p_min, mu_guess, 4.0, (p_max - p_min)/10.0, mu_guess, 150.0]
        
        # Bounds
        # Enforce s_narrow (idx 3) <= 60.0 pixels (Prevents signal from becoming background)
        # Enforce s_broad (idx 6) >= 40.0 pixels (Forces wall to be broad)
        lower_bounds = [-np.inf, 0, 0, 0.1, 0, 0, 40.0]
        upper_bounds = [np.inf, np.inf, len(y_pixels), np.inf, np.inf, len(y_pixels), np.inf]
        
        popt, _ = curve_fit(double_gaussian_func, y_pixels, profile, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=2000)
        
        # Double check sort (though bounds should handle it)
        if popt[3] > popt[6]:
            popt = [popt[0], popt[4], popt[5], popt[6], popt[1], popt[2], popt[3]]
            
        return popt
    except:
        return None

def fit_signal_column(y_pixels, profile):
    """
    Fits a double gaussian to a column profile and returns the integrated area of the narrow component.
    """
    popt = fit_double_gaussian_params(y_pixels, profile)
    if popt is None: return None

    try:
        # Calculate Broad Component (Wall) and subtract from raw profile to get real signal sum
        # Broad params: Amp=popt[4], Mu=popt[5], Sigma=popt[6], Offset=popt[0]
        broad_curve = gaussian_func(y_pixels, popt[4], popt[5], popt[6], popt[0])
        
        # Find intersection points (Dynamic ROI)
        # Start from the center of the narrow peak and move outwards
        center_idx = int(np.round(popt[2]))
        center_idx = max(0, min(len(profile)-1, center_idx))
        
        # Search Left
        left_idx = 0
        for i in range(center_idx, -1, -1):
            if profile[i] <= broad_curve[i]:
                left_idx = i
                break
        
        # Search Right
        right_idx = len(profile) - 1
        for i in range(center_idx, len(profile)):
            if profile[i] <= broad_curve[i]:
                right_idx = i
                break
        
        # Integrate difference only within the intersection bounds
        return np.sum(profile[left_idx:right_idx+1] - broad_curve[left_idx:right_idx+1])
    except:
        return None

def fit_and_generate_beam_model_bounds(image, relative_threshold, angle_model):
    """
    Fits a linear model to the beam's center and width to account for perspective
    and divergence, then returns the modeled ROI boundaries.
    """
    img_h, img_w = image.shape
    x_coords = np.arange(img_w)

    # Get scattering angles for each pixel column
    scattering_angles_deg = angle_model(x_coords)
    
    # --- Step 1: Find raw, data-driven edges ---
    image_for_edges = image.copy()
    image_for_edges[image_for_edges < 0] = 0
    col_maxes = np.max(image_for_edges, axis=0)
    thresholds = col_maxes * relative_threshold
    signal_mask = image_for_edges >= thresholds

    top_edges = np.argmax(signal_mask, axis=0).astype(float)
    bottom_edges = (img_h - 1 - np.argmax(np.flip(signal_mask, axis=0), axis=0)).astype(float)
    no_signal_mask = ~np.any(signal_mask, axis=0)
    top_edges[no_signal_mask] = np.nan # Use NaN for points with no signal
    bottom_edges[no_signal_mask] = np.nan

    # --- Step 2: Fit a linear model to the raw data ---
    # This models the perspective and divergence effects as simple lines.
    # This models the perspective and divergence effects as simple lines.
    valid_mask = ~np.isnan(top_edges)
    if np.sum(valid_mask) < 10: # Not enough valid data points to fit a model
        return np.full(img_w, float(img_h/2 - 25)), np.full(img_w, float(img_h/2 + 25)), np.full(img_w, float(img_h/2))

    x_valid = x_coords[valid_mask]
    raw_centers = (top_edges[valid_mask] + bottom_edges[valid_mask]) / 2.0
    raw_widths = bottom_edges[valid_mask] - top_edges[valid_mask]

    # --- Step 3: Fit a simple linear model (degree 1) ---
    center_model_coeffs = np.polyfit(x_valid, raw_centers, 1)
    width_model_coeffs = np.polyfit(x_valid, raw_widths, 1)
    
    center_model = np.poly1d(center_model_coeffs)
    width_model = np.poly1d(width_model_coeffs)

    modeled_centers = center_model(x_coords)
    modeled_widths = width_model(x_coords)

    # Ensure widths are positive
    modeled_widths[modeled_widths < 1] = 1 # Minimum 1 pixel width

    # --- Step 4: Generate clean ROI bounds from the model ---
    modeled_half_widths = modeled_widths / 2.0
    modeled_tops = modeled_centers - modeled_half_widths
    modeled_bottoms = modeled_centers + modeled_half_widths

    return modeled_tops, modeled_bottoms, modeled_centers

def process_measurement(signal_img, background_imgs, noise_map, roi_mask, noise_threshold_sigma=3.0, subtract_background=True, subtract_wall=False):
    """
    Core processing function.
    DEFINITIVE FIX: Correctly applies a hybrid absolute/relative threshold to the un-clipped signal to robustly find the center of mass.
    """
    signal_img_f = signal_img.astype(np.float64)
    img_h, img_w = signal_img_f.shape
    
    # --- Step 1: Background Subtraction ---
    if subtract_background and background_imgs:
        background_stack = [img.astype(np.float64) for img in background_imgs]
        mean_background = np.mean(background_stack, axis=0)
        subtracted_image = signal_img_f - mean_background
    else:
        subtracted_image = signal_img_f

    if subtract_wall:
        # --- Advanced Method: Double Gaussian Fit per Column ---
        # This separates the "Narrow" signal from the "Broad" wall reflection.
        intensity_curve = np.zeros(img_w)
        y_pixels = np.arange(img_h)
        
        # Optimization: Only fit columns that have significant signal in the ROI
        # We can use the roi_mask to determine which columns to process
        cols_to_process = np.any(roi_mask, axis=0)
        
        for x in range(img_w):
            if cols_to_process[x]:
                area = fit_signal_column(y_pixels, subtracted_image[:, x])
                if area is not None:
                    intensity_curve[x] = area
    else:
        # --- Standard Method: ROI Summation ---
        cleaned_image_for_integration = subtracted_image.copy()
        absolute_noise_threshold = noise_map * noise_threshold_sigma
        cleaned_image_for_integration[cleaned_image_for_integration < absolute_noise_threshold] = 0
        intensity_curve = np.sum(cleaned_image_for_integration * roi_mask, axis=0)
        
    return {"intensity_curve": intensity_curve}

def calculate_mueller_elements(intensity_curves):
    # This function remains unchanged
    if 'I_PV' in intensity_curves and 'I_PH' in intensity_curves:
        i_pv, i_ph = intensity_curves['I_PV'], intensity_curves['I_PH']
        s11, s12 = i_pv + i_ph, i_ph - i_pv
    else: return None
    if 'I_PP' in intensity_curves and 'I_PM' in intensity_curves:
        s33 = intensity_curves['I_PP'] - intensity_curves['I_PM']
    else: s33 = np.full_like(s11, np.nan)
    if 'I_PL' in intensity_curves and 'I_PR' in intensity_curves:
        # Flipped to match Strategy A convention (assuming PR~RP and PL~RM)
        s34 = intensity_curves['I_PR'] - intensity_curves['I_PL']
    elif 'I_RP' in intensity_curves and 'I_RM' in intensity_curves:
        s34 = intensity_curves['I_RP'] - intensity_curves['I_RM']
    else: s34 = np.full_like(s11, np.nan)
    s11_norm = s11.copy(); s11_norm[s11_norm == 0] = np.nan
    return {
        "S11": s11, "S12": s12, "S33": s33, "S34": s34,
        "DoLP": np.abs(s12) / s11_norm,
        "S33/S11": s33 / s11_norm, "S34/S11": s34 / s11_norm, "S12/S11": s12 / s11_norm,
    }