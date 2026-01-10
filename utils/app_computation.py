"""
utils/app_computation.py
Computation logic with Streamlit caching.
"""
import streamlit as st
import numpy as np
import time
from PIL import Image
from utils import polarimeter_processing
from utils import app_utils

def robust_load_image(path, retries=5, delay=2):
    for i in range(retries):
        try:
            return Image.open(path)
        except (OSError, FileNotFoundError):
            if i == retries - 1: return None
            time.sleep(delay)
    return None

@st.cache_data(persist="disk", show_spinner=False)
def calculate_curves_for_iteration(iteration, measurements, backgrounds, required_meas, analysis_type, bg_on, noise_sigma, rel_thresh, _angle_model, subtract_wall=False):
    iter_files = measurements.get(iteration, {})
    missing_files = [m for m in required_meas if m not in iter_files]
    if missing_files: return None

    ref_meas_type = 'I_PV' if analysis_type == "Mueller Matrix" else 'Depol_Parallel'
    if ref_meas_type not in iter_files: return None

    pil_img = robust_load_image(iter_files[ref_meas_type])
    if pil_img is None: return None
    ref_signal_img = np.array(pil_img)
    
    ref_receiver_key = app_utils.get_receiver_key_for_measurement(ref_meas_type)
    ref_bg_files = backgrounds.get(ref_receiver_key) if bg_on else []
    
    ref_bg_imgs = []
    for f in ref_bg_files:
        b_img = robust_load_image(f)
        if b_img is None: return None
        ref_bg_imgs.append(np.array(b_img))
    
    ref_subtracted_img = polarimeter_processing.get_subtracted_image(ref_signal_img, ref_bg_imgs)

    top_bounds, bottom_bounds, center_path = polarimeter_processing.fit_and_generate_beam_model_bounds(ref_subtracted_img, rel_thresh, _angle_model)
    
    img_h, img_w = ref_signal_img.shape
    roi_mask = np.zeros((img_h, img_w), dtype=bool)
    for i in range(img_w):
        top, bottom = int(round(top_bounds[i])), int(round(bottom_bounds[i]))
        roi_mask[top:bottom, i] = True

    intensity_curves = {}
    for meas_type in required_meas:
        signal_file = iter_files[meas_type]
        s_img = robust_load_image(signal_file)
        if s_img is None: return None
        signal_img = np.array(s_img)
        
        bg_files = backgrounds.get(app_utils.get_receiver_key_for_measurement(meas_type)) if bg_on else []
        bg_imgs = []
        for f in bg_files:
             b_img = robust_load_image(f)
             if b_img is None: return None
             bg_imgs.append(np.array(b_img))
        
        _, noise_map = polarimeter_processing.calculate_background_stats(bg_imgs) if bg_imgs else (None, np.zeros_like(signal_img))
        results = polarimeter_processing.process_measurement(signal_img, bg_imgs, noise_map, roi_mask, noise_sigma, subtract_background=bg_on, subtract_wall=subtract_wall)
        intensity_curves[meas_type] = results['intensity_curve']
    
    return intensity_curves, {"roi_path": center_path, "roi_top": top_bounds, "roi_bottom": bottom_bounds}

def run_batch_process(measurements, backgrounds, iterations, precomputed_data, analysis_type, bg_on, n_sigma, l_thresh, meas_path, angle_model, local_cache_root, export_fmt="NetCDF", subtract_wall=False, bg_info="", progress_callback=None):
    if precomputed_data is None: precomputed_data = {}
    current_data = precomputed_data
    req_meas = app_utils.get_required_measurements(measurements, analysis_type)
    rel_thresh = 10**(-l_thresh)
    
    progress_bar = None
    if progress_callback:
        progress_callback(0.0)
    else:
        # Fallback to Streamlit UI
        progress_bar = st.sidebar.progress(0)
        
    total = len(iterations)
    
    for i, iteration in enumerate(iterations):
        iter_key = str(iteration)
        if iter_key not in current_data:
            result = calculate_curves_for_iteration(iteration, measurements, backgrounds, req_meas, analysis_type, bg_on, n_sigma, rel_thresh, _angle_model=angle_model, subtract_wall=subtract_wall)
            if result:
                curves, _ = result
                iter_res = {}
                if analysis_type == "Mueller Matrix":
                    m = polarimeter_processing.calculate_mueller_elements(curves)
                    if m: iter_res = m
                else:
                    if 'Depol_Parallel' in curves and 'Depol_Cross' in curves:
                        iter_res['Depolarization Ratio'] = (curves['Depol_Cross'] + 1e-9) / (curves['Depol_Parallel'] + 1e-9)
                if iter_res: current_data[iter_key] = iter_res
        
        # Incremental Save (Every 10 iterations)
        if (i > 0 and i % 10 == 0):
             is_dynamic_run = subtract_wall
             meta = {
                "source_path": meas_path, 
                "analysis_type": analysis_type, 
                "parameters": {
                    "bg_subtraction": bg_on, 
                    "noise_sigma": n_sigma, 
                    "log_thresh": l_thresh, 
                    "subtract_wall": subtract_wall
                },
                "background_info": bg_info
            }
             app_utils.save_precomputed_data(meas_path, current_data, meta, fmt=export_fmt, local_cache_root=local_cache_root, is_dynamic=is_dynamic_run)
        
        progress_val = (i + 1) / total
        if progress_callback:
            progress_callback(progress_val)
        elif progress_bar:
            progress_bar.progress(progress_val)
    
    is_dynamic_run = subtract_wall # Save to dynamic file if ANY wall subtraction is used
    meta = {
        "source_path": meas_path, 
        "analysis_type": analysis_type, 
        "parameters": {
            "bg_subtraction": bg_on, 
            "noise_sigma": n_sigma, 
            "log_thresh": l_thresh, 
            "subtract_wall": subtract_wall
        },
        "background_info": bg_info
    }
    save_path = app_utils.save_precomputed_data(meas_path, current_data, meta, fmt=export_fmt, local_cache_root=local_cache_root, is_dynamic=is_dynamic_run)
    return save_path, current_data
