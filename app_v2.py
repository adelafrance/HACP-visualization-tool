import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.colors
import os
import json
from collections import defaultdict

from utils import pixel_angle_tool
from utils import polarimeter_processing, app_utils, plotting
from utils import app_computation
from utils import app_visualization
from utils import dashboard

st.set_page_config(layout="wide", page_title="Polarimeter Analysis v2", initial_sidebar_state="expanded")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE_PATH = os.path.join(SCRIPT_DIR, 'utils', 'angle_model_data.npz')
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'config.json')
PIXEL_OFFSET_X = 520
SESSION_FILE = os.path.join(SCRIPT_DIR, 'session_state.json')

# --- Load Config ---
DEFAULT_CACHE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../output/IPT_WIND_TUNNEL"))
config_dict = app_utils.load_config(CONFIG_FILE)
if isinstance(config_dict, str): # Handle legacy string-only config
    config_dict = {"last_path": config_dict, "cache_path": DEFAULT_CACHE_ROOT}
elif not config_dict:
    config_dict = {"last_path": "", "cache_path": DEFAULT_CACHE_ROOT}

LOCAL_CACHE_ROOT = config_dict.get('cache_path', DEFAULT_CACHE_ROOT)

@st.cache_resource
def load_angle_model():
    return pixel_angle_tool.load_angle_model_from_npz(CALIBRATION_FILE_PATH)

angle_model = load_angle_model()

def set_app_mode(mode):
    st.session_state.app_mode = mode

# --- Initialize Session State ---
if 'comparisons' not in st.session_state:
    st.session_state.comparisons = [] # List of comparison sets
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Data Dashboard" # Default mode
if 'iterations' not in st.session_state:
    st.session_state.iterations = []
if 'selected_iter_option' not in st.session_state:
    st.session_state.selected_iter_option = "Average All"
if 'measurements' not in st.session_state:
    st.session_state.measurements = {}
if 'backgrounds' not in st.session_state:
    st.session_state.backgrounds = {}
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = "Mueller Matrix"

# Global Session State from File
p_state = {}
if os.path.exists(SESSION_FILE):
    try:
        with open(SESSION_FILE, 'r') as f: p_state = json.load(f)
    except: pass
st.session_state.p_state = p_state
# --------------------------------

# --- Helper: Handle Dashboard Load Request ---
if 'target_load' in st.session_state and st.session_state.target_load:
    tl = st.session_state.target_load
    
    # 1. Update Persistent Session File
    current_state = {}
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f: current_state = json.load(f)
        except: pass
    
    # Update fields
    current_state['date'] = tl['date']
    current_state['meas_parent'] = tl['meas_parent']
    current_state['meas_seq'] = tl['meas_seq']
    if 'meas_path' in tl: current_state['meas_path'] = tl['meas_path']
    if 'description' in tl: current_state['description'] = tl['description']
    
    if tl.get('bg_parent'): 
        current_state['bg_parent'] = tl['bg_parent']
        # Try to infer a bg_path for robustness
        # Construct path: date_folder/bg_parent/best_match
        try:
             date_dir = os.path.dirname(os.path.dirname(tl['meas_path']))
             bg_parent_dir = os.path.join(date_dir, tl['bg_parent'])
             best_bg = app_utils.find_best_background_folder(tl['meas_seq'], tl['meas_path'], bg_parent_dir)
             if best_bg:
                 current_state['bg_path'] = os.path.join(bg_parent_dir, best_bg)
        except: pass
        
    # Best effort bg_seq usually handled by finding logic, can leave blank or try to set if known
    
    with open(SESSION_FILE, 'w') as f: json.dump(current_state, f)
    
    # 2. Switch Mode
    st.session_state.app_mode = tl.get('switch_to_mode', "Interactive Analysis")
    
    # 3. Clear Request & Rerun
    del st.session_state.target_load
    st.rerun()

# ... (Previous helper functions omitted for brevity, they remain unchanged) ...
def go_to_previous():
    if 'iterations' in st.session_state and st.session_state.iterations:
        formatted_labels = [f"{app_utils.translate_sequence_name(str(it))} ({it})" for it in st.session_state.iterations]
        opts = ["Average All"] + formatted_labels
        curr = st.session_state.selected_iter_option
        if curr in opts:
            idx = opts.index(curr)
            if idx > 1:
                st.session_state.selected_iter_option = opts[idx - 1]

def go_to_next():
    if 'iterations' in st.session_state and st.session_state.iterations:
        formatted_labels = [f"{app_utils.translate_sequence_name(str(it))} ({it})" for it in st.session_state.iterations]
        opts = ["Average All"] + formatted_labels
        curr = st.session_state.selected_iter_option
        if curr in opts:
            idx = opts.index(curr)
            if idx < len(opts) - 1:
                st.session_state.selected_iter_option = opts[idx + 1]

def render_signal_decomposition(img_float, key_name, col_idx, roi_paths, bit_depth):
    # ... (function body remains identical) ...
    """Renders the signal decomposition plot for a specific column."""
    with st.container():
        with st.container():
            try:
                # Extract profile at the selected column
                # Image is (Height, Width), so we slice [:, col_idx]
                norm_profile = img_float[:, col_idx]

                # --- ROI Intersection Logic ---
                # FIX: Use direct array access for ROI instead of trace iteration
                y_min, y_max = None, None
                if roi_paths and 'roi_top' in roi_paths and 'roi_bottom' in roi_paths:
                    # Ensure col_idx is within bounds
                    if 0 <= col_idx < len(roi_paths['roi_top']):
                        y_min = roi_paths['roi_top'][col_idx]
                        y_max = roi_paths['roi_bottom'][col_idx]
                
                x_max = np.max(norm_profile) * 1.1 if np.max(norm_profile) > 0 else 1.0
                
                # --- Gaussian Fitting ---
                y_pixels = np.arange(len(norm_profile))
                
                # Double Gaussian Fit (Broad + Narrow)
                fig_prof = go.Figure()
                
                # Use shared fitting logic with tighter bounds
                popt = polarimeter_processing.fit_double_gaussian_params(y_pixels, norm_profile)
                
                # Save stats for later rendering
                stats_msg = None
                if popt is not None:

                    # Generate curves
                    narrow_curve = polarimeter_processing.gaussian_func(y_pixels, popt[1], popt[2], popt[3], 0) # Isolated signal
                    broad_curve = polarimeter_processing.gaussian_func(y_pixels, popt[4], popt[5], popt[6], popt[0]) # Include offset in broad
                    total_curve = polarimeter_processing.double_gaussian_func(y_pixels, *popt)
                    
                    # 1. Broad (Wall) - Gray, Fill to axis
                    fig_prof.add_trace(go.Scatter(x=broad_curve, y=y_pixels, mode='lines', name='Wall (Broad)', line=dict(color='gray', dash='dash'), fill='tozerox', fillcolor='rgba(128, 128, 128, 0.2)'))
                    
                    # 2. Signal (Isolated) - Pink
                    fig_prof.add_trace(go.Scatter(x=narrow_curve, y=y_pixels, mode='lines', name='Signal (Isolated)', line=dict(color='magenta', width=2), fill='tozerox', fillcolor='rgba(255, 0, 255, 0.2)'))

                    # 3. Total Fit (Sum) - Dashed
                    fig_prof.add_trace(go.Scatter(x=total_curve, y=y_pixels, mode='lines', name='Total Fit (Sum)', line=dict(color='black', width=1, dash='dot')))

                    # Calculate Dynamic ROI Bounds for visualization
                    center_idx = int(np.round(popt[2]))
                    center_idx = max(0, min(len(norm_profile)-1, center_idx))
                    left_idx, right_idx = 0, len(norm_profile) - 1
                    
                    # Search Left
                    for i in range(center_idx, -1, -1):
                        if norm_profile[i] <= broad_curve[i]:
                            left_idx = i; break
                    # Search Right
                    for i in range(center_idx, len(norm_profile)):
                        if norm_profile[i] <= broad_curve[i]:
                            right_idx = i; break

                    # 2. Raw Data (Line only)
                    fig_prof.add_trace(go.Scatter(x=norm_profile, y=y_pixels, mode='lines', name='Raw Data', line=dict(color='#1f77b4', width=2)))

                    # 3. Dynamic Signal Region (Shaded Blue)
                    # Construct a polygon for the area between Raw and Wall ONLY within the bounds
                    if right_idx > left_idx:
                        y_poly = np.concatenate([y_pixels[left_idx:right_idx+1], y_pixels[right_idx:left_idx-1:-1]])
                        x_poly = np.concatenate([norm_profile[left_idx:right_idx+1], broad_curve[right_idx:left_idx-1:-1]])
                        fig_prof.add_trace(go.Scatter(x=x_poly, y=y_poly, fill='toself', fillcolor='rgba(31, 119, 180, 0.4)', line=dict(width=0), name='Signal (Dynamic ROI)', showlegend=True))
                        
                        # Add markers for the dynamic bounds
                        fig_prof.add_hline(y=left_idx, line_color="#FF4500", line_width=1, line_dash="dash", annotation_text="Dyn Top")
                        fig_prof.add_hline(y=right_idx, line_color="#FF4500", line_width=1, line_dash="dash", annotation_text="Dyn Bot")
                    
                    # Calculate analytic areas
                    real_signal_sum = np.sum(norm_profile[left_idx:right_idx+1] - broad_curve[left_idx:right_idx+1])
                    broad_area = popt[4] * popt[6] * np.sqrt(2 * np.pi)
                    
                    stats_msg = f"""
                    **Signal (Data - Wall):** Sum = `{real_signal_sum:.2e}` (σ_fit={popt[3]:.1f})  
                    **Wall (Broad Fit):** Area = `{broad_area:.2e}` (σ_fit={popt[6]:.1f})
                    """
                else:
                    # Fallback if fit fails: Just plot raw profile
                    fig_prof.add_trace(go.Scatter(x=norm_profile, y=np.arange(len(norm_profile)), mode='lines', name='Raw Profile', line=dict(color='#1f77b4', width=2), fill='tozerox', fillcolor='rgba(31, 119, 180, 0.1)'))

                # Plot the specific ROI intersection points for this column
                title_suffix = ""
                if y_min is not None and y_max is not None:
                    # Fill the Integrated Region (Area between ROI bounds) - Exact pixels
                    y_idx_min, y_idx_max = int(np.ceil(max(0, y_min))), int(np.floor(min(len(norm_profile)-1, y_max)))
                    
                    integrated_sum = 0.0
                    if y_idx_max > y_idx_min:
                        sub_y = np.arange(y_idx_min, y_idx_max+1)
                        sub_x = norm_profile[y_idx_min:y_idx_max+1]
                        integrated_sum = np.sum(sub_x)

                    # Draw Bounds
                    # Only show green ROI lines if we are NOT in dynamic mode, or just as faint reference
                    for y_r, lbl in [(y_min, "ROI Top"), (y_max, "ROI Bot")]:
                        fig_prof.add_hline(y=y_r, line_color="#2ca02c", line_width=1, line_dash="dash", opacity=0.5)
                        
                    title_suffix = f" | Sum: {integrated_sum:.2e}"

                elif not (roi_paths and 'roi_top' in roi_paths):
                    fig_prof.add_annotation(x=x_max/2, y=len(norm_profile)/2, text="No ROI Data", showarrow=False, font=dict(color="red", size=16))

                fig_prof.update_layout(title=f"Signal Decomposition ({key_name} @ Col {col_idx}){title_suffix}", xaxis_title="Intensity (Counts)", yaxis_title="Vertical Pixel (Y)", yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, x_max], exponentformat='e'), height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=80, b=20))
                st.plotly_chart(fig_prof, use_container_width=True, key=f"prof_plot_{key_name}_{col_idx}")
                
                if stats_msg:
                    st.info(stats_msg)
            except Exception as e:
                st.warning(f"Could not generate profile plot: {e}")

@st.cache_data
def load_cached_image(path):
    return np.array(Image.open(path))

@st.fragment
def render_custom_image_view(sel_iter, bg_on, log_thresh, roi_paths):
    """Renders images using a tabbed interface with consolidated controls."""
    if sel_iter not in st.session_state.measurements: return
    
    # 0. Check if raw data actually exists
    # If no measurements, show warning and continue (don't return, as we want to keep the UI structure)
    if sel_iter not in st.session_state.measurements:
        st.warning("⚠️ **Raw Measurement Data Unavailable**")
        st.info("No raw images found for this iteration. Plots below are from pre-computed results.")
        return

    meas_dict = st.session_state.measurements[sel_iter]
    bg_dict = st.session_state.backgrounds.get(sel_iter, {}) if bg_on else {}
    
    first_path = next(iter(meas_dict.values()))
    if not os.path.exists(first_path):
        st.warning("⚠️ **Raw Measurement Data Unavailable**")
        st.info("You are currently in **Local-Only Mode**. Matrix plots and ratios are generated from pre-computed results, but the original source images are not available on this machine.")
        return

    # 1. Load first image to get dimensions for slider
    first_key = next(iter(meas_dict))
    try:
        # Use cached loader
        img_arr = load_cached_image(meas_dict[first_key])
        pil_img = Image.fromarray(img_arr) # Lightweight wrapper if needed, or just use shape
        H, W = img_arr.shape # Numpy is (H, W)
    except: return

    # 2. Controls
    k_col = f"col_idx_{sel_iter}"
    if k_col not in st.session_state: st.session_state[k_col] = W//2
    
    # --- PRE-RENDER EVENT HANDLING ---
    # We must do this BEFORE rendering the slider (k_col widget) to avoid "Session State Modification" errors.
    keys = list(meas_dict.keys())
    for k in keys:
        c_key = f"img_{k}_{sel_iter}_{bg_on}_v5"
        if c_key in st.session_state and st.session_state[c_key]:
            event_data = st.session_state[c_key]
            if "selection" in event_data:
                sel = event_data["selection"]
                clicked_x = None
                if "points" in sel and sel["points"]:
                    clicked_x = int(sel["points"][0]["x"])
                
                if clicked_x is not None:
                    # Update the slider state if changed
                    if st.session_state.get(k_col) != clicked_x:
                        st.session_state[k_col] = clicked_x
                        st.toast(f"Jumped to column {clicked_x}")

    # Layout: Controls Row
    c1, c2, c3, c4 = st.columns([0.15, 0.25, 0.35, 0.25])
    
    with c1:
        show_roi = st.toggle("Show ROI", value=True, key=f"show_roi_{sel_iter}")
    with c2:
        show_decomp = st.toggle("Show Decomposition", value=False, key=f"show_decomp_{sel_iter}")
    with c3:
        # Colormap settings in an expander or just inline if space permits
        cmap_name = st.selectbox("Colormap", ["Viridis", "Plasma", "Inferno", "Magma", "Greys", "YlGnBu", "Hot"], index=1, key=f"cmap_{sel_iter}", label_visibility="collapsed")
    with c4:
        zero_mode = st.radio("Zero", ["Default", "Black", "White"], index=2, horizontal=True, key=f"zmode_{sel_iter}", label_visibility="collapsed")
        # implicit fix_zero=True for simplicity or add back if needed

    # Global Slider
    col_idx = st.slider("Column Position (Linked)", 0, W-1, key=k_col)

    # 3. Tabs for each Image
    # keys definitions moved up
    tabs = st.tabs(keys)

    for i, k in enumerate(keys):
        with tabs[i]:
            img_path = meas_dict[k]
            try:
                # Load & Process Image
                img_arr = load_cached_image(img_path)
                bit_depth = 65535.0 if img_arr.dtype == np.uint16 else 255.0
                img_float = img_arr.astype(float)
                
                if bg_on and k in bg_dict:
                    bg_img = load_cached_image(bg_dict[k]).astype(float)
                    img_float = img_float - bg_img
                    
                    
                
                    
                z_min_val = 0
                # Plot Heatmap using centralized plotting utility
                fig = plotting.create_heatmap_figure(
                    img_float,
                    cmap_name=cmap_name,
                    zero_mode=zero_mode,
                    zmin=z_min_val,
                    title=f"{k} Heatmap"
                )
                
                # Add Interaction Layers (Click Capture & ROI)
                h, w = img_float.shape
                plotting.add_click_capture_trace(fig, h, w)
                
                if show_roi:
                    plotting.add_roi_traces(fig, roi_paths)
                
                # Add Current Position Indicator
                fig.add_vline(x=col_idx, line_width=2, line_dash="dash", line_color="red", opacity=1.0)

                # Ensure layout is set for interaction
                fig.update_layout(
                    clickmode='event+select',
                    hovermode='x',
                    height=450
                )
                
                # Render chart and capture selection events
                # Force new key to reset component state
                st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    key=f"img_{k}_{sel_iter}_{bg_on}_v5",
                    on_select="rerun",           
                    selection_mode="points",
                    config={'displayModeBar': True} # Show modebar so user can see tools
                )
                
                # Debug Version
                # st.sidebar.caption(f"Streamlit: {st.__version__}")
                
                 # Optional: Decomposition Plot
                if show_decomp:
                    with st.spinner("Computing signal decomposition..."):
                        render_signal_decomposition(img_float, k, col_idx, roi_paths, bit_depth)
                    
            except Exception as e:
                st.error(f"Error render image {k}: {e}")

# --- Sidebar ---
st.header("HACP Visualization")
app_mode = st.sidebar.radio("App Mode", ["Data Dashboard", "Interactive Analysis", "Figure Builder"], key="app_mode")
st.sidebar.divider()

# --- Global Dataset Synchronizer ---
p_state = st.session_state.get('p_state', {})
active_meas_path = p_state.get('meas_path')
active_bg_path = p_state.get('bg_path')
active_desc = p_state.get('description', "Unknown Dataset")
active_seq = p_state.get('meas_seq', "Unknown")

if app_mode in ["Interactive Analysis", "Figure Builder"]:
    st.sidebar.markdown(f"**Current Dataset:**\n### {active_desc}")
    st.sidebar.caption(f"ID: `{active_seq}`")
    
    is_local_only = active_meas_path and "LOCAL_ONLY:" in active_meas_path
    
    if active_meas_path and (os.path.exists(active_meas_path) or is_local_only):
        if is_local_only:
             st.sidebar.warning("📁 **Local-Only Mode**")
             st.sidebar.caption("Using cached results. Raw images unavailable.")
        else:
             st.sidebar.success(f"Raw Data Linked")
        
        st.session_state.current_meas_path = active_meas_path
        
        # 3. Load Data Logic (Headless)
        scan_key = f"{active_meas_path}_{active_bg_path}"
        if st.session_state.get('last_scan_key') != scan_key:
            # Clear stale data to prevent confusion if loading fails
            st.session_state.measurements = {}
            st.session_state.backgrounds = {}
            st.session_state.iterations = [] 
            
            with st.spinner("Loading measurement data..."):
                m, b, err = polarimeter_processing.find_and_organize_files(active_meas_path, active_bg_path)
                if not m and not is_local_only: 
                    m, b, err = app_utils.robust_find_and_organize_files(active_meas_path, active_bg_path)
                
                if m:
                    st.session_state.measurements, st.session_state.backgrounds = m, b
                    st.session_state.iterations = sorted(m.keys())
                    st.session_state.last_scan_key = scan_key
                    if st.session_state.iterations:
                        st.session_state.selected_iter_option = st.session_state.iterations[0]
                        # Auto-detect type
                        first_keys = set(m[st.session_state.iterations[0]].keys())
                        if not {'I_PP', 'I_PM', 'I_RP', 'I_RM', 'I_PL', 'I_PR'}.isdisjoint(first_keys):
                            st.session_state.analysis_type = "Mueller Matrix"
                        elif {'Depol_Parallel', 'Depol_Cross', 'Parallel', 'Cross'}.intersection(first_keys):
                            st.session_state.analysis_type = "Depolarization Ratio"
                elif is_local_only:
                    st.session_state.last_scan_key = scan_key
                else:
                    st.session_state.raw_load_error = err

        # 4. Load Precomputed Data (Initial/Fresh Load)
        if st.session_state.get('loaded_meas_path') != active_meas_path or (is_local_only and not st.session_state.iterations):
            # Determine initial load mode based on current mode
            if app_mode == 'Figure Builder':
                use_wall_init = st.session_state.get('global_subtract_wall', True)
            else:
                use_wall_init = st.session_state.get('subtract_wall', False)
            
            load_mode = "dynamic" if use_wall_init else "standard"
            pre_data, pre_meta, pre_fmt = app_utils.try_load_precomputed(active_meas_path, LOCAL_CACHE_ROOT, mode=load_mode)
            
            # Fallback: If Dynamic was requested but not found, try Standard
            if not pre_data and load_mode == "dynamic":
                # st.warning("Dynamic (Wall Subtracted) data not found. Falling back to Standard.")
                load_mode = "standard"
                pre_data, pre_meta, pre_fmt = app_utils.try_load_precomputed(active_meas_path, LOCAL_CACHE_ROOT, mode=load_mode)
            
            if pre_data:
                valid_keys = set(str(i) for i in st.session_state.iterations)
                if not valid_keys: # Local only recovery
                    st.session_state.iterations = sorted([int(k) if str(k).isdigit() else k for k in pre_data.keys()])
                    filtered_pre = pre_data
                else:
                    filtered_pre = {k: v for k, v in pre_data.items() if k in valid_keys }
                
                st.session_state.precomputed_data = filtered_pre
                st.session_state.precomputed_fmt = pre_fmt
                st.session_state.precomputed_meta = pre_meta
                st.session_state.loaded_mode = load_mode
                st.session_state.loaded_meas_path = active_meas_path
                
                # Set type BEFORE widget
                st.session_state.analysis_type = pre_meta.get('analysis_type', st.session_state.get('analysis_type', 'Mueller Matrix'))
                
                if not st.session_state.selected_iter_option in (["Average All"] + st.session_state.iterations):
                    if st.session_state.iterations:
                        st.session_state.selected_iter_option = st.session_state.iterations[0]
            else:
                # If both failed, then we truly have nothing
                st.session_state.precomputed_data = None
                st.session_state.loaded_mode = None

        # 5. Smart Cache Reload (Toggle Detection)
        if st.session_state.get('precomputed_data'):
            if app_mode == "Figure Builder":
                target_wall = st.session_state.get('global_subtract_wall', True)
            elif app_mode == "Interactive Analysis":
                target_wall = st.session_state.get('subtract_wall', False)
            else:
                target_wall = None
            
            if target_wall is not None:
                target_mode = "dynamic" if target_wall else "standard"
                current_mode = st.session_state.get('loaded_mode', 'unknown')
                
                if current_mode != target_mode:
                    new_data, new_meta, new_fmt = app_utils.try_load_precomputed(active_meas_path, LOCAL_CACHE_ROOT, mode=target_mode)
                    if new_data:
                        st.session_state.precomputed_data = new_data
                        st.session_state.precomputed_meta = new_meta
                        st.session_state.precomputed_fmt = new_fmt
                        st.session_state.loaded_mode = target_mode
                        if not st.session_state.iterations:
                            st.session_state.iterations = sorted([int(k) if str(k).isdigit() else k for k in new_data.keys()])
                    else:
                        st.warning(f"Note: Could not load '{target_mode}' data (Wall Subtraction {'On' if target_wall else 'Off'}). Keeping current data.")
        st.session_state.loaded_meas_path = active_meas_path

if app_mode == "Data Dashboard":
    st.sidebar.title("Configuration")
    
    # 1. Raw Data Path
    base_data_path = config_dict.get('last_path', '')
    data_folder = st.sidebar.text_input("Raw Data Root Path", value=base_data_path, help="Path to the remote server or local folder containing raw measurement sequences.")
    
    # 2. Processed Data Path (Cache)
    cache_folder = st.sidebar.text_input("Processed Data Path", value=LOCAL_CACHE_ROOT, help="Local directory where preprocessed NetCDF/JSON results are stored.")
    
    # Save if changed
    if (data_folder != base_data_path) or (cache_folder != LOCAL_CACHE_ROOT):
         new_config = {"last_path": data_folder, "cache_path": cache_folder}
         app_utils.save_config(CONFIG_FILE, new_config)
         # We don't necessarily need to rerun, but it ensures LOCAL_CACHE_ROOT is updated globally
         st.rerun()

    st.sidebar.divider()
    st.sidebar.info("Use the main dashboard panel to manage datasets and processing.")

elif app_mode == "Interactive Analysis":
    st.sidebar.title("Analysis Setup")
    
    # --- Sidebar Controls ---
    st.sidebar.title("Analysis Controls")
    
    # 1. Analysis Type (Primary Filter)
    st.sidebar.selectbox("Analysis Type", ["Mueller Matrix", "Depolarization Ratio"], key="analysis_type")
    st.sidebar.divider()

    # 2. Dataset Quick-Switcher (Filtered by Analysis Type)
    cached_datasets = app_utils.scan_local_cache(LOCAL_CACHE_ROOT)
    if cached_datasets:
        # Filter cached datasets by selected analysis type
        target_type = st.session_state.analysis_type
        filtered_cached = [d for d in cached_datasets if app_utils.get_analysis_type_from_sequence(d['Sequence']) == target_type]
        
        if filtered_cached:
            # Human-readable options
            cached_opts = [f"{d['Date']} | {app_utils.translate_sequence_name(d['Sequence'])} ({d['Sequence']})" for d in filtered_cached]
            
            # Find current in filtered
            curr_idx = 0
            if active_seq:
                for i, d in enumerate(filtered_cached):
                    if d['Sequence'] == active_seq:
                        curr_idx = i
                        break
            
            sel_switch = st.sidebar.selectbox("Jump to Dataset", cached_opts, index=curr_idx, key="sidebar_switcher")
            
            if sel_switch:
                target_d = filtered_cached[cached_opts.index(sel_switch)]
                if active_seq != target_d['Sequence']:
                    # Reset comparisons and other state
                    st.session_state.comparisons = []
                    # trigger reload by setting target_load
                    st.session_state.target_load = {
                        "date": target_d['Date'],
                        "meas_parent": os.path.basename(os.path.dirname(os.path.dirname(target_d['LocalPath']))),
                        "meas_seq": target_d['Sequence'],
                        "meas_path": target_d['LocalPath'],
                        "description": app_utils.translate_sequence_name(target_d['Sequence'])
                    }
                    st.rerun()
        else:
            st.sidebar.info(f"No cached {target_type} datasets.")
    
    st.sidebar.divider()
    
    if active_meas_path and (os.path.exists(active_meas_path) or is_local_only):
        if st.session_state.get('precomputed_data'):
            fmt = st.session_state.get('precomputed_fmt', 'Data')
            st.success(f"Pre-computed results loaded: {len(st.session_state.precomputed_data)} steps.")
            
            # Check if we should mention raw data failure
            if st.session_state.get('raw_load_error') and not is_local_only:
                st.info(f"💡 **Limited Mode**: Raw data unavailable ({st.session_state.raw_load_error}). Viewing cached results only.")
            
            st.session_state.use_precomputed = st.checkbox("Use pre-computed results", value=True)
        elif st.session_state.get('raw_load_error'):
            # Only show as red error if we have NO precomputed data either
            st.error(f"❌ **Loading Failed**: {st.session_state.raw_load_error}")
        elif not is_local_only and not st.session_state.get('measurements'):
             st.warning("⚠️ No data found at the specified path.")
    else:
        st.sidebar.warning("No dataset loaded.")
        if st.sidebar.button("Go to Dashboard", on_click=set_app_mode, args=("Data Dashboard",)):
            st.rerun()

    # Comparisons
    with st.sidebar.expander("Comparisons"):
        all_cached = app_utils.scan_local_cache(LOCAL_CACHE_ROOT)
        
        # Filter cached datasets by CURRENT SELECTED analysis type
        target_type = st.session_state.analysis_type
        comp_cached = [d for d in all_cached if app_utils.get_analysis_type_from_sequence(d['Sequence']) == target_type]
            
        if comp_cached:
            cached_opts = [f"{d['Date']} | {app_utils.translate_sequence_name(d['Sequence'])} ({d['Sequence']})" for d in comp_cached]
            sel_cached = st.selectbox("Select Dataset to Contrast", cached_opts, key="interactive_comp_cache_sel")
            
            if st.button("Add as Comparison", use_container_width=True):
                d_idx = cached_opts.index(sel_cached)
                target_d = comp_cached[d_idx]
                
                # Try to load precomputed data
                pre_data, _, _ = app_utils.try_load_precomputed(target_d['LocalPath'])
                if pre_data:
                    # Avoid duplicates
                    existing_names = [c['name'] for c in st.session_state.comparisons]
                    if target_d['Sequence'] not in existing_names:
                        desc = app_utils.translate_sequence_name(target_d['Sequence'])
                        st.session_state.comparisons.append({
                            'name': target_d['Sequence'],
                            'description': desc,
                            'precomputed_data': pre_data
                        })
                        st.success(f"Added {target_d['Sequence']}")
                        st.rerun()
                    else:
                        st.warning("Comparison already added.")
                else:
                    st.error("Failed to load preprocessed data.")
        else:
            st.info(f"No matching {target_type} datasets found.")

        if st.session_state.comparisons:
            st.divider()
            st.write(f"**Active Comparisons ({len(st.session_state.comparisons)})**")
            for i, comp in enumerate(st.session_state.comparisons):
                st.caption(f"• {comp['name']}")
            
            if st.button("Clear All Comparisons", type="secondary", use_container_width=True):
                st.session_state.comparisons = []
                st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("Back to Dashboard", icon="🏠", on_click=set_app_mode, args=("Data Dashboard",)):
        st.rerun()

# --- Main Panel ---
if app_mode == "Data Dashboard":
    dashboard.render_dashboard(base_data_path, LOCAL_CACHE_ROOT)
    st.stop()

if app_mode == "Figure Builder":
    # --- Safety Net for Local-Only Iteration Recovery ---
    if not st.session_state.get('iterations') and st.session_state.get('precomputed_data'):
         pre = st.session_state.precomputed_data
         st.session_state.iterations = sorted([int(k) if str(k).isdigit() else k for k in pre.keys()])

    from utils import figure_composer
    figure_composer.render_figure_composer(st.session_state.measurements, LOCAL_CACHE_ROOT)
    st.stop()

# --- Safety Net for Local-Only Iteration Recovery ---
# If iterations were cleared (e.g. by scan logic) but we have precomputed data, recover them now.
# This MUST happen before the `if ... iterations:` check below.
if not st.session_state.get('iterations') and st.session_state.get('precomputed_data'):
     pre = st.session_state.precomputed_data
     st.session_state.iterations = sorted([int(k) if str(k).isdigit() else k for k in pre.keys()])
     if st.session_state.iterations:
         # Ensure a valid selection exists
         # formatting matches the loop below
         current_sel = st.session_state.get('selected_iter_option')
         
         # If current selection is invalid or missing, default to the first iteration
         # But we must format it exactly as the widget expects: "Description (ID)"
         # We can't easily check "if current_sel not in options" because options aren't built yet.
         # So we'll check if it's "Average All" or if it LOOKS roughly valid. 
         # Actually, simpler: if it's not set or likely invalid (e.g. integer), set it to the formatted first iter.
         if not current_sel or (isinstance(current_sel, int)) or (isinstance(current_sel, str) and "(" not in current_sel and current_sel != "Average All"):
              first_it = st.session_state.iterations[0]
              desc = app_utils.translate_sequence_name(str(first_it))
              st.session_state.selected_iter_option = f"{desc} ({first_it})"

if 'iterations' in st.session_state and st.session_state.iterations:
    controls = st.container(border=True)
    
    # Row 1: Iteration & Plotting Controls
    r1_c1, r1_c2 = controls.columns([0.5, 0.5])
    
    with r1_c1:

        iter_cols = st.columns([0.2, 0.6, 0.2])
        # Format labels with descriptions
        formatted_labels = []
        for it in st.session_state.iterations:
            desc = app_utils.translate_sequence_name(str(it))
            formatted_labels.append(f"{desc} ({it})")
            
        iter_opts = ["Average All"] + formatted_labels
        sel_iter_label = iter_cols[1].selectbox("Iteration", iter_opts, key="selected_iter_option", label_visibility="collapsed")
        
        # Decode selection
        if sel_iter_label == "Average All":
            sel_iter = "Average All"
        else:
            # Extract ID from "Description (ID)" - use rsplit to handle descriptions with (
            sel_iter = sel_iter_label.rsplit(" (", 1)[-1][:-1]
            
        is_avg = sel_iter == "Average All"
        
        if not is_avg:
            curr_idx = iter_opts.index(sel_iter_label)
            iter_cols[0].button("Previous", use_container_width=True, disabled=(curr_idx <= 1), on_click=go_to_previous)
            iter_cols[2].button("Next", use_container_width=True, disabled=(curr_idx >= len(iter_opts) - 1), on_click=go_to_next)
            
    with r1_c2:
        scale_cols = st.columns(2)
        y_scale = scale_cols[0].radio("Y-Axis", ["Log", "Linear", "Symlog"], horizontal=True, index=1, help="Scale for the Y-axis of the plots.")
        match_intensity = False
        if st.session_state.comparisons:
            match_intensity = st.toggle("Scale Comparisons", help="Match peak intensity")
            
    # Row 2: Parameters
    p_cols = controls.columns(4)
    noise_sigma = p_cols[0].slider("Noise (Sigma)", 1.0, 10.0, 3.0, 0.5, key="noise_sigma")
    log_thresh = p_cols[1].slider("Signal Thresh", 0.1, 4.0, 0.6, 0.1, key="log_thresh")
    bg_on = p_cols[2].toggle("Background Subtraction", value=True, key="bg_toggle")
    
    with p_cols[3]:
        subtract_wall = st.toggle("Subtract Wall Effect", value=False, key="subtract_wall")
        show_std = st.toggle("Show Standard Deviation", value=True, key="show_std_toggle")
        # Only show method selector if wall subtraction is on
        wall_method = "dynamic" # Static method is removed.

    # (Smart Reload logic moved to top level for reliability across modes)

    # --- Calculation ---
    final_results, curves_to_show, roi_paths_to_show = {}, {}, {}
    
    # Ensure analysis_type is defined
    analysis_type = st.session_state.get('analysis_type', "Mueller Matrix")
    
    req_meas = app_utils.get_required_measurements(st.session_state.measurements, analysis_type)
    
    # 1. Try Precomputed
    if st.session_state.get('use_precomputed') and st.session_state.get('precomputed_data'):
        pre = st.session_state.precomputed_data
        if is_avg:
            # -- Robust Ratio-of-Means Logic --
            # Helper to average components
            def get_component_mean_std(k_list):
                 stack = [v for v in k_list if v is not None]
                 if not stack: return None, None
                 return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)

            # 1. Collect all raw components
            raw_components = defaultdict(list)
            for d in pre.values():
                for k, v in d.items():
                    raw_components[k].append(v)
            
            # 2. Compute Means/Stds (Ratio-of-Means priority)
            for k in raw_components.keys():
                # Define derived logic
                num_key, den_key = None, None
                if k == "S12/S11": num_key, den_key = "S12", "S11"
                elif k == "S33/S11": num_key, den_key = "S33", "S11"
                elif k == "S34/S11": num_key, den_key = "S34", "S11"
                elif k == "Depolarization Ratio": 
                     num_key = 'Depol_Cross' if 'Depol_Cross' in raw_components else 'Cross'
                     den_key = 'Depol_Parallel' if 'Depol_Parallel' in raw_components else 'Parallel'

                if num_key and den_key and num_key in raw_components and den_key in raw_components:
                    # Calculate Ratio of Means
                    m_num, s_num = get_component_mean_std(raw_components[num_key])
                    m_den, s_den = get_component_mean_std(raw_components[den_key])
                    
                    if m_num is not None and m_den is not None:
                         with np.errstate(divide='ignore', invalid='ignore'):
                             mean_val = m_num / m_den
                             # Error prop for STD
                             # std = |f| * sqrt( (sn/n)^2 + (sd/d)^2 )
                             term1 = (s_num / m_num)**2
                             term2 = (s_den / m_den)**2
                             std_val = np.abs(mean_val) * np.sqrt(term1 + term2)
                         final_results[k] = {'mean': mean_val, 'std': std_val}
                else:
                    # Fallback to standard Mean of Ratios (if components missing)
                    # or standard variable
                    m_val, s_val = get_component_mean_std(raw_components[k])
                    final_results[k] = {'mean': m_val, 'std': s_val}
        else:
            iter_key = str(sel_iter) if str(sel_iter) in pre else sel_iter
            if iter_key in pre:
                raw_res = pre[iter_key]
                for k, v in raw_res.items():
                    final_results[k] = {'mean': v}

    # 2. Calculate if missing
    if not final_results:
        # Check if we HAVE measurements to calculate from
        has_meas = len(st.session_state.get('measurements', {})) > 0
        
        if has_meas:
            if is_avg:
                all_res = defaultdict(list)
                prog = st.progress(0, text="Averaging iterations...")
                for i, it in enumerate(st.session_state.iterations):
                    res_tuple = app_computation.calculate_curves_for_iteration(it, st.session_state.measurements, st.session_state.backgrounds, req_meas, analysis_type, bg_on, noise_sigma, 10**(-log_thresh), angle_model, subtract_wall)
                    if res_tuple:
                        curves, _ = res_tuple
                        if curves:
                            if analysis_type == "Mueller Matrix":
                                m = polarimeter_processing.calculate_mueller_elements(curves)
                                if m: 
                                    for k, v in m.items(): all_res[k].append(v)
                            else:
                                if 'Depol_Parallel' in curves:
                                    dep = (curves['Depol_Cross']+1e-9)/(curves['Depol_Parallel']+1e-9)
                                    all_res['Depolarization Ratio'].append(dep)
                                    all_res['Depol_Parallel'].append(curves['Depol_Parallel'])
                                    all_res['Depol_Cross'].append(curves['Depol_Cross'])
                        prog.progress((i+1)/len(st.session_state.iterations))
                if all_res:
                    # Same robust logic for live calculation
                    def get_comp_stats(k_list):
                         stack = [v for v in k_list if v is not None]
                         if not stack: return None, None
                         return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
                         
                    for k in all_res.keys():
                        num_key, den_key = None, None
                        if k == "S12/S11": num_key, den_key = "S12", "S11"
                        elif k == "S33/S11": num_key, den_key = "S33", "S11"
                        elif k == "S34/S11": num_key, den_key = "S34", "S11"
                        elif k == "Depolarization Ratio": 
                             num_key = 'Depol_Cross' if 'Depol_Cross' in all_res else 'Cross'
                             den_key = 'Depol_Parallel' if 'Depol_Parallel' in all_res else 'Parallel'

                        if num_key and den_key and num_key in all_res and den_key in all_res:
                             m_num, s_num = get_comp_stats(all_res[num_key])
                             m_den, s_den = get_comp_stats(all_res[den_key])
                             if m_num is not None and m_den is not None:
                                  with np.errstate(divide='ignore', invalid='ignore'):
                                      mean_val = m_num / m_den
                                      term1 = (s_num / m_num)**2
                                      term2 = (s_den / m_den)**2
                                      std_val = np.abs(mean_val) * np.sqrt(term1 + term2)
                                  final_results[k] = {'mean': mean_val, 'std': std_val}
                        else:
                             m_val, s_val = get_comp_stats(all_res[k])
                             final_results[k] = {'mean': m_val, 'std': s_val}
            else:
                res_tuple = app_computation.calculate_curves_for_iteration(sel_iter, st.session_state.measurements, st.session_state.backgrounds, req_meas, analysis_type, bg_on, noise_sigma, 10**(-log_thresh), angle_model, subtract_wall)
                if res_tuple:
                    curves, roi_paths = res_tuple
                    roi_paths_to_show = roi_paths
                    curves_to_show = curves
                    res = {}
                    if analysis_type == "Mueller Matrix":
                        m = polarimeter_processing.calculate_mueller_elements(curves)
                        if m: res = m
                    else:
                        if 'Depol_Parallel' in curves:
                            res['Depolarization Ratio'] = (curves['Depol_Cross']+1e-9)/(curves['Depol_Parallel']+1e-9)
                            res['Depol_Parallel'] = curves['Depol_Parallel']
                            res['Depol_Cross'] = curves['Depol_Cross']
                    final_results = res
                    
                    # Auto-save newly computed single iteration
                    if st.session_state.get('precomputed_data') is None: st.session_state.precomputed_data = {}
                    st.session_state.precomputed_data[str(sel_iter)] = res
                    meta = {"source_path": st.session_state.current_meas_path, "analysis_type": analysis_type, "parameters": {"bg_subtraction": bg_on, "subtract_wall": subtract_wall}}
                    export_fmt = st.session_state.get('export_fmt', 'NetCDF')
                    app_utils.save_precomputed_data(st.session_state.current_meas_path, st.session_state.precomputed_data, meta, fmt=export_fmt, local_cache_root=LOCAL_CACHE_ROOT, is_dynamic=subtract_wall)
        else:
            if not st.session_state.get('precomputed_data'):
                st.warning("⚠️ **No data to display.**")
                st.info("Raw measurement folders are missing or unreachable, and no pre-computed results were found for this dataset.")
                st.stop()
            else:
                # Precomputed data exists, but we failed to extract results for THIS iteration/settings
                if not final_results:
                     avail_keys = list(st.session_state.precomputed_data.keys())
                     if subtract_wall:
                         st.warning(f"⚠️ **Wall-Subtracted Data Unavailable for {sel_iter}**")
                         st.info(f"The 'Subtract Wall Effect' dataset is loaded, but it does not contain data for iteration **{sel_iter}**.")
                         st.caption(f"**Diagnostic Info:**\nRequested: `{sel_iter}` (Type: {type(sel_iter).__name__})\nAvailable in File: `{avail_keys}`")
                         st.markdown("👉 **Try disabling 'Subtract Wall Effect'** to see the standard processed data.")
                     else:
                         st.warning(f"⚠️ **Data Unavailable for {sel_iter}**")
                         st.info(f"Pre-computed results are loaded, but no data was found for iteration **{sel_iter}**. It might have been skipped during processing.")
                         st.caption(f"**Diagnostic Info:**\nRequested: `{sel_iter}`\nAvailable: `{avail_keys}`")
                     st.stop()

                st.info("💡 Pro-tip: Ensure 'Use pre-computed results' is checked if raw images are missing.")
                # We already loaded precomputed data into final_results in Step 1 if available


    # --- Plotting ---
    if final_results:
        ref_w = list(final_results.values())[0]['mean'].shape[0]
        x_angles = angle_model(np.arange(ref_w) + PIXEL_OFFSET_X)
        
        # Angle Selection (Synchronized Slider + Numeric Inputs)
        if 'angle_range' not in st.session_state: st.session_state.angle_range = (100.0, 167.0)
        curr_low, curr_high = st.session_state.angle_range
        min_a, max_a = float(x_angles.min()), float(x_angles.max())

        # Callbacks for sync
        def _sync_from_slider_ia():
            low, high = st.session_state.angle_range_slider_ia
            st.session_state.angle_min_ia = int(low)
            st.session_state.angle_max_ia = int(high)
            st.session_state.angle_range = (float(low), float(high))

        def _sync_from_inputs_ia():
            low = st.session_state.angle_min_ia
            high = st.session_state.angle_max_ia
            st.session_state.angle_range_slider_ia = (float(low), float(high))
            st.session_state.angle_range = (float(low), float(high))

        # Ensure widget keys are in sync with global state and CLAMPED to current data bounds
        # Use floor/ceil or float clamping to avoid "value below min" due to int truncation
        clamped_low = max(min_a, min(max_a, float(curr_low)))
        clamped_high = max(min_a, min(max_a, float(curr_high)))
        
        if st.session_state.get('angle_min_ia') != int(clamped_low): 
            st.session_state.angle_min_ia = int(np.ceil(clamped_low)) if clamped_low > int(clamped_low) else int(clamped_low)
        if st.session_state.get('angle_max_ia') != int(clamped_high): 
            st.session_state.angle_max_ia = int(clamped_high)
        
        # Slider MUST be floats within [min_a, max_a]
        s_low = max(min_a, float(st.session_state.angle_min_ia))
        s_high = min(max_a, float(st.session_state.angle_max_ia))
        
        if st.session_state.get('angle_range_slider_ia') != (s_low, s_high):
            st.session_state.angle_range_slider_ia = (s_low, s_high)

        ac1, ac2, ac3 = st.columns([0.4, 0.3, 0.3])
        ac2.number_input("Min Angle [deg]", int(min_a), int(max_a), step=1, key="angle_min_ia", on_change=_sync_from_inputs_ia)
        ac3.number_input("Max Angle [deg]", int(min_a), int(max_a), step=1, key="angle_max_ia", on_change=_sync_from_inputs_ia)
        ac1.slider("Angle Range", min_a, max_a, key="angle_range_slider_ia", on_change=_sync_from_slider_ia, help="Range of scattering angles to display.")
        
        # Final sync
        st.session_state.angle_range = (float(st.session_state.angle_min_ia), float(st.session_state.angle_max_ia))
        
        def get_yrange(y_data_list):
            mask = (x_angles >= st.session_state.angle_range[0]) & (x_angles <= st.session_state.angle_range[1])
            vis_values = []
            for y in y_data_list:
                if len(y) == len(mask):
                    vis_values.append(y[mask])
            if not vis_values: return None
            vis = np.concatenate(vis_values)
            if vis.size == 0 or np.all(np.isnan(vis)): return None
            mn, mx = np.nanmin(vis), np.nanmax(vis)
            pad = (mx - mn) * 0.1
            if pad == 0: pad = 0.1 * abs(mx) if mx != 0 else 1.0
            return [mn - pad, mx + pad]

        # Define consistent color palette for comparisons
        comp_colors = plotly.colors.qualitative.Dark24

        if analysis_type == "Mueller Matrix":
            t1, t2, t3, t4 = st.tabs(["Matrix", "|S₁|² & |S₂|²", "Images", "Background Images"])
            with t1:
                configs = {"S11": "#1f77b4", "DoLP": "#1f77b4", "S12/S11": "#1f77b4", "S33/S11": "#1f77b4", "S34/S11": "#1f77b4"}
                for key, color in configs.items():
                    if key in final_results:
                        mean = final_results[key]['mean']
                        
                        # Add Standard Deviation if requested
                        y_err_primary = None
                        if show_std and is_avg and 'std' in final_results[key]:
                             y_err_primary = final_results[key]['std']
                             
                        fig = plotting.create_line_figure(
                            x_angles, mean, y_err=y_err_primary, 
                            name='Primary', color=color,
                            style=plotting.STYLES["Default"],
                            y_log=(y_scale == "Log" and key == 'S11')
                        )
                        
                        y_stack = [mean]
                        
                        for i, comp in enumerate(st.session_state.comparisons):
                            # Logic to get comparison data (precomputed or calculated)
                            c_res = None
                            if comp.get('precomputed_data'):
                                d = comp['precomputed_data']
                                if is_avg:
                                    vals = [v[key] for v in d.values() if key in v]
                                    if vals: c_res = np.mean(vals, axis=0)
                                else:
                                    c_res = d.get(str(sel_iter), {}).get(key)
                            
                            if c_res is not None:
                                c_plot = c_res.copy()
                                suffix = ""
                                if match_intensity:
                                    p_max, c_max = np.nanmax(mean), np.nanmax(c_plot)
                                    if c_max != 0: 
                                        c_plot *= (p_max/c_max)
                                        suffix = f" (x{p_max/c_max:.2f})"
                                label = comp.get('description', app_utils.get_smart_label(comp['name'], st.session_state.measurements, comp.get('measurements')))
                                color_c = comp_colors[i % len(comp_colors)]
                                fig.add_trace(go.Scatter(x=x_angles, y=c_plot, mode='lines', line=dict(color=color_c), name=f"{label}{suffix}"))
                                y_stack.append(c_plot)

                        # Primary already added by create_line_figure
                        
                        if y_scale == "Log" and key == 'S11':
                            fig.update_yaxes(type="log", exponentformat="e")
                        else:
                            fig.update_yaxes(range=get_yrange(y_stack), exponentformat="e")
                        
                        fig.update_layout(title=key, xaxis_title="Angle")
                        fig.update_xaxes(range=st.session_state.angle_range)
                        st.plotly_chart(fig, use_container_width=True)

            with t2:
                if 'S11' in final_results and 'S12' in final_results:
                    s11 = final_results['S11']['mean']
                    s12 = final_results['S12']['mean']
                    i_pv = (s11 - s12) / 2.0
                    i_ph = (s11 + s12) / 2.0
                    
                    for plot_label, data, color in [("|S₁|² (I_PV)", i_pv, "#1f77b4"), ("|S₂|² (I_PH)", i_ph, "#1f77b4")]:
                        fig = go.Figure()
                        y_stack = [data]
                        
                        for i, comp in enumerate(st.session_state.comparisons):
                            c_res = None
                            if comp.get('precomputed_data'):
                                d = comp['precomputed_data']
                                if is_avg:
                                    vals_s11 = [v['S11'] for v in d.values() if 'S11' in v]
                                    vals_s12 = [v['S12'] for v in d.values() if 'S12' in v]
                                    if vals_s11 and vals_s12:
                                        m_s11 = np.mean(vals_s11, axis=0)
                                        m_s12 = np.mean(vals_s12, axis=0)
                                        c_res = (m_s11 - m_s12)/2.0 if "PV" in plot_label else (m_s11 + m_s12)/2.0
                                else:
                                    it_d = d.get(str(sel_iter), {})
                                    if 'S11' in it_d and 'S12' in it_d:
                                        c_res = (it_d['S11'] - it_d['S12'])/2.0 if "PV" in plot_label else (it_d['S11'] + it_d['S12'])/2.0
                            
                            # Note: Comparisons don't have shaded Std for now to keep it clean.
                            if c_res is not None:
                                c_plot = c_res.copy()
                                suffix = ""
                                if match_intensity:
                                    p_max, c_max = np.nanmax(data), np.nanmax(c_plot)
                                    if c_max != 0:
                                        c_plot *= (p_max/c_max)
                                        suffix = f" (x{p_max/c_max:.2f})"
                                label = comp.get('description', app_utils.get_smart_label(comp['name'], st.session_state.measurements, comp.get('measurements')))
                                color_c = comp_colors[i % len(comp_colors)]
                                fig.add_trace(go.Scatter(x=x_angles, y=c_plot, mode='lines', line=dict(color=color_c), name=f"{label}{suffix}"))
                                y_stack.append(c_plot)

                        # Primary with Shading
                        y_err = None
                        if show_std and is_avg:
                            # Derived PV/PH std is a bit complex, let's approximate or just skip if too heavy.
                            # But wait, final_results only has Sij.
                            # Let's Skip shading for derived PV/PH for now to be safe, or implement if easy.
                            pass

                        fig.add_trace(go.Scatter(x=x_angles, y=data, mode='lines', line=dict(color=color), name='Primary'))
                        
                        if y_scale == "Log":
                            fig.update_yaxes(type="log", exponentformat="e")
                        else:
                            fig.update_yaxes(range=get_yrange(y_stack), exponentformat="e")
                        
                        fig.update_layout(title=plot_label, xaxis_title="Angle")
                        fig.update_xaxes(range=st.session_state.angle_range)
                        st.plotly_chart(fig, use_container_width=True)

            with t3:
                if not is_avg:
                    render_custom_image_view(sel_iter, bg_on, log_thresh, roi_paths_to_show)
                else:
                    app_visualization.render_raw_images_tab(st.session_state.measurements, st.session_state.backgrounds, st.session_state.comparisons, sel_iter, is_avg, req_meas, bg_on, roi_paths_to_show)

            with t4:
                app_visualization.render_background_images_tab(st.session_state.backgrounds, st.session_state.comparisons, req_meas, bg_on)

        else: # Depolarization Ratio
            t1, t2, t3 = st.tabs(["Ratio", "Images", "Background Images"])
            with t1:
                if 'Depolarization Ratio' in final_results:
                    mean = final_results['Depolarization Ratio']['mean']
                    
                    y_err_primary = None
                    if show_std and is_avg and 'std' in final_results['Depolarization Ratio']:
                        y_err_primary = final_results['Depolarization Ratio']['std']

                    fig = plotting.create_line_figure(
                        x_angles, mean, y_err=y_err_primary, 
                        name='Primary', color="#1f77b4",
                        style=plotting.STYLES["Default"],
                        y_log=(y_scale == "Log")
                    )
                    
                    y_stack = [mean]
                    
                    for i, comp in enumerate(st.session_state.comparisons):
                        c_res = None
                        if comp.get('precomputed_data'):
                            d = comp['precomputed_data']
                            if is_avg:
                                vals = [v['Depolarization Ratio'] for v in d.values() if 'Depolarization Ratio' in v]
                                if vals: c_res = np.mean(vals, axis=0)
                            else:
                                c_res = d.get(str(sel_iter), {}).get('Depolarization Ratio')
                        
                        if c_res is not None:
                            c_plot = c_res.copy()
                            suffix = ""
                            if match_intensity:
                                p_max, c_max = np.nanmax(mean), np.nanmax(c_plot)
                                if c_max != 0:
                                    c_plot *= (p_max/c_max)
                                    suffix = f" (x{p_max/c_max:.2f})"
                            label = comp.get('description', app_utils.get_smart_label(comp['name'], st.session_state.measurements, comp.get('measurements')))
                            color_c = comp_colors[i % len(comp_colors)]
                            fig.add_trace(go.Scatter(x=x_angles, y=c_plot, mode='lines', line=dict(color=color_c), name=f"{label}{suffix}"))
                            y_stack.append(c_plot)

                    if y_scale == "Log":
                        fig.update_yaxes(type="log", exponentformat="e")
                    else:
                        fig.update_yaxes(range=get_yrange(y_stack), exponentformat="e")
                    
                    fig.update_layout(title="Depolarization Ratio", xaxis_title="Angle")
                    fig.update_xaxes(range=st.session_state.angle_range)
                    st.plotly_chart(fig, use_container_width=True)
                
                # NEW: Intensity Curves Plot
                if 'Depol_Parallel' in final_results and 'Depol_Cross' in final_results:
                    # Plot Parallel
                    fig_par = go.Figure()
                    mean_par = final_results['Depol_Parallel']['mean']
                    fig_par.add_trace(go.Scatter(x=x_angles, y=mean_par, mode='lines', name='I_Parallel', line=dict(color='#1f77b4')))
                    if y_scale == "Log": 
                        fig_par.update_yaxes(type="log", exponentformat="e")
                    else:
                        fig_par.update_yaxes(exponentformat="e")
                    fig_par.update_layout(title="Raw Intensity: I_Parallel", xaxis_title="Angle", yaxis_title="Counts")
                    fig_par.update_xaxes(range=st.session_state.angle_range)
                    st.plotly_chart(fig_par, use_container_width=True)

                    # Plot Cross
                    fig_cross = go.Figure()
                    mean_cross = final_results['Depol_Cross']['mean']
                    fig_cross.add_trace(go.Scatter(x=x_angles, y=mean_cross, mode='lines', name='I_Cross', line=dict(color='#1f77b4')))
                    if y_scale == "Log": 
                        fig_cross.update_yaxes(type="log", exponentformat="e")
                    else:
                        fig_cross.update_yaxes(exponentformat="e")
                    fig_cross.update_layout(title="Raw Intensity: I_Cross", xaxis_title="Angle", yaxis_title="Counts")
                    fig_cross.update_xaxes(range=st.session_state.angle_range)
                    st.plotly_chart(fig_cross, use_container_width=True)

            with t2:
                if not is_avg:
                    render_custom_image_view(sel_iter, bg_on, log_thresh, roi_paths_to_show)
                else:
                    app_visualization.render_raw_images_tab(st.session_state.measurements, st.session_state.backgrounds, st.session_state.comparisons, sel_iter, is_avg, req_meas, bg_on, roi_paths_to_show)

            with t3:
                app_visualization.render_background_images_tab(st.session_state.backgrounds, st.session_state.comparisons, req_meas, bg_on)
    # DEBUG SECTION (Moved to Bottom)
    st.divider()
    with st.expander("Debug Info (App V2)", expanded=False):
        st.write(f"Subtract Wall: {subtract_wall}")
        st.write(f"Precomputed Data Loaded: {bool(st.session_state.get('precomputed_data'))}")
        if st.session_state.get('precomputed_data'):
            keys = list(st.session_state.precomputed_data.keys())
            st.write(f"Precomputed Keys (First 5): {keys[:5]}")
            st.write(f"Precomputed Keys (Types): {[type(k).__name__ for k in keys[:5]]}")
        st.write(f"Selected Iteration: {sel_iter} (Type: {type(sel_iter).__name__})")
        st.write(f"Final Results Empty?: {not bool(final_results)}")
        st.json(final_results)
