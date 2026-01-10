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
LOCAL_CACHE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../output/IPT_WIND_TUNNEL"))

@st.cache_resource
def load_angle_model():
    return pixel_angle_tool.load_angle_model_from_npz(CALIBRATION_FILE_PATH)

angle_model = load_angle_model()

# --- Initialize Session State ---
if 'comparisons' not in st.session_state:
    st.session_state.comparisons = [] # List of comparison sets
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Data Dashboard" # Default mode
if 'iterations' not in st.session_state:
    st.session_state.iterations = []
if 'selected_iter_option' not in st.session_state:
    st.session_state.selected_iter_option = "Average All"
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
    st.session_state.app_mode = "Interactive Analysis"
    
    # 3. Clear Request & Rerun
    del st.session_state.target_load
    st.rerun()

# ... (Previous helper functions omitted for brevity, they remain unchanged) ...
def go_to_previous():
    if 'iterations' in st.session_state and st.session_state.iterations:
        opts = ["Average All"] + st.session_state.iterations
        curr = st.session_state.selected_iter_option
        if curr in opts:
            idx = opts.index(curr)
            if idx > 1:
                st.session_state.selected_iter_option = opts[idx - 1]

def go_to_next():
    if 'iterations' in st.session_state and st.session_state.iterations:
        opts = ["Average All"] + st.session_state.iterations
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
    
    meas_dict = st.session_state.measurements[sel_iter]
    bg_dict = st.session_state.backgrounds.get(sel_iter, {}) if bg_on else {}
    
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



if app_mode == "Data Dashboard":
    st.sidebar.title("Configuration")
    base_data_path = app_utils.load_config(CONFIG_FILE)
    data_folder = st.sidebar.text_input("Data Root Path", value=base_data_path)
    if data_folder and os.path.isdir(data_folder) and data_folder != base_data_path:
         app_utils.save_config(CONFIG_FILE, data_folder)
    st.sidebar.info("Use the main dashboard panel to manage datasets and processing.")

elif app_mode == "Interactive Analysis":
    st.sidebar.title("Analysis Setup")
    
    # 1. Recover State
    base_data_path = app_utils.load_config(CONFIG_FILE)
    p_state = {}
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f: p_state = json.load(f)
        except: pass
        
    # 2. Check for Active Dataset (from Dashboard or Persistence)
    active_meas_path = p_state.get('meas_path')
    active_bg_path = p_state.get('bg_path')
    
    # If we have a target_load request that hasn't been processed into persistence fully?
    # The top-level target_load handler (already implemented) updates SESSION_FILE.
    
    if active_meas_path and os.path.exists(active_meas_path):
        st.session_state.current_meas_path = active_meas_path
        st.sidebar.success(f"Loaded: `{os.path.basename(active_meas_path)}`")
        
        # 3. Load Data Logic (Headless)
        # We need to run this to ensure st.session_state.measurements is populated
        scan_key = f"{active_meas_path}_{active_bg_path}"
        if st.session_state.get('last_scan_key') != scan_key:
            with st.spinner("Loading measurement data..."):
                m, b, err = polarimeter_processing.find_and_organize_files(active_meas_path, active_bg_path)
                if err or not m: m, b, err = app_utils.robust_find_and_organize_files(active_meas_path, active_bg_path)
                
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
                        elif {'Depol_Parallel', 'Depol_Cross'}.issubset(first_keys):
                            st.session_state.analysis_type = "Depolarization Ratio"
                else:
                    st.error(f"Failed to load data: {err}")

        # 4. Load Precomputed Data (Dynamic/Standard)
        if st.session_state.get('loaded_meas_path') != active_meas_path:
             # Default to standard on first load, or auto
             load_mode = "dynamic" if st.session_state.get('subtract_wall', False) else "standard"
             pre_data, pre_meta, pre_fmt = app_utils.try_load_precomputed(active_meas_path, LOCAL_CACHE_ROOT, mode=load_mode)
             
             if pre_data: # and pre_meta.get('analysis_type') == st.session_state.get('analysis_type', ''):
                 # Loose check on analysis type to allow viewing
                 valid_keys = set(str(i) for i in st.session_state.iterations)
                 filtered_pre = {k: v for k, v in pre_data.items() if k in valid_keys}
                 st.session_state.precomputed_data = filtered_pre
                 st.session_state.precomputed_fmt = pre_fmt
                 st.session_state.analysis_type = pre_meta.get('analysis_type', st.session_state.get('analysis_type', 'Mueller Matrix'))
             else:
                 st.session_state.precomputed_data = None
             st.session_state.loaded_meas_path = active_meas_path

        if st.session_state.get('precomputed_data'):
            fmt = st.session_state.get('precomputed_fmt', 'Data')
            st.success(f"Pre-computed: {len(st.session_state.precomputed_data)} steps")
            st.session_state.use_precomputed = st.checkbox("Use pre-computed results", value=True)

        if st.sidebar.button("Change Dataset", icon="🔄"):
            st.session_state.app_mode = "Data Dashboard"
            st.rerun()

    else:
        st.sidebar.warning("No dataset loaded.")
        if st.sidebar.button("Go to Dashboard"):
            st.session_state.app_mode = "Data Dashboard"
            st.rerun()

    # Analysis Type Override
    st.sidebar.divider()
    curr_type = st.session_state.get('analysis_type', 'Mueller Matrix')
    st.sidebar.selectbox("Analysis Type", ["Mueller Matrix", "Depolarization Ratio"], index=0 if curr_type == "Mueller Matrix" else 1, key="analysis_type")

    # Comparisons (Simplified for now - can re-enable full scan if needed)
    with st.sidebar.expander("Comparisons"):
        st.info("To add comparisons, please ensure the Data Root Path is set correctly.")
        # Logic to scan folders for comparison addition could go here if requested.
        pass
    
# --- Main Panel ---
if app_mode == "Data Dashboard":
    dashboard.render_dashboard(base_data_path, LOCAL_CACHE_ROOT)
    st.stop()

if app_mode == "Figure Builder":
    from utils import figure_composer
    figure_composer.render_figure_composer(st.session_state.measurements)
    st.stop()

if 'iterations' in st.session_state and st.session_state.iterations:
    controls = st.container(border=True)
    
    # Row 1: Iteration & Plotting Controls
    r1_c1, r1_c2 = controls.columns([0.5, 0.5])
    
    with r1_c1:
        iter_cols = st.columns([0.2, 0.6, 0.2])
        iter_opts = ["Average All"] + st.session_state.iterations
        sel_iter = iter_cols[1].selectbox("Iteration", iter_opts, key="selected_iter_option", label_visibility="collapsed")
        is_avg = sel_iter == "Average All"
        
        if not is_avg:
            curr_idx = iter_opts.index(sel_iter)
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
        # Only show method selector if wall subtraction is on
        wall_method = "dynamic" # Static method is removed.

    # --- Smart Reload Logic ---
    # Check if loaded data matches the current 'subtract_wall' state
    if st.session_state.get('precomputed_data'):
        # We can infer the type from the filename if we stored it, or check metadata if available
        # For now, we'll try to reload if the user toggles the switch and we suspect a mismatch
        # A simple heuristic: If subtract_wall is ON, we want dynamic file. If OFF, standard.
        
        target_mode = "dynamic" if subtract_wall else "standard"
        current_mode = st.session_state.get('loaded_mode', 'unknown')
        
        if current_mode != target_mode:
            new_data, new_meta, new_fmt = app_utils.try_load_precomputed(st.session_state.current_meas_path, LOCAL_CACHE_ROOT, mode=target_mode)
            if new_data:
                st.session_state.precomputed_data = new_data
                st.session_state.loaded_mode = target_mode

    # --- Calculation ---
    final_results, curves_to_show, roi_paths_to_show = {}, {}, {}
    
    # Ensure analysis_type is defined
    analysis_type = st.session_state.get('analysis_type', "Mueller Matrix")
    
    req_meas = app_utils.get_required_measurements(st.session_state.measurements, analysis_type)
    
    # 1. Try Precomputed
    if st.session_state.get('use_precomputed') and st.session_state.get('precomputed_data'):
        pre = st.session_state.precomputed_data
        if is_avg:
            all_res = defaultdict(list)
            for d in pre.values():
                for k, v in d.items(): all_res[k].append(v)

    # 2. Calculate if missing
    if not final_results:
        if is_avg:
            all_res = defaultdict(list)
            prog = st.progress(0)
            for i, it in enumerate(st.session_state.iterations):
                curves, _ = app_computation.calculate_curves_for_iteration(it, st.session_state.measurements, st.session_state.backgrounds, req_meas, analysis_type, bg_on, noise_sigma, 10**(-log_thresh), angle_model, subtract_wall)
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
            for k, v in all_res.items(): final_results[k] = {'mean': np.mean(v, axis=0), 'std': np.std(v, axis=0)}
        else:
            curves, roi_paths = app_computation.calculate_curves_for_iteration(sel_iter, st.session_state.measurements, st.session_state.backgrounds, req_meas, analysis_type, bg_on, noise_sigma, 10**(-log_thresh), angle_model, subtract_wall)
            if curves:
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
                
                if res:
                    for k, v in res.items(): final_results[k] = {'mean': v}
                    # Auto-save
                    if st.session_state.get('precomputed_data') is None: st.session_state.precomputed_data = {}
                    if str(sel_iter) not in st.session_state.precomputed_data:
                        st.session_state.precomputed_data[str(sel_iter)] = res
                        meta = {"source_path": st.session_state.current_meas_path, "analysis_type": analysis_type, "parameters": {"bg_subtraction": bg_on, "subtract_wall": subtract_wall}}
                        is_dynamic_run = subtract_wall
                        app_utils.save_precomputed_data(st.session_state.current_meas_path, st.session_state.precomputed_data, meta, fmt=export_fmt, local_cache_root=LOCAL_CACHE_ROOT, is_dynamic=is_dynamic_run)

    # --- Plotting ---
    if final_results:
        ref_w = list(final_results.values())[0]['mean'].shape[0]
        x_angles = angle_model(np.arange(ref_w) + PIXEL_OFFSET_X)
        
        # Angle Slider
        if 'angle_range' not in st.session_state: st.session_state.angle_range = (100.0, 168.0)
        min_a, max_a = float(x_angles.min()), float(x_angles.max())
        st.session_state.angle_range = st.slider("Angle Range", min_a, max_a, st.session_state.angle_range, help="Range of scattering angles to display.")
        
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
                        fig = go.Figure()
                        mean = final_results[key]['mean']
                        y_stack = [mean]
                        
                        # Comparisons
                        for i, comp in enumerate(st.session_state.comparisons):
                            # Logic to get comparison data (precomputed or calculated)
                            c_res = {}
                            if comp.get('precomputed_data'):
                                d = comp['precomputed_data']
                                if is_avg:
                                    vals = [v[key] for v in d.values() if key in v]
                                    if vals: c_res = np.mean(vals, axis=0)
                                else:
                                    c_res = d.get(str(sel_iter), {}).get(key)
                            
                            if c_res is None: # Fallback calc
                                # (Simplified for brevity: assume precomputed mostly used or add calc logic here)
                                pass
                            
                            if c_res is not None:
                                c_plot = c_res.copy()
                                suffix = ""
                                if match_intensity:
                                    p_max, c_max = np.nanmax(mean), np.nanmax(c_plot)
                                    if c_max != 0: 
                                        c_plot *= (p_max/c_max)
                                        suffix = f" (x{p_max/c_max:.2f})"
                                label = app_utils.get_smart_label(comp['name'], st.session_state.measurements, comp.get('measurements'))
                                color_c = comp_colors[i % len(comp_colors)]
                                fig.add_trace(go.Scatter(x=x_angles, y=c_plot, mode='lines', line=dict(color=color_c), name=f"{label}{suffix}"))
                                y_stack.append(c_plot)

                        fig.add_trace(go.Scatter(x=x_angles, y=mean, mode='lines', line=dict(color=color), name='Primary'))
                        
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
                            
                            if c_res is not None:
                                c_plot = c_res.copy()
                                suffix = ""
                                if match_intensity:
                                    p_max, c_max = np.nanmax(data), np.nanmax(c_plot)
                                    if c_max != 0:
                                        c_plot *= (p_max/c_max)
                                        suffix = f" (x{p_max/c_max:.2f})"
                                label = app_utils.get_smart_label(comp['name'], st.session_state.measurements, comp.get('measurements'))
                                color_c = comp_colors[i % len(comp_colors)]
                                fig.add_trace(go.Scatter(x=x_angles, y=c_plot, mode='lines', line=dict(color=color_c), name=f"{label}{suffix}"))
                                y_stack.append(c_plot)

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
                    fig = go.Figure()
                    mean = final_results['Depolarization Ratio']['mean']
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
                            label = app_utils.get_smart_label(comp['name'], st.session_state.measurements, comp.get('measurements'))
                            color_c = comp_colors[i % len(comp_colors)]
                            fig.add_trace(go.Scatter(x=x_angles, y=c_plot, mode='lines', line=dict(color=color_c), name=f"{label}{suffix}"))
                            y_stack.append(c_plot)

                    fig.add_trace(go.Scatter(x=x_angles, y=mean, mode='lines', line=dict(color="#1f77b4"), name='Primary'))
                    
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