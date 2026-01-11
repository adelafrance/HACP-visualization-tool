import streamlit as st
import os # Standard library
import json
import plotly.graph_objects as go
from plotly import subplots
import matplotlib.pyplot as plt
import io
import pandas as pd # Added pandas import
import warnings # Added warnings import
from utils import plotting, app_computation, app_utils, polarimeter_processing
import numpy as np
from PIL import Image

# Constants
PIXEL_OFFSET_X = 520


def generate_composite_figure(fig_specs, layout_type, width, height, unit, check_dependencies=None, 
                            common_labels=False, global_xlabel="", global_ylabel="", 
                            active_style=None, colors=None, primary_label="Primary", comp_label="Comparison",
                            show_legend=False, legend_loc="best", disable_sci_angle=False, disable_sci_y=False,
                            angle_range=None, angle_model=None, pixel_offset=0, anchor_to_primary=False, subtract_wall=False):
    """
    Generates a matplotlib figure based on the provided specifications.
    Decoupled from Streamlit for use in optimization scripts.
    
    check_dependencies: dict containing 'measurements', 'backgrounds', 'comp_measurements', 'comp_backgrounds', 
                       'precomputed_data', 'comp_precomputed_data', 'active_dataset_type'
    """
    if check_dependencies is None: check_dependencies = {}
    measurements = check_dependencies.get('measurements')
    backgrounds = check_dependencies.get('backgrounds')
    
    # Dimensions
    if unit == "px":
            final_w, final_h = width / 100.0, height / 100.0 # Convert px to inches for matplotlib
    else:
            final_w, final_h = width, height

    plt.close('all')
    fig = plt.figure(figsize=(final_w, final_h))
    
    # Spacing
    if common_labels: fig.subplots_adjust(wspace=0.25, hspace=0.05)
    else: fig.subplots_adjust(wspace=0.35, hspace=0.35)

    rows = 2 if "2x" in layout_type or "1x2" in layout_type else 1
    cols = 2 if "2x2" in layout_type or "2x1" in layout_type else 1

    # Define helper internally to close over dependencies
    def get_series_data(t_iters, t_meas, t_bg, t_ana_type, t_var):
        t_curves = []
        t_x = None
        
        # MEANINGFUL DIFF: In the app, we fall back to st.session_state.iterations.
        # Here, we must rely on what's passed in t_meas or check_dependencies.
        fallback_iters = check_dependencies.get('iterations', [])
        actual_iters = t_iters if t_iters else (list(t_meas.keys()) if t_meas else fallback_iters)
        
        if not actual_iters: return None, None, None
        
        subtract_wall = st.session_state.get('global_subtract_wall', False)
        
        for it in actual_iters:
            iter_res = None
            # Case 1: Primary data - check precomputed cache
            # We check if the PASSED measurements object matches the PRIMARY one
            is_primary = (t_meas is measurements)
            
            p_data = check_dependencies.get('precomputed_data')
            p_meta = check_dependencies.get('precomputed_meta', {})
            is_cached_dynamic = p_meta.get('parameters', {}).get('subtract_wall', False)
            
            # Robust key check for measurements
            it_in_meas = t_meas and (it in t_meas or str(it) in t_meas)
            can_recompute = t_meas and it_in_meas

            # Case 1: Use precomputed cache if it matches our subtract_wall setting
            # OR if we are in Limited Mode and CANNOT recompute from raw files.
            if is_primary and p_data:
                if (is_cached_dynamic == subtract_wall) or not can_recompute:
                    # Check both string and int keys for robustness
                    cached = p_data.get(str(it))
                    if cached is None: cached = p_data.get(int(it)) if str(it).isdigit() else None
                    if cached: iter_res = cached
            
            # Case 2: Comparison data
            if iter_res is None and not is_primary and check_dependencies.get('comp_precomputed_data'):
                c_data = check_dependencies['comp_precomputed_data']
                iter_res = c_data.get(str(it))
                if iter_res is None: iter_res = c_data.get(int(it)) if str(it).isdigit() else None

            # Case 3: Raw data available - compute
            if iter_res is None and can_recompute:
                # Ensure we have the correct files for the iteration (handle str/int)
                it_meas_files = t_meas.get(it, t_meas.get(str(it)))
                req = app_utils.get_required_measurements({it: it_meas_files}, t_ana_type)
                try:
                    calc_res, _ = app_computation.calculate_curves_for_iteration(
                        it, {it: it_meas_files}, t_bg, req, t_ana_type, 
                        True, 3.0, 1e-4, angle_model, subtract_wall=subtract_wall
                    )
                    if calc_res: iter_res = calc_res
                except Exception as e:
                    st.error(f"Error computing iteration {it}: {e}")
            
            if not iter_res: continue
            
            y_val = None
            # Calculate derived vars if needed
            mm_elems = polarimeter_processing.calculate_mueller_elements(iter_res)
            
            # Support aliases for Depolarization Ratio calculation
            
            # --- New Ratio-of-Means Logic ---
            # Identify if we need component-wise processing
            ratio_components = None
            if t_var == "S12/S11": ratio_components = ("S12", "S11")
            elif t_var == "S33/S11": ratio_components = ("S33", "S11")
            elif t_var == "S34/S11": ratio_components = ("S34", "S11")
            elif t_var == "DoLP": ratio_components = ("DoLP_Num", "S11") # Special case requires |S12| check
            
            # Special DoLP handling if keys missing
            if t_var == "DoLP" and "DoLP" in iter_res:
                # Use pre-calculated DoLP if available (simple case)
                y_val = iter_res["DoLP"]
            elif ratio_components:
                num_key, den_key = ratio_components
                
                # Handle DoLP Numerator on the fly if needed
                if num_key == "DoLP_Num":
                   num_val = np.abs(iter_res.get("S12")) if "S12" in iter_res else None
                else:
                   num_val = iter_res.get(num_key)
                
                den_val = iter_res.get(den_key)
                
                if num_val is not None and den_val is not None:
                     # For Ratio-of-Means, we store the components tuple
                     # We reuse t_curves to store tuples if ratio mode
                     y_val = (num_val, den_val)
                else:
                     y_val = None
            
            # Standard single variable handling
            elif t_var in iter_res: y_val = iter_res[t_var]
            elif t_var == "Depolarization Ratio":
                # Depol Ratio is also a ratio-of-means candidate, but let's stick to existing logic for now unless requested
                p_key = 'Depol_Parallel' if 'Depol_Parallel' in iter_res else 'Parallel'
                c_key = 'Depol_Cross' if 'Depol_Cross' in iter_res else 'Cross'
                if p_key in iter_res and c_key in iter_res:
                    y_val = (iter_res[c_key]+1e-9)/(iter_res[p_key]+1e-9)
            elif mm_elems and t_var in mm_elems: y_val = mm_elems[t_var]
            
            if y_val is not None:
                # Re-map X to Angle if model exists
                if t_x is None:
                    # Determine length based on y_val type
                    val_len = len(y_val[0]) if isinstance(y_val, tuple) else len(y_val)
                    if angle_model:
                        t_x = angle_model(np.arange(val_len) + pixel_offset)
                    else:
                        t_x = np.arange(val_len)
                t_curves.append(y_val)
        
        if not t_curves or t_x is None: return None, None, None
        
        # Stack and calculate mean/std
        with warnings.catch_warnings(): # Use standard warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            # Check if we have tuples (Ratio of Means mode)
            if t_curves and isinstance(t_curves[0], tuple):
                # Unpack
                nums = np.vstack([x[0] for x in t_curves])
                dens = np.vstack([x[1] for x in t_curves])
                
                mean_num = np.nanmean(nums, axis=0)
                mean_den = np.nanmean(dens, axis=0)
                
                # Ratio of Means
                with np.errstate(divide='ignore', invalid='ignore'):
                     mean_curve = mean_num / mean_den
                
                # Error Propagation for STD
                if len(t_curves) > 1:
                    std_num = np.nanstd(nums, axis=0)
                    std_den = np.nanstd(dens, axis=0)
                    
                    # std_f / |f| = sqrt( (std_A/A)^2 + (std_B/B)^2 )
                    # std_f = |f| * ...
                    with np.errstate(divide='ignore', invalid='ignore'):
                         term1 = (std_num / mean_num)**2
                         term2 = (std_den / mean_den)**2
                         std_curve = np.abs(mean_curve) * np.sqrt(term1 + term2)
                else:
                    std_curve = None
            else:
                # Standard Mean of Curves
                stack = np.vstack(t_curves) # (N_iter, N_pixels)
                mean_curve = np.nanmean(stack, axis=0)
                std_curve = np.nanstd(stack, axis=0) if len(t_curves) > 1 else None
        
        # Filter by Angle Range
        if angle_range:
            mask = (t_x >= angle_range[0]) & (t_x <= angle_range[1])
            return t_x[mask], mean_curve[mask], std_curve[mask] if std_curve is not None else None
        else:
            return t_x, mean_curve, std_curve

    for (r, c), spec in fig_specs.items():
        idx = (r-1)*cols + c
        ax = fig.add_subplot(rows, cols, idx)
        
        show_y = True 
        show_x = (r == rows) or (not common_labels)
        
        panel_xlabel = spec.get("xlabel", global_xlabel) if show_x else ""
        panel_ylabel = spec.get("ylabel", global_ylabel) if show_y else ""

        try:
            if spec["type"] == "Raw Image":
                img = np.array(Image.open(spec["file"])) # (H, W)
                x_pixel_min, x_pixel_max = 0, img.shape[1]
                if angle_model and angle_range:
                    all_angles = angle_model(np.arange(img.shape[1]))
                    valid_mask = (all_angles >= angle_range[0]) & (all_angles <= angle_range[1])
                    if np.any(valid_mask):
                            valid_indices = np.where(valid_mask)[0]
                            x_pixel_min, x_pixel_max = valid_indices[0], valid_indices[-1]
                            img = img[:, x_pixel_min:x_pixel_max+1]
                
                plotting.create_mpl_heatmap(
                    img, ax, cmap='viridis', 
                    title=f"({r},{c})", 
                    xlabel=panel_xlabel, ylabel=panel_ylabel,
                    show_x=show_x, show_y=show_y,
                    style=active_style
                )

            elif spec["type"] == "Signal Decomposition":
                    img = np.array(Image.open(spec["file"])).astype(float)
                    col_idx = int(spec['col_idx'])
                    if col_idx < 0 or col_idx >= img.shape[1]: continue
                    norm_profile = img[:, col_idx]
                    y_pixels = np.arange(len(norm_profile))
                    popt = polarimeter_processing.fit_double_gaussian_params(y_pixels, norm_profile)
                    
                    plotting.create_mpl_decomposition(
                    y_pixels, norm_profile, ax, popt=popt,
                    xlabel=panel_xlabel, ylabel=panel_ylabel,
                    show_x=show_x, show_y=show_y,
                    style=active_style
                    )

            elif spec["type"] == "Computed Data":
                # Determine Analysis Type
                ana_type = check_dependencies.get('active_dataset_type', "Mueller Matrix")
                
                # Heuristic override
                if spec['iters'] and measurements:
                    first_iter_keys = list(measurements[spec['iters'][0]].keys())
                    if "Depolarization Ratio" == spec["variable"] and 'Depol_Parallel' in first_iter_keys:
                        ana_type = "Depolarization Ratio"

                x_p, y_p, std_p = get_series_data(spec['iters'], measurements, backgrounds, ana_type, spec['variable'])
                
                # Smart Scaling: Collect mean curves to determine Y-range
                means_to_scale = []
                s_factor = spec.get("scale_factor", 1.0)
                
                # --- Label Logic (Match Optimizer Defaults) ---
                # 1. Internal Panel Label (Top Left) -> Always Variable Name
                int_label = spec["variable"]

                # 2. Smart Y-Axis Label
                # Use custom "Panel Title/Y Label" if provided, otherwise smart default
                custom_y = spec.get("panel_title")
                if custom_y and custom_y.strip():
                    panel_ylabel = custom_y 
                elif not spec.get("ylabel") or spec.get("ylabel") == global_ylabel:
                    if spec["variable"] == "S11" and s_factor < 0.01:
                         panel_ylabel = "Intensity ($10^4$)"
                    elif spec["variable"] == "Depolarization Ratio":
                         panel_ylabel = "Depolarization Ratio"
                    elif spec["variable"] != "S11": 
                         panel_ylabel = "Normalized Element Intensity"

                if y_p is not None: means_to_scale.append(y_p * s_factor)

                # Comparison(s)
                comparison_runs = [] # List of (x, y, std, label, color)
                
                # Handling for legacy single comparison
                if spec.get("show_comp") and check_dependencies.get('comp_measurements'):
                     c_meas = check_dependencies['comp_measurements']
                     c_bg = check_dependencies.get('comp_backgrounds', {})
                     c_iters = check_dependencies.get('global_comp_iters', [])
                     x_c, y_c, std_c = get_series_data(c_iters, c_meas, c_bg, ana_type, spec['variable'])
                     if y_c is not None:
                         comparison_runs.append((x_c, y_c, std_c, comp_label, colors[1] if len(colors)>1 else None))
                         means_to_scale.append(y_c * s_factor)

                # Handling for new multi-comparison structure
                if spec.get("show_comp") and check_dependencies.get('multi_comparisons'):
                    multi_comp = check_dependencies['multi_comparisons']
                    for i, comp_deps in enumerate(multi_comp):
                        c_meas = comp_deps.get('measurements')
                        c_bg = comp_deps.get('backgrounds', {})
                        c_pre = comp_deps.get('precomputed_data', {})
                        c_iters = comp_deps.get('iterations', [])
                        
                        # We need to temporarily add precomputed_data to check_dependencies for get_series_data to find it
                        # This is a bit hacky but keeps the helper simple.
                        orig_comp_pre = check_dependencies.get('comp_precomputed_data')
                        check_dependencies['comp_precomputed_data'] = c_pre
                        
                        x_mc, y_mc, std_mc = get_series_data(c_iters, c_meas, c_bg, ana_type, spec['variable'])
                        
                        # Restore
                        check_dependencies['comp_precomputed_data'] = orig_comp_pre
                        
                        if y_mc is not None:
                            c_label = comp_deps.get('label', f"Comp {i+1}")
                            c_color = comp_deps.get('color')
                            comparison_runs.append((x_mc, y_mc, std_mc, c_label, c_color))
                            means_to_scale.append(y_mc * s_factor)

                # Calculate Y-range if not manual
                panel_ylim = spec.get("ylim")
                if panel_ylim is None and means_to_scale:
                    if anchor_to_primary and y_p is not None:
                         # Scale to PRIMARY only
                         scale_source = y_p * s_factor
                    else:
                         # Scale to ALL
                         scale_source = np.concatenate([m for m in means_to_scale if m is not None])
                    
                    if scale_source.size > 0 and not np.all(np.isnan(scale_source)):
                        mn, mx = np.nanmin(scale_source), np.nanmax(scale_source)
                        pad = (mx - mn) * 0.15 
                        if pad == 0: pad = 0.1 * abs(mn) if mn != 0 else 1.0
                        panel_ylim = (mn - pad, mx + pad)
                
                # DEBUG: Trace why plot might be blank
                # DEBUG: Trace why plot might be blank
                
                if y_p is not None:
                    plotting.create_mpl_line(
                        x_p, y_p, ax, y_err=std_p if spec['show_std'] else None,
                        label=primary_label,
                        xlabel=panel_xlabel, ylabel=panel_ylabel,
                        show_x=show_x, show_y=show_y,
                        style=active_style, color=colors[0],
                        disable_sci_x=disable_sci_angle,
                        disable_sci_y=disable_sci_y,
                        ylim=panel_ylim,
                        scale_factor=s_factor,
                        internal_label=int_label
                    )
                
                # Plot all collected comparisons
                for x_c, y_c, std_c, c_lab, c_col in comparison_runs:
                    plotting.create_mpl_line(
                        x_c, y_c, ax, 
                        y_err=std_c if spec.get('show_std_comp', False) else None, 
                        label=c_lab,
                        color=c_col if c_col else None,
                        linestyle='--',
                        xlabel=panel_xlabel, ylabel=panel_ylabel,
                        show_x=show_x, show_y=show_y,
                        style=active_style,
                        disable_sci_x=disable_sci_angle,
                        disable_sci_y=disable_sci_y,
                        ylim=panel_ylim,
                        scale_factor=s_factor
                    )

                if show_legend and (y_p is not None or comparison_runs):
                    ax.legend(loc=legend_loc)
                
        except Exception as e:
            st.error(f"Error in panel ({r},{c}): {e}")

    try:
        plt.tight_layout()
    except np.linalg.LinAlgError:
        pass # Ignore singular matrix error in layout engine
    except Exception as e:
        print(f"Warning: tight_layout failed: {e}")
    return fig

def render_figure_composer(measurements, cache_root):
    # --- Sidebar Dataset Info & Switcher ---
    p_state = st.session_state.get('p_state', {})
    active_desc = p_state.get('description', "Unknown Dataset")
    active_seq = p_state.get('meas_seq', "Unknown")
    
    st.sidebar.title("Figure Builder Tools")
    st.sidebar.markdown(f"**Current Dataset:**\n### {active_desc}")
    st.sidebar.caption(f"ID: `{active_seq}`")
    
    st.sidebar.divider()
    
    # --- Analysis Type Filter ---
    st.sidebar.subheader("Filter Options")
    fb_filter_type = st.sidebar.selectbox("Filter Analysis Type", ["All", "Mueller Matrix", "Depolarization Ratio"], key="fb_filter_type")
    
    cached_datasets = app_utils.scan_local_cache(cache_root)
    if cached_datasets:
        st.sidebar.subheader("Quick Switch")
        
        # Filter cached datasets by selected analysis type
        if fb_filter_type != "All":
            filtered_cached = [d for d in cached_datasets if app_utils.get_analysis_type_from_sequence(d.get('Sequence')) == fb_filter_type]
        else:
            filtered_cached = cached_datasets
            
        cached_opts = [f"{d['Date']} | {app_utils.translate_sequence_name(d['Sequence'])} ({d['Sequence']})" for d in filtered_cached]
        
        curr_idx = 0
        current_combined = f"{p_state.get('date')} | {active_desc} ({active_seq})"
        if current_combined in cached_opts:
            curr_idx = cached_opts.index(current_combined)
            
        sel_switch = st.sidebar.selectbox("Jump to Dataset", cached_opts, index=curr_idx, key="fb_sidebar_switcher")
        
        if sel_switch != current_combined:
            parts = sel_switch.split(" | ", 1)
            date_val = parts[0]
            rest = parts[1]
            seq_val = rest.rsplit(" (", 1)[-1][:-1]
            
            new_dataset = next((d for d in cached_datasets if d['Date'] == date_val and d['Sequence'] == seq_val), None)
            if new_dataset:
                st.session_state.target_load = {
                    "date": date_val,
                    "meas_parent": os.path.basename(os.path.dirname(os.path.dirname(new_dataset['LocalPath']))),
                    "meas_seq": seq_val,
                    "meas_path": new_dataset['LocalPath'],
                    "description": app_utils.translate_sequence_name(seq_val),
                    "switch_to_mode": "Figure Builder"
                }
                st.rerun()
    
    st.sidebar.divider()
    
    st.header("Figure Builder")
    
    # --- 0. Configuration Management ---
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'viz_configs')
    if not os.path.exists(config_dir): os.makedirs(config_dir)

    with st.expander("Configuration Manager", expanded=False):
        cm1, cm2 = st.columns(2)
        
        # Save
        with cm1:
            save_name = st.text_input("Config Name (Save)", value="my_config")
            if st.button("Save Current Configuration"):
                try:
                    # Filter session state for relevant keys
                    keys_to_save = [k for k in st.session_state.keys() if 
                                    k.startswith(('global_', 'pt_', 'ug_', 'it_', 'sit_', 'mk_', 'var_', 'std_', 'cp_', 'ms_', 'ymn_', 'ymx_', 'dk_', 'ci_', 'ov_', 'xl_', 'yl_', 'lbl_')) or 
                                    k in ['angle_range', 'fb_layout', 'angle_min_fb', 'angle_max_fb', 'angle_range_slider_fb', 'global_iters', 'global_select_all', 'global_subtract_wall']]
                    config_data = {k: st.session_state[k] for k in keys_to_save}
                    
                    fpath = os.path.join(config_dir, f"{save_name}.json")
                    with open(fpath, 'w') as f:
                        json.dump(config_data, f, indent=4)
                    st.success(f"Saved: {save_name}")
                except Exception as e:
                    st.error(f"Save failed: {e}")

        # Load
        with cm2:
            existing_configs = [f.replace(".json","") for f in os.listdir(config_dir) if f.endswith(".json")]
            if existing_configs:
                load_name = st.selectbox("Select Config", sorted(existing_configs))
                if st.button("Load Configuration"):
                    try:
                        fpath = os.path.join(config_dir, f"{load_name}.json")
                        with open(fpath, 'r') as f:
                            data = json.load(f)
                        st.session_state.update(data)
                        st.session_state.angle_range = tuple(data.get('angle_range', (100.0, 167.0))) # Ensure tuple
                        st.success(f"Loaded: {load_name}")
                        st.rerun() # Force immediate refresh of widgets (including dimensions)
                    except Exception as e:
                        st.error(f"Load failed: {e}")
            else:
                st.info("No saved configurations found.")

    # --- 1. Global Settings ---
    with st.expander("Global Figure Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        # Dimensions
        unit = c1.selectbox("Unit", ["inch", "px"], index=0, key="global_unit")
        if unit == "inch":
            w = c2.number_input("Width (in)", step=0.1, key="global_w_in")
            h = c3.number_input("Height (in)", step=0.1, key="global_h_in")
        else:
            w = c2.number_input("Width (px)", step=100, key="global_w_px")
            h = c3.number_input("Height (px)", step=100, key="global_h_px")
            
        # Angle Range
        if 'angle_range' not in st.session_state: st.session_state.angle_range = (100.0, 167.0)
        
        # Callbacks for sync
        def _sync_from_slider_fb():
            low, high = st.session_state.angle_range_slider_fb
            st.session_state.angle_min_fb = int(low)
            st.session_state.angle_max_fb = int(high)
            st.session_state.angle_range = (float(low), float(high))

        def _sync_from_inputs_fb():
            low = st.session_state.angle_min_fb
            high = st.session_state.angle_max_fb
            st.session_state.angle_range_slider_fb = (float(low), float(high))
            st.session_state.angle_range = (float(low), float(high))

        # Init session state if missing or out of bounds
        low_init = max(50.0, min(170.0, st.session_state.angle_range[0]))
        high_init = max(50.0, min(170.0, st.session_state.angle_range[1]))
        
        if 'angle_min_fb' not in st.session_state: st.session_state.angle_min_fb = int(low_init)
        if 'angle_max_fb' not in st.session_state: st.session_state.angle_max_fb = int(high_init)
        if 'angle_range_slider_fb' not in st.session_state: st.session_state.angle_range_slider_fb = (float(st.session_state.angle_min_fb), float(st.session_state.angle_max_fb))

        # Ensure widget keys are in sync with global state if changed elsewhere (e.g. Interactive Analysis)
        # We use ceil for the min input to stay safely above or at low_init
        target_min = int(np.ceil(low_init)) if low_init > int(low_init) else int(low_init)
        if st.session_state.angle_min_fb != target_min: st.session_state.angle_min_fb = target_min
        if st.session_state.angle_max_fb != int(high_init): st.session_state.angle_max_fb = int(high_init)
        
        # Slider state MUST be strictly within [50.0, 170.0]
        st_low = max(50.0, min(170.0, float(st.session_state.angle_min_fb)))
        st_high = max(50.0, min(170.0, float(st.session_state.angle_max_fb)))
        
        if st.session_state.angle_range_slider_fb != (st_low, st_high):
             st.session_state.angle_range_slider_fb = (st_low, st_high)

        ac1, ac2, ac3 = st.columns([0.4, 0.3, 0.3])
        
        # User widgets with callbacks
        ac2.number_input("Min Angle [deg]", 50, 170, step=1, key="angle_min_fb", on_change=_sync_from_inputs_fb)
        ac3.number_input("Max Angle [deg]", 50, 170, step=1, key="angle_max_fb", on_change=_sync_from_inputs_fb)
        ac1.slider("Coarse Adjust", 50.0, 170.0, key="angle_range_slider_fb", on_change=_sync_from_slider_fb)
        
        # Final sync to global key used by plotting
        st.session_state.angle_range = (float(st.session_state.angle_min_fb), float(st.session_state.angle_max_fb))
        
        # Style & Scaling
        c_s1, c_s2, c_s3 = st.columns(3)
        style_name = c_s1.selectbox("Style Preset", list(plotting.STYLES.keys()), key="global_style")
        font_scale = c_s2.number_input("Font Scale", 0.5, 2.0, step=0.1, key="global_font_scale")
        disable_sci_angle = c_s3.checkbox("No Sci Notation (X)", key="global_no_sci")
        
        selected_style = plotting.STYLES[style_name]
        # Apply scaling to style copy
        import copy
        active_style = copy.deepcopy(selected_style)
        active_style.font_size = int(active_style.font_size * font_scale)
        
        # Label Settings
        c_l1, c_l2 = st.columns(2)
        common_labels = c_l1.checkbox("Common Labels", help="Hide inner axis labels in grid layouts.", key="global_common_labels")
        show_std_global = c_l2.checkbox("Show Standard Deviation", help="Shade standard deviation if multiple iterations are averaged.", key="global_show_std")
        
        global_xlabel = st.text_input("Global X Label", value="Scattering Angle [deg]", key="global_xlabel")
        global_ylabel = st.text_input("Global Y Label", value="Normalized Element Intensity", key="global_ylabel")
        
        c_l3, c_l4 = st.columns(2)
        subtract_wall_global = c_l3.checkbox("Subtract Wall (Dynamic)", value=True, help="Enable advanced wall subtraction fitting. Requires raw measurement files.", key="global_subtract_wall")
        anchor_to_primary = c_l4.checkbox("Anchor Y-Scaling to Primary Only", value=True, help="Determine Y-axis limits using only primary data, even when comparisons are added.", key="global_anchor")

        # Color Schemes
        COLOR_SCHEMES = {
            "Default (Blue/Red)": ("#1f77b4", "#d62728"),
            "Colorblind Safe (Teal/Orange)": ("#1b9e77", "#d95f02"),
            "Publication (Black/Red)": ("black", "red"),
            "Grayscale (Black/Gray)": ("black", "gray"),
            "High Contrast (Yellow/Cyan)": ("#bcbd22", "#17becf"),
            "Pastel (Blue/Pink)": ("#a1c9f4", "#fbe4ff"),
            "Custom": (None, None)
        }
        scheme_name = st.selectbox("Color Scheme", list(COLOR_SCHEMES.keys()), key="global_color_scheme")
        
        if scheme_name == "Custom":
            c_col1, c_col2 = st.columns(2)
            color1 = c_col1.color_picker("Primary Color", "#1f77b4")
            color2 = c_col2.color_picker("Comparison Color", "#d62728")
            colors = (color1, color2)
        else:
            colors = COLOR_SCHEMES[scheme_name]

        # Labels & Legend
        st.markdown("**Legend & Labels**")
        l_c1, l_c2 = st.columns(2)
        primary_label = l_c1.text_input("Primary Label", value="Primary", key="lbl_prim")
        comp_label = l_c2.text_input("Comparison Label", value="Comparison", key="lbl_comp")
        
        ll_c1, ll_c2 = st.columns(2)
        show_legend = ll_c1.checkbox("Show Legend", key="lbl_show")
        legend_loc = ll_c2.selectbox("Legend Loc", ["best", "upper right", "upper left", "lower left", "lower right", "center right"], key="lbl_loc")

        st.markdown("---")
        st.markdown("**Primary Data Iterations**")
# ... [Omitted iteration selection lines for brevity, need to match exact context] ...
# Actually better to just update the specific blocks rather than one giant replace if context is far apart.
# Context is line 84 to 90.

# AND update lines ~400 and ~420 for plotting.
# I will do this in 2 chunks to be safe.

        # Determine the true analysis type for the current primary dataset
        active_dataset_type = app_utils.get_analysis_type_from_sequence(active_seq)

        # Dataset Change Detection: Reset iteration selections if sequence ID changed
        last_seq = st.session_state.get('fb_last_seq')
        if last_seq != active_seq:
            if 'global_iters' in st.session_state: del st.session_state['global_iters']
            if 'global_select_all' in st.session_state: del st.session_state['global_select_all']
            st.session_state.fb_last_seq = active_seq

        all_iters = sorted(measurements.keys()) if measurements else []
        if not all_iters and st.session_state.get('precomputed_data'):
            pre_data = st.session_state.precomputed_data
            all_iters = sorted([int(k) if str(k).isdigit() else k for k in pre_data.keys() if k not in ['metadata', 'Average']])
        iter_label_map = {f"{app_utils.translate_sequence_name(str(it))} ({it})": it for it in all_iters}
        iter_labels = list(iter_label_map.keys())
        
        ic1, ic2 = st.columns([0.2, 0.8])
        select_all_iters = ic1.checkbox("Select All", value=True, key="global_select_all")
        
        if select_all_iters:
             global_iters_labels = ic2.multiselect("Iterations", iter_labels, default=iter_labels, disabled=True, key="global_iters_disabled")
             global_iters = all_iters
        else:
             default_iter_labels = [iter_labels[0]] if iter_labels else []
             global_iters_labels = ic2.multiselect("Iterations", iter_labels, default=default_iter_labels, key="global_iters")
             global_iters = [iter_label_map[l] for l in global_iters_labels]

        # Robust Fallback: If nothing selected, pick first to prevent blank plots
        if not global_iters and all_iters:
             global_iters = [all_iters[0]]


    # --- 1b. Comparison Data Source ---
    with st.expander("Comparison Data Source", expanded=True):
        if 'multi_comparisons' not in st.session_state: st.session_state.multi_comparisons = []
        
        source_mode = st.radio("Source Mode", ["Select Preprocessed Dataset", "Manual Folder Path"], horizontal=True, key="comp_source_mode")
        
        comp_to_add = None
        
        if source_mode == "Select Preprocessed Dataset":
            # Select from Preprocessed Data (Local Cache)
            cached_datasets = app_utils.scan_local_cache(cache_root)
            if cached_datasets:
                if fb_filter_type != "All":
                    comp_cached = [d for d in cached_datasets if app_utils.get_analysis_type_from_sequence(d.get('Sequence', "")) == fb_filter_type]
                else:
                    comp_cached = cached_datasets
                
                if comp_cached:
                    cached_opts = [f"{d['Date']} | {app_utils.translate_sequence_name(d['Sequence'])} ({d['Sequence']})" for d in comp_cached]
                    sel_cached = st.selectbox("Select Cached Dataset", cached_opts, key="comp_cache_sel")
                    
                    if st.button("Add to Comparisons"):
                        target_d = comp_cached[cached_opts.index(sel_cached)]
                        pre_data, _, _ = app_utils.try_load_precomputed(target_d['LocalPath'])
                        if pre_data:
                            comp_to_add = {
                                "label": target_d['Sequence'],
                                "precomputed_data": pre_data,
                                "iterations": sorted(pre_data.keys(), key=lambda x: (x != "Average", x)),
                                "color": None # Will assign from palette
                            }
                else: st.info(f"No {fb_filter_type} datasets found in cache.")

        else:
            comp_path = st.text_input("Comparison Folder Path", value="", key="comp_path_input")
            if st.button("Load & Add comparison"):
                if os.path.isdir(comp_path):
                    m_path = os.path.join(comp_path, 'measurements')
                    b_path = os.path.join(comp_path, 'backgrounds')
                    if os.path.exists(m_path) and os.path.exists(b_path):
                         c_meas, c_bg, _ = polarimeter_processing.find_and_organize_files(m_path, b_path)
                         if c_meas:
                             comp_to_add = {
                                 "label": os.path.basename(comp_path),
                                 "measurements": c_meas,
                                 "backgrounds": c_bg,
                                 "iterations": sorted(c_meas.keys())
                             }
                    else: st.error("Missing folders.")

        if comp_to_add:
            # Check for duplicates
            if not any(c['label'] == comp_to_add['label'] for c in st.session_state.multi_comparisons):
                # Assign Color
                colors_comp = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
                idx = len(st.session_state.multi_comparisons)
                comp_to_add['color'] = colors_comp[idx % len(colors_comp)]
                st.session_state.multi_comparisons.append(comp_to_add)
                st.success(f"Added: {comp_to_add['label']}")
            else: st.warning("Already added.")

        # Display and Manage Comparisons
        if st.session_state.multi_comparisons:
            st.divider()
            st.markdown("**Active Comparisons**")
            to_remove = None
            for i, comp in enumerate(st.session_state.multi_comparisons):
                c_col1, c_col2, c_col3 = st.columns([0.1, 0.7, 0.2])
                c_col1.markdown(f"<div style='width:20px;height:20px;background-color:{comp['color']};border-radius:3px;'></div>", unsafe_allow_html=True)
                c_col2.caption(comp['label'])
                if c_col3.button("🗑️", key=f"rm_{i}"): to_remove = i
            
            if to_remove is not None:
                st.session_state.multi_comparisons.pop(to_remove)
                st.rerun()
            
            if st.button("Clear All"):
                st.session_state.multi_comparisons = []
                st.rerun()
        else:
            st.info("No comparisons added yet.")


    # --- 2. Layout & Content ---
    layout_type = st.radio("Layout", ["Single Panel", "1x2 Grid (V)", "2x1 Grid (H)", "2x2 Grid"], index=0, horizontal=True, key="fb_layout")
    
    slots = []
    if layout_type == "Single Panel": slots = [(1,1)]
    elif layout_type == "1x2 Grid (V)": slots = [(1,1), (2,1)]
    elif layout_type == "2x1 Grid (H)": slots = [(1,1), (1,2)]
    elif layout_type == "2x2 Grid": slots = [(1,1), (1,2), (2,1), (2,2)]
    
    fig_specs = {}
    
    # Load Angle Model globally for slicing
    angle_range = st.session_state.angle_range
    angle_model = None
    if hasattr(app_utils, 'load_angle_model_from_npz'):
        angle_model = app_utils.load_angle_model_from_npz(app_utils.CALIBRATION_FILE_PATH)

    # Prepare Dependencies for generate_composite_figure
    check_dependencies = {
        'measurements': measurements,
        'backgrounds': st.session_state.backgrounds,
        'precomputed_data': st.session_state.get('precomputed_data'),
        'precomputed_meta': st.session_state.get('precomputed_meta', {}),
        'active_dataset_type': st.session_state.get('analysis_type', 'Mueller Matrix'),
        'multi_comparisons': st.session_state.get('multi_comparisons', []),
        'iterations': global_iters
    }

    st.subheader("Panel Configuration")
    
    # Helper for Panel Config
    def _render_panel_config(r, c):
        with st.container(border=True):
            st.markdown(f"**Panel ({r},{c})**")
            cols = st.columns([0.4, 0.6]) # Adjust ratio
            
            # Plot Type
            p_type = cols[1].selectbox(f"Type", ["Computed Data", "Raw Image", "Signal Decomposition"], key=f"pt_{r}_{c}")

            # Iteration Logic
            iters_to_use = []
            if p_type == "Computed Data":
                use_global = cols[0].checkbox(f"Use Global Iters", value=True, key=f"ug_{r}_{c}")
                if use_global:
                    iters_to_use = global_iters
                else:
                    panel_it_labels = cols[0].multiselect(f"Iterations", iter_labels, default=[iter_labels[0]] if iter_labels else [], key=f"it_{r}_{c}")
                    iters_to_use = [iter_label_map[l] for l in panel_it_labels]
            elif p_type in ["Raw Image", "Signal Decomposition"]:
                # Single Iteration Selection usually
                sit_label = cols[0].selectbox(f"Iteration", iter_labels, key=f"sit_{r}_{c}")
                iters_to_use = [iter_label_map[sit_label]] if sit_label else []

            spec = {
                "iters": iters_to_use,
                "type": p_type,
                "row": r, "col": c,
                "show_comp": False,
                "ylim": None
            }
            
            # Type Specific Config
            if p_type == "Raw Image":
                if len(iters_to_use) > 0:
                    first_iter = iters_to_use[0]
                    meas_files = measurements[first_iter]
                    sel_file_key = st.selectbox(f"Image Key", list(meas_files.keys()), key=f"mk_{r}_{c}")
                    spec["file"] = meas_files[sel_file_key]
                    if len(iters_to_use) > 1: st.caption("Only first iteration shown.")
                else:
                    st.warning("Select iteration.")
            
            elif p_type == "Computed Data":
                var_options = ["I_PV", "I_PH", "Depolarization Ratio", "S11", "S12/S11", "S33/S11", "S34/S11", "DoLP"]
                spec["variable"] = st.selectbox(f"Variable", var_options, key=f"var_{r}_{c}")
                
                c_opt1, c_opt2 = st.columns(2)
                spec["show_std"] = c_opt1.checkbox(f"Show Std Dev", value=st.session_state.get('global_show_std', len(iters_to_use)>1), key=f"std_{r}_{c}")
                
                # Comparison Overlay Option
                if st.session_state.get('multi_comparisons'):
                     spec["show_comp"] = c_opt2.checkbox(f"Overlay Comp(s)", value=False, key=f"cp_{r}_{c}")
                
                # Advanced Plot Options
                with st.expander("Advanced Options (Scaling & Labels)"):
                    c_min, c_max = st.columns(2)
                    ymin = c_min.number_input("Y Min", value=0.0, step=0.1, key=f"ymn_{r}_{c}")
                    ymax = c_max.number_input("Y Max", value=1.0, step=0.1, key=f"ymx_{r}_{c}")
                    use_manual = st.checkbox("Enable Manual Scale", value=False, key=f"ms_{r}_{c}")
                    if use_manual:
                        spec["ylim"] = (ymin, ymax)
                    
                    c_sf, c_pl = st.columns(2)
                    # Scale factor default 0.0001 if loaded from config or 1.0
                    def_sf = st.session_state.get(f"sf_{r}_{c}", 1.0)
                    spec["scale_factor"] = c_sf.number_input("Scale Factor", value=float(def_sf), format="%.1e", key=f"sf_{r}_{c}")
                    
                    # Panel Title / Custom Y-Label
                    def_yl = st.session_state.get(f"yl_{r}_{c}", "")
                    spec["panel_title"] = c_pl.text_input("Custom Panel/Y Label", value=def_yl, key=f"yl_{r}_{c}")

            elif p_type == "Signal Decomposition":
                 if len(iters_to_use) > 0:
                    meas_files = measurements[iters_to_use[0]]
                    def_key = 'I_PV' if 'I_PV' in meas_files else list(meas_files.keys())[0]
                    spec["key"] = st.selectbox(f"Signal Image", list(meas_files.keys()), index=list(meas_files.keys()).index(def_key) if def_key in meas_files else 0, key=f"dk_{r}_{c}")
                    spec["file"] = meas_files[spec["key"]]
                    spec["col_idx"] = st.number_input(f"Column Index", min_value=0, value=500, step=1, key=f"ci_{r}_{c}")

            # Per-panel label override
            use_override = st.checkbox(f"Override Axis Labels", key=f"ov_{r}_{c}")
            if use_override:
                spec["xlabel"] = st.text_input(f"X Label", value=global_xlabel, key=f"xl_{r}_{c}")
                spec["ylabel"] = st.text_input(f"Y Label", value=global_ylabel, key=f"yl_{r}_{c}")
            
            return spec

    # Render Grid Layout
    
    if layout_type == "Single Panel":
        fig_specs[(1,1)] = _render_panel_config(1,1)
        
    elif layout_type == "1x2 Grid (V)":
        fig_specs[(1,1)] = _render_panel_config(1,1)
        fig_specs[(2,1)] = _render_panel_config(2,1)
        
    elif layout_type == "2x1 Grid (H)":
        gc1, gc2 = st.columns(2)
        with gc1:
            fig_specs[(1,1)] = _render_panel_config(1,1)
        with gc2:
            fig_specs[(1,2)] = _render_panel_config(1,2)
            
    elif layout_type == "2x2 Grid":
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            fig_specs[(1,1)] = _render_panel_config(1,1)
        with r1_c2:
            fig_specs[(1,2)] = _render_panel_config(1,2)
            
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            fig_specs[(2,1)] = _render_panel_config(2,1)
        with r2_c2:
            fig_specs[(2,2)] = _render_panel_config(2,2)

    # Final Figure Generation
    fig = generate_composite_figure(
        fig_specs, layout_type, w, h, unit, 
        check_dependencies=check_dependencies,
        common_labels=common_labels,
        global_xlabel=global_xlabel, global_ylabel=global_ylabel,
        active_style=active_style,
        colors=colors,
        primary_label=primary_label,
        show_legend=show_legend,
        legend_loc=legend_loc,
        disable_sci_angle=disable_sci_angle,
        angle_range=angle_range,
        angle_model=angle_model,
        pixel_offset=PIXEL_OFFSET_X,
        anchor_to_primary=st.session_state.get('global_anchor', False),
        subtract_wall=st.session_state.get('global_subtract_wall', False)
    )
    st.session_state.fig_preview = fig

    st.divider()
    
    if 'fig_preview' in st.session_state:
        st.pyplot(st.session_state.fig_preview)
        
        # Export Controls
        st.subheader("Export")
        e_col1, e_col2 = st.columns(2)
        fmt = e_col1.selectbox("Format", ["png", "pdf", "svg", "eps"])
        dpi = e_col2.number_input("DPI", 100, 600, 300)
        
        try:
            buf = io.BytesIO()
            st.session_state.fig_preview.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
            buf.seek(0)
            st.download_button(label=f"Download .{fmt}", data=buf, file_name=f"figure.{fmt}", mime=f"image/{fmt}" if fmt != 'pdf' else 'application/pdf')
        except Exception as e: st.error(f"Export failed: {e}")

        st.divider()
        with st.expander("Save to Local Output Directory", expanded=False):
            ex_c1, ex_c2 = st.columns([0.7, 0.3])
            out_dir = ex_c1.text_input("Output Directory", value=os.getcwd())
            out_fname = ex_c2.text_input("Filename", value="figure_export")
            
            if st.button("Save to Disk"):
                if os.path.isdir(out_dir):
                    final_path = os.path.join(out_dir, f"{out_fname}.{fmt}")
                    try:
                        st.session_state.fig_preview.savefig(final_path, format=fmt, dpi=dpi, bbox_inches='tight')
                        st.success(f"Successfully saved to: {final_path}")
                    except Exception as e:
                        st.error(f"Error saving file: {e}")
                else:
                    st.error(f"Directory not found: {out_dir}")

            if st.session_state.get('multi_comparisons'):
                if st.button("Generate Double Export (_base & _symmetric)"):
                    if not os.path.exists(out_dir): 
                        st.error("Invalid Output Directory")
                    else:
                        try:
                            # 1. Base Figure
                            orig_mc = check_dependencies.get('multi_comparisons', [])
                            check_dependencies['multi_comparisons'] = []
                            fig_base = generate_composite_figure(
                                fig_specs, layout_type, w, h, unit, 
                                check_dependencies=check_dependencies,
                                common_labels=common_labels,
                                global_xlabel=global_xlabel, global_ylabel=global_ylabel,
                                active_style=active_style, colors=colors, primary_label=primary_label,
                                show_legend=show_legend, legend_loc=legend_loc, disable_sci_angle=disable_sci_angle,
                                angle_range=angle_range, angle_model=angle_model, pixel_offset=PIXEL_OFFSET_X,
                                anchor_to_primary=st.session_state.get('global_anchor', False),
                                subtract_wall=st.session_state.get('global_subtract_wall', False)
                            )
                            base_path = os.path.join(out_dir, f"{out_fname}_base.{fmt}")
                            fig_base.savefig(base_path, format=fmt, dpi=dpi, bbox_inches='tight')
                            plt.close(fig_base)

                            # 2. Symmetric Figure
                            check_dependencies['multi_comparisons'] = orig_mc
                            fig_sym = generate_composite_figure(
                                fig_specs, layout_type, w, h, unit, 
                                check_dependencies=check_dependencies,
                                common_labels=common_labels,
                                global_xlabel=global_xlabel, global_ylabel=global_ylabel,
                                active_style=active_style, colors=colors, primary_label=primary_label,
                                show_legend=show_legend, legend_loc=legend_loc, disable_sci_angle=disable_sci_angle,
                                angle_range=angle_range, angle_model=angle_model, pixel_offset=PIXEL_OFFSET_X,
                                anchor_to_primary=st.session_state.get('global_anchor', False),
                                subtract_wall=st.session_state.get('global_subtract_wall', False)
                            )
                            sym_path = os.path.join(out_dir, f"{out_fname}_symmetric.{fmt}")
                            fig_sym.savefig(sym_path, format=fmt, dpi=dpi, bbox_inches='tight')
                            plt.close(fig_sym)
                            
                            st.success(f"Generated double export:\n- {os.path.basename(base_path)}\n- {os.path.basename(sym_path)}")
                        except Exception as e:
                            st.error(f"Double export failed: {e}")
                            check_dependencies['multi_comparisons'] = orig_mc
