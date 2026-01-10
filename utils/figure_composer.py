import streamlit as st
import os # Standard library
import json
import plotly.graph_objects as go
from plotly import subplots
import matplotlib.pyplot as plt
import io
from utils import plotting, app_computation, app_utils, polarimeter_processing
import numpy as np
from PIL import Image

def render_figure_composer(measurements):
    """
    Renders the Figure Composer Interface.
    """
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
                                    k.startswith(('global_', 'pt_', 'ug_', 'it_', 'sit_', 'mk_', 'var_', 'std_', 'cp_', 'ms_', 'ymn_', 'ymx_', 'dk_', 'ci_', 'ov_', 'xl_', 'yl_')) or 
                                    k in ['angle_range']]
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
                        st.session_state.angle_range = tuple(data.get('angle_range', (100.0, 168.0))) # Ensure tuple
                        st.rerun()
                    except Exception as e:
                        st.error(f"Load failed: {e}")
            else:
                st.info("No saved configurations found.")

    # --- 1. Global Settings ---
    with st.expander("Global Chart Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        # Dimensions
        unit = c1.selectbox("Unit", ["inch", "px"], index=0, key="global_unit")
        if unit == "inch":
            w = c2.number_input("Width (in)", value=6.0, step=0.1, key="global_w_in")
            h = c3.number_input("Height (in)", value=4.0, step=0.1, key="global_h_in")
        else:
            w = c2.number_input("Width (px)", value=1000, step=100, key="global_w_px")
            h = c3.number_input("Height (px)", value=800, step=100, key="global_h_px")
            
        # Angle Range
        if 'angle_range' not in st.session_state: st.session_state.angle_range = (100.0, 168.0)
        angle_range = st.slider("Global Angle Range [deg]", 50.0, 170.0, st.session_state.angle_range, key="angle_range_widget")
        st.session_state.angle_range = angle_range # Sync
        
        # Style & Scaling
        c_s1, c_s2, c_s3 = st.columns(3)
        style_name = c_s1.selectbox("Style Preset", list(plotting.STYLES.keys()), index=0, key="global_style")
        font_scale = c_s2.number_input("Font Scale", 0.5, 2.0, 1.0, 0.1, key="global_font_scale")
        disable_sci_angle = c_s3.checkbox("No Sci Notation (X)", value=True, key="global_no_sci")
        
        selected_style = plotting.STYLES[style_name]
        # Apply scaling to style copy
        import copy
        active_style = copy.deepcopy(selected_style)
        active_style.font_size = int(active_style.font_size * font_scale)
        
        # Label Settings
        common_labels = st.checkbox("Common Labels", value=True, help="Hide inner axis labels in grid layouts.", key="global_common_labels")
        global_xlabel = st.text_input("Global X Label", value="Scattering Angle [deg]", key="global_xlabel")
        global_ylabel = st.text_input("Global Y Label", value="Intensity [a.u.]", key="global_ylabel")
        
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
        scheme_name = st.selectbox("Color Scheme", list(COLOR_SCHEMES.keys()), index=0, key="global_color_scheme")
        
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
        show_legend = ll_c1.checkbox("Show Legend", value=True, key="lbl_show")
        legend_loc = ll_c2.selectbox("Legend Loc", ["best", "upper right", "upper left", "lower left", "lower right", "center right"], index=0, key="lbl_loc")

        st.markdown("---")
        st.markdown("**Primary Data Iterations**")
# ... [Omitted iteration selection lines for brevity, need to match exact context] ...
# Actually better to just update the specific blocks rather than one giant replace if context is far apart.
# Context is line 84 to 90.

# AND update lines ~400 and ~420 for plotting.
# I will do this in 2 chunks to be safe.

        all_iters = sorted(measurements.keys())
        ic1, ic2 = st.columns([0.2, 0.8])
        select_all_iters = ic1.checkbox("Select All", value=False, key="global_select_all")
        default_iters = all_iters if select_all_iters else ([all_iters[0]] if all_iters else [])
        
        if select_all_iters:
             global_iters = ic2.multiselect("Iterations", all_iters, default=all_iters, disabled=True, key="global_iters_disabled")
             global_iters = all_iters
        else:
             global_iters = ic2.multiselect("Iterations", all_iters, default=default_iters, key="global_iters")


    # --- 1b. Comparison Data Source ---
    with st.expander("Comparison Data Source", expanded=True):
        comp_path = st.text_input("Comparison Folder Path", value="")
        if st.button("Load Comparison Measurement"):
            if os.path.isdir(comp_path):
                # Let's assume input is the PARENT folder containing 'measurements' and 'backgrounds'
                m_path = os.path.join(comp_path, 'measurements')
                b_path = os.path.join(comp_path, 'backgrounds')
                if os.path.exists(m_path) and os.path.exists(b_path):
                     c_meas, c_bg, err = polarimeter_processing.find_and_organize_files(m_path, b_path)
                     if c_meas:
                         st.session_state.comp_measurements = c_meas
                         st.session_state.comp_backgrounds = c_bg
                         st.success(f"Loaded {len(c_meas)} iterations for comparison.")
                     else: st.error(err)
                else:
                    st.error("Could not find 'measurements' and 'backgrounds' subfolders in the specified path.")
            else:
                st.error("Invalid Path: Folder does not exist.")

        # Comparison Iterations
        global_comp_iters = []
        if 'comp_measurements' in st.session_state and st.session_state.comp_measurements:
            st.markdown("**Comparison Iterations**")
            c_iters = sorted(st.session_state.comp_measurements.keys())
            cc1, cc2 = st.columns([0.2, 0.8])
            comp_select_all = cc1.checkbox("Select All (Comp)", value=True, key="comp_select_all")
            if comp_select_all:
                global_comp_iters = cc2.multiselect("Comp Iters", c_iters, default=c_iters, disabled=True, key="global_comp_iters_disabled")
                global_comp_iters = c_iters # Ensure it's all even if disabled
            else:
                default_comp_iters = [c_iters[0]] if c_iters else []
                global_comp_iters = cc2.multiselect("Comp Iters", c_iters, default=default_comp_iters, key="global_comp_iters")
        else:
            st.info("Load comparison data to enable comparison iteration selection.")


    # --- 2. Layout & Content ---
    layout_type = st.radio("Layout", ["Single Panel", "1x2 Grid (V)", "2x1 Grid (H)", "2x2 Grid"], index=0, horizontal=True)
    
    slots = []
    if layout_type == "Single Panel": slots = [(1,1)]
    elif layout_type == "1x2 Grid (V)": slots = [(1,1), (2,1)]
    elif layout_type == "2x1 Grid (H)": slots = [(1,1), (1,2)]
    elif layout_type == "2x2 Grid": slots = [(1,1), (1,2), (2,1), (2,2)]
    
    fig_specs = {}
    
    # Load Angle Model globally for slicing
    angle_model = None
    if hasattr(app_utils, 'load_angle_model_from_npz'):
        angle_model = app_utils.load_angle_model_from_npz(app_utils.CALIBRATION_FILE_PATH)

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
                    iters_to_use = cols[0].multiselect(f"Iterations", all_iters, default=[all_iters[0]], key=f"it_{r}_{c}")
            elif p_type in ["Raw Image", "Signal Decomposition"]:
                # Single Iteration Selection usually
                iters_to_use = [cols[0].selectbox(f"Iteration", all_iters, key=f"sit_{r}_{c}")]

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
                spec["show_std"] = c_opt1.checkbox(f"Show Std Dev", value=len(iters_to_use)>1, key=f"std_{r}_{c}")
                
                # Comparison Overlay Option
                if global_comp_iters and 'comp_measurements' in st.session_state and st.session_state.comp_measurements:
                     spec["show_comp"] = c_opt2.checkbox(f"Overlay Comp", value=False, key=f"cp_{r}_{c}")
                
                # Manual Scaling
                with st.expander("Manual Axis Scaling (Optional)"):
                    c_min, c_max = st.columns(2)
                    ymin = c_min.number_input("Y Min", value=0.0, step=0.1, key=f"ymn_{r}_{c}")
                    ymax = c_max.number_input("Y Max", value=1.0, step=0.1, key=f"ymx_{r}_{c}")
                    use_manual = st.checkbox("Enable Manual Scale", value=False, key=f"ms_{r}_{c}")
                    if use_manual:
                        spec["ylim"] = (ymin, ymax)

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

    st.divider()
    
    # 3. Composition
    if st.button("Generate Preview"):
        rows = 2 if "2x" in layout_type or "1x2" in layout_type else 1
        cols = 2 if "2x2" in layout_type or "2x1" in layout_type else 1
        
        # Dimensions
        if unit == "px":
             final_w, final_h = w / 100.0, h / 100.0 # Convert px to inches for matplotlib
        else:
             final_w, final_h = w, h

        plt.close('all')
        fig = plt.figure(figsize=(final_w, final_h))
        
        # Spacing
        if common_labels: fig.subplots_adjust(wspace=0.25, hspace=0.05)
        else: fig.subplots_adjust(wspace=0.35, hspace=0.35)

        for (r, c), spec in fig_specs.items():
            idx = (r-1)*cols + c
            ax = fig.add_subplot(rows, cols, idx)
            
            # Determine Axes Visibility
            # User requested Y labels on right plots too, as they might differ.
            # So we typically always show Y labels unless explicitly handled otherwise.
            show_y = True 
            show_x = (r == rows) or (not common_labels)
            
            panel_xlabel = spec.get("xlabel", global_xlabel) if show_x else ""
            panel_ylabel = spec.get("ylabel", global_ylabel) if show_y else ""

            try:
                # --- PREPARE DATA ---
                # Resolve X-axis (Angles)
                # Using 1280 pixels standard width assumption if model unused?
                # Better: Use angle_model if available.
                
                if spec["type"] == "Raw Image":
                    # For Heatmap, we need to slice the image to the angle range
                    img = np.array(Image.open(spec["file"])) # (H, W)
                    
                    x_pixel_min, x_pixel_max = 0, img.shape[1]
                    if angle_model:
                        # Map all pixels to angles
                        all_angles = angle_model(np.arange(img.shape[1]))
                        # Find indices within range
                        valid_mask = (all_angles >= angle_range[0]) & (all_angles <= angle_range[1])
                        if np.any(valid_mask):
                             # Get bounds
                             valid_indices = np.where(valid_mask)[0]
                             x_pixel_min, x_pixel_max = valid_indices[0], valid_indices[-1]
                             
                             # Adjust Image
                             img = img[:, x_pixel_min:x_pixel_max+1]
                    
                    # Plot
                    plotting.create_mpl_heatmap(
                        img, ax, cmap='viridis', 
                        title=f"({r},{c})", 
                        xlabel=panel_xlabel, ylabel=panel_ylabel,
                        show_x=show_x, show_y=show_y,
                        style=active_style
                    )

                elif spec["type"] == "Signal Decomposition":
                     # Decomposition is vertical, so X-axis is Intensity, Y-axis is Pixel Height
                     # Angle range doesn't apply here usually (it's a column).
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
                    # Function to process a list of iterations and return Mean/Std/X
                    def get_series_data(t_iters, t_meas, t_bg, t_ana_type, t_var):
                        t_curves = []
                        t_x = None
                        
                        if not t_iters: return None, None, None
                        
                        for it in t_iters:
                            # Fetch Data Logic
                            temp_meas = {it: t_meas[it]}
                            req = app_utils.get_required_measurements(temp_meas, t_ana_type)
                            
                            iter_res = None
                            # For primary data, check precomputed cache
                            if t_meas is st.session_state.measurements and 'precomputed_data' in st.session_state:
                                 cached = st.session_state.precomputed_data.get(str(it))
                                 if cached and (('I_PV' in cached) or ('Depol_Parallel' in cached)): 
                                     iter_res = cached
                            
                            # If not cached or for comparison data, compute
                            if iter_res is None:
                                calc_res, _ = app_computation.calculate_curves_for_iteration(
                                    it, t_meas, t_bg, req, t_ana_type, 
                                    True, 3.0, 1e-4, angle_model or app_utils.load_angle_model_from_npz(app_utils.CALIBRATION_FILE_PATH)
                                )
                                if calc_res: iter_res = calc_res
                            
                            if not iter_res: continue
                            
                            y_val = None
                            # Calculate derived vars if needed
                            mm_elems = polarimeter_processing.calculate_mueller_elements(iter_res)
                            
                            if t_var in iter_res: y_val = iter_res[t_var]
                            elif mm_elems and t_var in mm_elems: y_val = mm_elems[t_var]
                            elif t_var == "Depolarization Ratio":
                                if 'Depol_Parallel' in iter_res and 'Depol_Cross' in iter_res:
                                    y_val = (iter_res['Depol_Cross']+1e-9)/(iter_res['Depol_Parallel']+1e-9)
                            
                            if y_val is not None:
                                # Re-map X to Angle if model exists
                                if t_x is None:
                                    if angle_model:
                                        t_x = angle_model(np.arange(len(y_val)))
                                    else:
                                        t_x = np.arange(len(y_val))
                                t_curves.append(y_val)
                        
                        if not t_curves or t_x is None: return None, None, None
                        
                        # Stack and calculate mean/std
                        stack = np.vstack(t_curves) # (N_iter, N_pixels)
                        mean_curve = np.mean(stack, axis=0)
                        std_curve = np.std(stack, axis=0) if len(t_curves) > 1 else None
                        
                        # Filter by Angle Range
                        mask = (t_x >= angle_range[0]) & (t_x <= angle_range[1])
                        return t_x[mask], mean_curve[mask], std_curve[mask] if std_curve is not None else None

                    # Determine Analysis Type (based on first iteration's data keys)
                    # This is a heuristic, might need refinement if data structure varies wildly
                    ana_type = "Mueller Matrix" 
                    if spec['iters']:
                        first_iter_keys = list(measurements[spec['iters'][0]].keys())
                        if not any('I_PV' in k for k in first_iter_keys): # If no I_PV, assume it's not MM
                            if "Depolarization Ratio" == spec["variable"]:
                                ana_type = "Depolarization Ratio"
                            # else, default to MM and hope for the best or error out

                    # 1. Primary Data
                    x_p, y_p, std_p = get_series_data(spec['iters'], measurements, st.session_state.backgrounds, ana_type, spec['variable'])
                    
                    if x_p is not None:
                         # Plot Primary
                        plotting.create_mpl_line(
                            x_p, y_p, ax, y_err=std_p if spec['show_std'] else None,
                            label=primary_label,
                            xlabel=panel_xlabel, ylabel=panel_ylabel,
                            show_x=show_x, show_y=show_y,
                            style=active_style, color=colors[0],
                            disable_sci_x=disable_sci_angle,
                            ylim=spec.get("ylim")
                        )
                    else:
                        st.warning(f"No valid primary data for {spec['iters']}")
                    
                    # 2. Comparison Data Overlay
                    if spec.get("show_comp") and 'comp_measurements' in st.session_state and st.session_state.comp_measurements:
                         c_meas = st.session_state.comp_measurements
                         c_bg = st.session_state.get('comp_backgrounds', {})
                         c_iters = global_comp_iters # Use globally selected comp iters
                         
                         x_c, y_c, std_c = get_series_data(c_iters, c_meas, c_bg, ana_type, spec['variable'])
                         
                         if x_c is not None:
                             plotting.create_mpl_line(
                                x_c, y_c, ax, y_err=std_c if spec['show_std'] else None, 
                                label=comp_label,
                                color=colors[1],
                                linestyle='--',
                                xlabel=panel_xlabel, ylabel=panel_ylabel, # These will be overridden by primary if not empty
                                show_x=show_x, show_y=show_y,
                                style=active_style,
                                disable_sci_x=disable_sci_angle
                             )
                             if show_legend: ax.legend(loc=legend_loc) # Show legend if comparison exists
                         else:
                            st.warning(f"No valid comparison data for {c_iters}")
                    
                    elif show_legend: # If no comparison but legend requested (e.g. for Primary std dev)
                        ax.legend(loc=legend_loc)
                    
            except Exception as e:
                st.error(f"Error in ({r},{c}): {e}")

        plt.tight_layout()
        st.session_state.fig_preview = fig

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
