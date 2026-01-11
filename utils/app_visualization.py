import streamlit as st
import plotly.graph_objects as go
import plotly.colors
import numpy as np
from PIL import Image
import os
from utils import polarimeter_processing
from utils import app_utils

def render_raw_images_tab(measurements, backgrounds, comparisons, selected_iter_option, is_avg, req_meas, bg_subtraction_on, roi_paths_to_show):
    st.subheader("Raw Measurement Images")
    toggle_cols = st.columns(3)
    show_subtracted = toggle_cols[0].toggle("Show Background Subtracted Image", key="raw_subtracted", help="If enabled, displays the raw image after the mean background has been subtracted. The main 'Background Subtraction' toggle must also be on.")
    show_regions = toggle_cols[1].toggle("Show Integration Regions", key="raw_regions", help="Overlays the signal and wall-effect integration regions on the image.")
    show_profile = toggle_cols[2].toggle("Show Vertical Profile", key="show_profile", help="Enables a slider to select a column and view its vertical intensity profile.")
    img_cols = st.columns(4)
    
    # Dataset Selector for Images
    dataset_options = ["Primary"] + [f"{app_utils.translate_sequence_name(c['name'])} ({c['name']})" for c in comparisons]
    selected_dataset_label = img_cols[0].selectbox("Dataset", dataset_options, key="img_dataset")
    
    # Decode selection
    selected_dataset_name = "Primary"
    if selected_dataset_label != "Primary":
        selected_dataset_name = selected_dataset_label.rsplit(" (", 1)[-1][:-1]
    
    active_measurements = measurements or {}
    active_backgrounds = backgrounds or {}
    
    if selected_dataset_name != "Primary":
        for c in comparisons:
            if c['name'] == selected_dataset_name:
                active_measurements = c.get('measurements') or {}
                active_backgrounds = c.get('backgrounds') or {}
                break

    image_color_scale = img_cols[1].radio("Scale", ["Linear", "Log", "Symlog"], horizontal=True, key="img_scale")
    image_range_mode = img_cols[2].radio("Range", ["Auto", "Fixed"], horizontal=True, key="img_range")
    colormap_options = ['plasma', 'cividis', 'viridis', 'magma', 'inferno']
    selected_colormap = img_cols[3].selectbox("Colormap", colormap_options, key="img_cmap")
    
    # Determine image width for slider
    img_w_ref = 3200
    if active_measurements:
        # Try to find a valid image to get width
        for i_data in active_measurements.values():
            if i_data:
                try:
                    with Image.open(list(i_data.values())[0]) as tmp: img_w_ref = tmp.width; break
                except: continue
    
    profile_col = 0
    if show_profile:
        slider_col, _ = st.columns([0.8, 0.2])
        with slider_col:
            sl_pad_l, sl_main, sl_pad_r = st.columns([0.05, 0.9, 0.05])
            with sl_main:
                profile_col = st.slider("Select Column for Profile", 0, img_w_ref - 1, img_w_ref // 2, key="profile_col")

    # Handle iteration selection for comparisons
    iter_to_show = selected_iter_option if not is_avg else (sorted(active_measurements.keys())[0] if active_measurements else None)
    if iter_to_show not in active_measurements and active_measurements: iter_to_show = sorted(active_measurements.keys())[0]
    
    iter_files = active_measurements.get(iter_to_show, {})
    
    for meas_type in req_meas:
        signal_img, display_filename = None, None
        if is_avg:
            all_files_for_meas = [iter_dict[meas_type] for iter_dict in active_measurements.values() if meas_type in iter_dict]
            if all_files_for_meas:
                signal_img = np.mean([np.array(Image.open(f)).astype(np.float64) for f in all_files_for_meas], axis=0)
                display_filename = f"Average of {len(all_files_for_meas)} images"
        elif meas_type in iter_files:
            signal_img = np.array(Image.open(iter_files[meas_type]))
            display_filename = os.path.basename(iter_files[meas_type])

        if signal_img is not None:
            if show_subtracted:
                if not bg_subtraction_on:
                    st.warning("Cannot show subtracted image because the main 'Background Subtraction' toggle is off.", icon="⚠️")
                else:
                    receiver_key = app_utils.get_receiver_key_for_measurement(meas_type)
                    bg_files = active_backgrounds.get(receiver_key)
                    if bg_files:
                        bg_imgs = [np.array(Image.open(f)) for f in bg_files]
                        mean_bg, _ = polarimeter_processing.calculate_background_stats(bg_imgs)
                        signal_img = signal_img.astype(np.float64) - mean_bg
                    else:
                        st.warning(f"No background files found for {meas_type} to perform subtraction.", icon="⚠️")

            z_data = signal_img.astype(float)
            
            raw_colors = getattr(plotly.colors.sequential, selected_colormap.capitalize())
            scale = [[i / (len(raw_colors) - 1), color] for i, color in enumerate(raw_colors)]
            scale[0][1] = '#FFFFFF'

            if image_color_scale == "Log":
                z_data[z_data <= 0] = np.nan
                log_z_data = np.log10(z_data)
                cmin, cmax = (0, np.log10(65520)) if image_range_mode == "Fixed" else (np.nanmin(log_z_data), np.nanmax(log_z_data))
                tickvals = [v for v in range(int(np.ceil(cmin)), int(np.floor(cmax)) + 1) if v >= 0]
                ticktext = [f"{10**v:.0e}" for v in tickvals]
                fig_img = go.Figure(go.Heatmap(z=log_z_data, colorscale=scale, zmin=cmin, zmax=cmax, colorbar={'title': 'Intensity', 'tickvals': tickvals, 'ticktext': ticktext, 'lenmode': 'fraction', 'len': 0.6, 'yanchor': 'middle', 'y': 0.5}))
            elif image_color_scale == "Symlog":
                sym_thresh = st.session_state.get('symlog_threshold', 10000.0)
                symlog_z_data = np.arcsinh(z_data / sym_thresh)
                cmin, cmax = (np.arcsinh(-65520 / sym_thresh), np.arcsinh(65520 / sym_thresh)) if image_range_mode == "Fixed" else (np.nanmin(symlog_z_data), np.nanmax(symlog_z_data))
                tick_values = [-100000, -10000, -1000, 0, 1000, 10000, 100000]
                tick_positions = [np.arcsinh(v / sym_thresh) for v in tick_values]
                fig_img = go.Figure(go.Heatmap(z=symlog_z_data, colorscale=scale, zmin=cmin, zmax=cmax, colorbar={'title': 'Intensity', 'tickvals': tick_positions, 'ticktext': [f"{v:.0e}" for v in tick_values], 'lenmode': 'fraction', 'len': 0.6, 'yanchor': 'middle', 'y': 0.5}))
            else:
                cmin, cmax = (0, np.nanmax(z_data)) if image_range_mode == "Auto" else (0, 65520)
                fig_img = go.Figure(go.Heatmap(z=z_data, colorscale=scale, zmin=cmin, zmax=cmax, colorbar={'title': 'Intensity', 'tickformat': '.2e', 'lenmode': 'fraction', 'len': 0.6, 'yanchor': 'middle', 'y': 0.5}))
                
            if show_regions:
                if selected_dataset_name == "Primary" and roi_paths_to_show and all(k in roi_paths_to_show for k in ['roi_path', 'roi_top', 'roi_bottom']):
                    x_coords = np.arange(len(roi_paths_to_show['roi_path']))
                    fig_img.add_trace(go.Scatter(x=x_coords, y=roi_paths_to_show['roi_top'], mode='lines', line=dict(color='#00CC96', width=1, dash='dash'), showlegend=False, name='ROI Upper'))
                    fig_img.add_trace(go.Scatter(x=x_coords, y=roi_paths_to_show['roi_bottom'], mode='lines', line=dict(color='#00CC96', width=1, dash='dash'), showlegend=False, name='ROI Lower'))
                    fig_img.add_trace(go.Scatter(x=x_coords, y=roi_paths_to_show['roi_path'], mode='lines', line=dict(color='red', width=1.5), name='Signal Center'))

            if show_profile:
                fig_img.add_vline(x=profile_col, line_width=2, line_dash="dash", line_color="cyan")

            unique_key_base = f"{meas_type}_{selected_iter_option}_{selected_dataset_name}"
            
            if show_profile:
                col_img, col_prof = st.columns([0.8, 0.2])
            else:
                col_img = st.container()
                col_prof = None
            
            with col_img:
                fig_img.update_layout(title=f"{meas_type} ({display_filename})", yaxis_title="Pixel", xaxis_title="Pixel", height=500, margin=dict(t=50, b=50), yaxis=dict(autorange='reversed'))
                st.plotly_chart(fig_img, use_container_width=True, key=f"{unique_key_base}_img")
            
            if show_profile and col_prof:
                with col_prof:
                    if signal_img is not None and profile_col < signal_img.shape[1]:
                        col_data = signal_img[:, profile_col]
                        fig_prof = go.Figure()
                        fig_prof.add_trace(go.Scatter(x=col_data, y=np.arange(len(col_data)), mode='lines', name='Profile', line=dict(color='blue')))
                        fig_prof.update_layout(title="Profile", xaxis_title="Intensity", height=500, margin=dict(l=10, r=10, t=50, b=50), yaxis=dict(showticklabels=False, range=[len(col_data), 0]))
                        st.plotly_chart(fig_prof, use_container_width=True, key=f"{unique_key_base}_prof")

def render_background_images_tab(backgrounds, comparisons, req_meas, bg_subtraction_on):
    st.subheader("Background Images")
    st.info("This tab shows the mean and standard deviation of the background images used for subtraction.")
    toggle_cols = st.columns(2)
    show_regions_bg = toggle_cols[1].toggle("Show Integration Regions", key="bg_regions", help="Overlays the signal and wall-effect integration regions on the image.")
    img_cols = st.columns(4)
    bg_dataset_options = ["Primary"] + [f"{app_utils.translate_sequence_name(c['name'])} ({c['name']})" for c in comparisons]
    bg_selected_dataset_label = img_cols[0].selectbox("Dataset", bg_dataset_options, key="bg_dataset")
    
    bg_selected_dataset_name = "Primary"
    if bg_selected_dataset_label != "Primary":
        bg_selected_dataset_name = bg_selected_dataset_label.rsplit(" (", 1)[-1][:-1]
    active_bg_view = backgrounds or {}
    if bg_selected_dataset_name != "Primary":
        for c in comparisons:
            if c['name'] == bg_selected_dataset_name:
                active_bg_view = c.get('backgrounds') or {}
                break
    bg_image_color_scale = img_cols[1].radio("Scale", ["Linear", "Log", "Symlog"], horizontal=True, key="bg_img_scale")
    bg_image_range_mode = img_cols[2].radio("Range", ["Auto", "Fixed"], horizontal=True, key="bg_img_range")
    colormap_options = ['plasma', 'cividis', 'viridis', 'magma', 'inferno']
    bg_selected_colormap = img_cols[3].selectbox("Colormap", colormap_options, key="bg_cmap")
    unique_receiver_keys = set(app_utils.get_receiver_key_for_measurement(m) for m in req_meas)
    if not bg_subtraction_on:
        st.warning("Background subtraction is turned off. No background images are being used.")
    elif not active_bg_view:
        st.warning("No background files were found or loaded.")
    else:
        for r_key in sorted(list(unique_receiver_keys)):
            bg_files = active_bg_view.get(r_key)
            if bg_files:
                st.markdown(f"--- \n #### Receiver State: RP={r_key[0]}, RW={r_key[1]}")
                st.write(f"Found {len(bg_files)} background image(s).")
                bg_imgs = [np.array(Image.open(f)) for f in bg_files]
                mean_bg, std_bg = polarimeter_processing.calculate_background_stats(bg_imgs)
                for title, data in [("Mean Background", mean_bg), ("Background Noise Map (Std. Dev.)", std_bg)]:
                    z_data = data.astype(float)
                    raw_colors = getattr(plotly.colors.sequential, bg_selected_colormap.capitalize())
                    scale = [[i / (len(raw_colors) - 1), color] for i, color in enumerate(raw_colors)]; scale[0][1] = '#FFFFFF'

                    if bg_image_color_scale == "Log":
                        z_data[z_data <= 0] = np.nan
                        log_z_data = np.log10(z_data)
                        cmin, cmax = (0, np.log10(65520)) if bg_image_range_mode == "Fixed" else (np.nanmin(log_z_data), np.nanmax(log_z_data))
                        tickvals = [v for v in range(int(np.ceil(cmin)), int(np.floor(cmax)) + 1) if v >= 0]
                        ticktext = [f"{10**v:.0e}" for v in tickvals]
                        fig = go.Figure(go.Heatmap(z=log_z_data, colorscale=scale, zmin=cmin, zmax=cmax, colorbar={'title': 'Intensity', 'tickvals': tickvals, 'ticktext': ticktext, 'lenmode': 'fraction', 'len': 0.6, 'yanchor': 'middle', 'y': 0.5}))
                    elif bg_image_color_scale == "Symlog":
                        sym_thresh = st.session_state.get('symlog_threshold', 10000.0)
                        symlog_z_data = np.arcsinh(z_data / sym_thresh)
                        cmin, cmax = (np.arcsinh(-65520 / sym_thresh), np.arcsinh(65520 / sym_thresh)) if bg_image_range_mode == "Fixed" else (np.nanmin(symlog_z_data), np.nanmax(symlog_z_data))
                        tick_values = [-100000, -10000, -1000, 0, 1000, 10000, 100000]
                        tick_positions = [np.arcsinh(v / sym_thresh) for v in tick_values]
                        fig = go.Figure(go.Heatmap(z=symlog_z_data, colorscale=scale, zmin=cmin, zmax=cmax, colorbar={'title': 'Intensity', 'tickvals': tick_positions, 'ticktext': [f"{v:.0e}" for v in tick_values], 'lenmode': 'fraction', 'len': 0.6, 'yanchor': 'middle', 'y': 0.5}))
                    else:  # Linear scale
                        cmin, cmax = (0, np.nanmax(z_data)) if bg_image_range_mode == "Auto" else (0, 65520)
                        fig = go.Figure(go.Heatmap(z=z_data, colorscale=scale, zmin=cmin, zmax=cmax, colorbar={'title': 'Intensity', 'tickformat': '.2e', 'lenmode': 'fraction', 'len': 0.6, 'yanchor': 'middle', 'y': 0.5}))
                    
                    # ADDED: Overlay integration regions if toggled.
                    if show_regions_bg:
                        img_h, img_w = z_data.shape
                        signal_top, signal_bottom = img_h // 3, 2 * img_h // 3
                        fig.add_shape(type="rect", x0=-0.5, y0=signal_top-0.5, x1=img_w-0.5, y1=signal_bottom-0.5, line=dict(color="#00CC96", width=2), fillcolor="rgba(0, 204, 150, 0.1)")
                        fig.add_shape(type="rect", x0=-0.5, y0=-0.5, x1=img_w-0.5, y1=signal_top-0.5, line=dict(color="red", width=2), fillcolor="rgba(255, 0, 0, 0.1)")
                        fig.add_shape(type="rect", x0=-0.5, y0=signal_bottom-0.5, x1=img_w-0.5, y1=img_h-0.5, line=dict(color="red", width=2), fillcolor="rgba(255, 0, 0, 0.1)")

                    fig.update_layout(title=title, yaxis_title="Pixel", xaxis_title="Pixel")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No background files found for receiver state: RP={r_key[0]}, RW={r_key[1]}")