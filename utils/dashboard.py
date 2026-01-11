
import streamlit as st
import os
import time
import pandas as pd
import json
import subprocess
import sys
from datetime import datetime
from utils import app_utils, app_computation, polarimeter_processing, pixel_angle_tool

def get_file_mtime(path, compare_time=None):
    if not os.path.exists(path): return 0
    if os.path.isfile(path): return os.path.getmtime(path)
    
    # Directory: Check folder mtime first to avoid expensive scan
    folder_mtime = os.path.getmtime(path)
    if compare_time is not None and folder_mtime <= compare_time:
        return folder_mtime

    # Find latest image modification time (ignore status/json files)
    try:
        max_mtime = 0
        has_images = False
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith(('.png', '.tif', '.tiff', '.bmp')):
                    has_images = True
                    m = entry.stat().st_mtime
                    if m > max_mtime: max_mtime = m
        if has_images: return max_mtime
    except: pass
    return folder_mtime

def format_timestamp(ts):
    if ts == 0: return "N/A"
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')

def scan_datasets(base_path, cache_root):
    """
    Scans the base_path for datasets following the structure:
    DateFolder / MeasurementFolder / SequenceFolder
    """
    datasets = []
    
    if base_path and os.path.exists(base_path):
        try:
            date_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        except OSError:
            date_folders = []
    else:
        date_folders = []
    
    for date_folder in date_folders:
        date_path = os.path.join(base_path, date_folder)
        try:
            subfolders = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
        except OSError:
            continue
        
        # Look for 'measurements' folder or custom measurement parents
        meas_parents = [d for d in subfolders if "meas" in d.lower()]
        
        for mp in meas_parents:
            mp_path = os.path.join(date_path, mp)
            try:
                sequences = [d for d in os.listdir(mp_path) if os.path.isdir(os.path.join(mp_path, d))]
            except OSError:
                continue
            
            for seq in sequences:
                seq_path = os.path.join(mp_path, seq)
                
                # Check for processed data
                proc_std, meta_std, _ = app_utils.try_load_precomputed(seq_path, cache_root, mode="standard")
                proc_dyn, meta_dyn, _ = app_utils.try_load_precomputed(seq_path, cache_root, mode="dynamic")
                
                # Get Background Info
                bg_info = "Unknown"
                if proc_std and meta_std and 'background_info' in meta_std: bg_info = meta_std['background_info']
                elif proc_dyn and meta_dyn and 'background_info' in meta_dyn: bg_info = meta_dyn['background_info']
                
                # Determine status for Standard
                std_status = "Missing"
                std_time = 0
                if proc_std:
                    # Find the file path to check mtime (Cache priority)
                    c_path = os.path.join(cache_root, date_folder, "preprocessed_data", seq, "processed_data.nc")
                    if os.path.exists(c_path): std_time = get_file_mtime(c_path)
                    else:
                        c_path_json = os.path.join(cache_root, date_folder, "preprocessed_data", seq, "processed_data.json")
                        if os.path.exists(c_path_json): std_time = get_file_mtime(c_path_json)
                        else:
                            s_path = os.path.join(seq_path, "processed_data.nc")
                            if os.path.exists(s_path): std_time = get_file_mtime(s_path)
                            else:
                                s_path_json = os.path.join(seq_path, "processed_data.json")
                                if os.path.exists(s_path_json): std_time = get_file_mtime(s_path_json)
                    std_status = "Processed"
                
                # Determine status for Dynamic
                dyn_status = "Missing"
                dyn_time = 0
                if proc_dyn:
                    # Mtime check similar to above
                    c_path = os.path.join(cache_root, date_folder, "preprocessed_data", seq, "processed_data_dynamic.nc")
                    if os.path.exists(c_path): dyn_time = get_file_mtime(c_path)
                    else:
                        c_path_json = os.path.join(cache_root, date_folder, "preprocessed_data", seq, "processed_data_dynamic.json")
                        if os.path.exists(c_path_json): dyn_time = get_file_mtime(c_path_json)
                        else:
                            s_path = os.path.join(seq_path, "processed_data_dynamic.nc")
                            if os.path.exists(s_path): dyn_time = get_file_mtime(s_path)
                            else:
                                s_path_json = os.path.join(seq_path, "processed_data_dynamic.json")
                                if os.path.exists(s_path_json): dyn_time = get_file_mtime(s_path_json)
                    dyn_status = "Processed"

                valid_times = [t for t in [std_time, dyn_time] if t > 0]
                compare_time = min(valid_times) if valid_times else 0
                raw_mtime = get_file_mtime(seq_path, compare_time=compare_time)
                
                # Refine Status
                # Check for Partial
                expected_iters = 0
                try: expected_iters = app_utils.count_iterations(seq_path)
                except: pass
                
                # Helper to check partial
                def check_partial_status(status, data_dict):
                    if status == "Processed" and expected_iters > 0:
                        # Count processed iterations
                        # Data dict keys are iterations
                        if data_dict:
                            processed_cnt = len(data_dict)
                            if processed_cnt < expected_iters:
                                return f"Partial ({processed_cnt}/{expected_iters})"
                    return status

                std_status = check_partial_status(std_status, proc_std)
                dyn_status = check_partial_status(dyn_status, proc_dyn)

                if std_status == "Processed" and raw_mtime > std_time: std_status = "Outdated"
                if dyn_status == "Processed" and raw_mtime > dyn_time: dyn_status = "Outdated"
                
                datasets.append({
                    "Date": date_folder,
                    "Parent": mp,
                    "Sequence": seq,
                    "Description": app_utils.translate_sequence_name(seq),
                    "Path": seq_path,
                    "Standard": std_status,
                    "Standard Time": format_timestamp(std_time),
                    "Dynamic": dyn_status,
                    "Dynamic Time": format_timestamp(dyn_time),
                    "Raw Time": format_timestamp(raw_mtime),
                    "Background Used": bg_info,
                    "Expected Iters": expected_iters,
                    "IsLocal": False
                })

    # --- MERGE LOCAL-ONLY DATASETS ---
    local_datasets = app_utils.scan_local_cache(cache_root)
    existing_keys = {(d['Date'], d['Sequence']) for d in datasets}

    for ld in local_datasets:
        key = (ld['Date'], ld['Sequence'])
        if key in existing_keys:
            continue
        
        # This dataset is in local cache but NOT in the remote scan
        # We need to load its metadata to fill the row
        dummy_path = ld['LocalPath'] # Not actually the raw path, but used as reference for loading
        proc_std, meta_std, _ = app_utils.try_load_precomputed(dummy_path, cache_root, mode="standard")
        proc_dyn, meta_dyn, _ = app_utils.try_load_precomputed(dummy_path, cache_root, mode="dynamic")

        bg_info = "Unknown"
        if meta_std and 'background_info' in meta_std: bg_info = meta_std['background_info']
        elif meta_dyn and 'background_info' in meta_dyn: bg_info = meta_dyn['background_info']

        std_status = "Processed" if proc_std else "Missing"
        dyn_status = "Processed" if proc_dyn else "Missing"
        
        # Reconstruct a likely remote path (best guess for reference)
        ref_path = f"LOCAL_ONLY:{ld['LocalPath']}"

        datasets.append({
            "Date": ld['Date'],
            "Parent": "Local Cache",
            "Sequence": ld['Sequence'],
            "Description": app_utils.translate_sequence_name(ld['Sequence']),
            "Path": ref_path, 
            "Standard": std_status,
            "Standard Time": "N/A",
            "Dynamic": dyn_status,
            "Dynamic Time": "N/A",
            "Raw Time": "Disconnected",
            "Background Used": bg_info,
            "Expected Iters": 0,
            "IsLocal": True
        })
                
    return pd.DataFrame(datasets)

def render_dashboard(data_folder, cache_root):
    st.title("Data Dashboard")
    
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Scanning datasets..."):
        df = scan_datasets(data_folder, cache_root)
        
    if df.empty:
        st.info("No datasets found or data folder not configured.")
        return

    # Filters
    c1, c2, c3, c4 = st.columns([1, 1, 1, 0.6])
    filter_date = c1.multiselect("Filter Date", df["Date"].unique())
    filter_status = c2.multiselect("Filter Status (Standard)", ["Processed", "Missing", "Outdated"])
    filter_bg = c3.multiselect("Filter Background", df["Background Used"].unique())
    show_raw_seq = c4.checkbox("Show Raw IDs", value=False, help="Display the cryptic sequence names alongside descriptions.")
    
    filtered_df = df
    if filter_date: filtered_df = filtered_df[filtered_df["Date"].isin(filter_date)]
    if filter_status: filtered_df = filtered_df[filtered_df["Standard"].apply(lambda x: any(s in x for s in filter_status))]
    if filter_bg: filtered_df = filtered_df[filtered_df["Background Used"].isin(filter_bg)]
        
    # Display
    filtered_df["BgDescription"] = filtered_df["Background Used"].apply(app_utils.translate_full_path)
    
    st.dataframe(
        filtered_df[["Date", "Description", "Standard", "Dynamic", "BgDescription", "Raw Time", "Parent"] + (["Sequence"] if show_raw_seq else [])],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Parent": st.column_config.TextColumn("Source Folder", width="small"),
            "Description": st.column_config.TextColumn("Description", help="Human-readable translation of the sequence name."),
            "Sequence": st.column_config.TextColumn("Sequence ID", width="small"),
            "Standard": st.column_config.TextColumn("Standard", validate="^Processed$|^Missing$|^Outdated$|^Partial.*$"),
            "Dynamic": st.column_config.TextColumn("Dynamic"),
            "BgDescription": st.column_config.TextColumn("Background Used", width="medium", help="Translated background folder and sequence."),
            "Raw Time": st.column_config.TextColumn("Raw Modified")
        }
    )
    
    # Action Panel
    st.divider()
    
    col_left, col_right = st.columns([0.4, 0.6])
    
    # --- PROCESSING PANEL (Replacement for Sidebar Setup) ---
    with col_right:
        st.subheader("Data Processor")
        
        # Dataset Selection for Processing
        proc_opts = filtered_df.apply(lambda x: f"{x['Date']} | {x['Description']} ({x['Sequence']})", axis=1).tolist()
        sel_proc = st.selectbox("Select Dataset to Process", proc_opts, key="proc_sel")
        
        if sel_proc:
            # Decode
            parts = sel_proc.split(" | ", 1)
            date_val = parts[0]
            # Format is Description (Sequence) - use rsplit to handle descriptions with ( or |
            rest = parts[1]
            seq_val = rest.rsplit(" (", 1)[-1][:-1]
            proc_row = filtered_df[(filtered_df["Date"] == date_val) & (filtered_df["Sequence"] == seq_val)].iloc[0]
            
            # Background Discovery (Only if not Local-Only)
            bg_candidates = []
            bg_seq = None
            is_local = proc_row.get('IsLocal', False)

            if not is_local:
                try:
                    date_path = os.path.dirname(os.path.dirname(proc_row['Path']))
                    bg_candidates = [d for d in os.listdir(date_path) if "back" in d.lower() or "bg" in d.lower()]
                except:
                    pass
                
                p_c1, p_c2 = st.columns(2)
                
                # Default to 'background_laser' if present
                bg_def_idx = 0
                for i, c in enumerate(bg_candidates):
                    if "background_laser" in c.lower(): 
                       bg_def_idx = i
                       break
                
                bg_opts = [f"{app_utils.translate_bg_folder(c)} ({c})" for c in bg_candidates]
                bg_parent_sel = p_c1.selectbox("Background Folder", bg_opts, index=bg_def_idx if bg_candidates else None)
                bg_parent = bg_parent_sel.split(" (")[1][:-1] if bg_parent_sel else None
                
                if bg_parent:
                    bg_parent_path = os.path.join(date_path, bg_parent)
                    try:
                        bg_seqs = sorted([d for d in os.listdir(bg_parent_path) if os.path.isdir(os.path.join(bg_parent_path, d))])
                        # Try to Auto-Select Best Background
                        best_bg = app_utils.find_best_background_folder(proc_row['Sequence'], proc_row['Path'], bg_parent_path)
                        def_idx = bg_seqs.index(best_bg) if best_bg in bg_seqs else 0
                        
                        bg_seq_opts = [f"{app_utils.translate_sequence_name(s)} ({s})" for s in bg_seqs]
                        bg_seq_sel = p_c2.selectbox("Background Sequence", bg_seq_opts, index=def_idx)
                        bg_seq = bg_seq_sel.split(" (")[1][:-1] if bg_seq_sel else None
                    except:
                        st.error("Error reading background sequences.")
            else:
                st.info("ℹ️ **Local-Only Dataset**")
                st.caption("Processing is unavailable because the raw measurement files are not present on this machine.")
                
            st.caption(f"Will process: `{proc_row['Sequence']}` using `{bg_seq}`")
            
            # Parameters
            with st.expander("Processing Parameters", expanded=False):
                p_wall = st.checkbox("Subtract Wall (Dynamic)", value=False, help="Enable advanced Double Gaussian fitting to separate beam signal from wall reflections. Recommended for grazing angle measurements. Default: Off")
                
                # Analysis Type Selection
                p_ana_type = st.selectbox("Analysis Type", ["Auto-Detect", "Mueller Matrix", "Depolarization Ratio"], help="Manually force the analysis type if Auto-Detect fails.")
                
                pp_c1, pp_c2 = st.columns(2)
                p_thresh = pp_c1.slider("Log Thresh (ROI)", 0.1, 2.0, 0.6, help="Defines the Region of Interest (ROI) for the beam. Lower values = wider ROI. Default: 0.6")
                
                p_sigma = 3.0 # Default
                if not p_wall:
                    p_sigma = pp_c2.slider("Noise Sigma", 1.0, 5.0, 3.0, help="Standard Deviation threshold for background noise subtraction. Pixels below Mean + Sigma*StdDev are zeroed. Only used when Wall Subtraction is OFF. Default: 3.0")
                else:
                    pp_c2.info("Noise Sigma ignored in Dynamic mode.")
                
            # Define Local Run Directory (Redirect writes to Cache)
            run_dir = os.path.join(cache_root, proc_row['Date'], "preprocessed_data", proc_row['Sequence'])
            os.makedirs(run_dir, exist_ok=True)
            status_file = os.path.join(run_dir, 'processing_status.json')
            
            is_running = False
            status_data = {}
            
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f: status_data = json.load(f)
                    # Check if stale
                    # 'initializing' should be fast (<30s). 'processing' can take longer between updates (10m).
                    ts = status_data.get('timestamp', 0)
                    state = status_data.get('state')
                    timeout = 30 if state == 'initializing' else 600
                    
                    if time.time() - ts < timeout:
                        if state not in ['completed', 'failed']:
                            is_running = True
                except: pass

            if is_running:
                st.info(f"Background Task Active: {status_data.get('state')}")
                st.progress(min(1.0, max(0.0, status_data.get('progress', 0.0))))
                st.caption(f"Status: {status_data.get('message')}")
                
                if st.button("Stop Monitoring / Reset"):
                    # This doesn't kill the process, just ignores the file
                    try: os.remove(status_file)
                    except: pass
                    st.rerun()
                
                if st.button("Refresh Status"):
                    st.rerun()
                
            else:
                # Show Result of last run if exists
                if status_data.get('state') == 'completed':
                    if st.button("Processing Complete! (Click to Dismiss)", type="primary"):
                         try: os.remove(status_file)
                         except: pass
                         st.cache_data.clear()
                         st.rerun()
                elif status_data.get('state') == 'failed':
                    st.error(f"Last Run Failed: {status_data.get('error')}")
                    if st.button("Clear Error"):
                        try: os.remove(status_file)
                        except: pass
                        st.rerun()

                if st.button("Start Background Processing", type="primary", disabled=bool(proc_row.get('IsLocal', False))):
                    if proc_row.get('IsLocal'):
                         st.error("Cannot process: Raw data only available on remote server.")
                    elif not bg_seq:
                        st.error("Please select a background sequence.")
                    else:
                        import subprocess
                        import sys
                        
                        # Prepare Config
                        meas_path = proc_row['Path']
                        bg_path = os.path.join(date_path, bg_parent, bg_seq)
                        bg_info = f"{bg_parent}/{bg_seq}"
                        # Analysis Type Logic
                        a_type = "Mueller Matrix"
                        if p_ana_type != "Auto-Detect":
                            a_type = p_ana_type
                        else:
                             try:
                                # Robust Auto-Check based on filenames
                                all_files = [f for f in os.listdir(meas_path) if os.path.isfile(os.path.join(meas_path, f))]
                                
                                import re
                                has_depol_parallel = False
                                has_depol_cross = False
                                has_45 = False
                                
                                pattern = re.compile(r"EP(\d+)_EW(\d+)_RP(\d+)_RW(\d+)", re.IGNORECASE)
                                
                                for f in all_files:
                                    m = pattern.search(f)
                                    if m:
                                        ep, ew, rp, rw = [int(x) % 180 for x in m.groups()]
                                        
                                        # Depol Input: 0 (modulo 180)
                                        if ep == 0 and ew == 0:
                                            if rp == 0 and rw == 0: has_depol_parallel = True
                                            if rp == 90 and rw == 90: has_depol_cross = True
                                        
                                        # Mueller Input: 45 or 135
                                        if ep == 45 or ep == 135: has_45 = True
                                
                                if has_depol_parallel and has_depol_cross and not has_45:
                                    a_type = "Depolarization Ratio"
                                else:
                                    a_type = "Mueller Matrix"
                             except: pass

                        config = {
                            "meas_path": meas_path, "bg_path": bg_path, "cache_root": cache_root,
                            "analysis_type": a_type, "bg_on": True, "n_sigma": p_sigma,
                            "l_thresh": p_thresh, "subtract_wall": p_wall, "bg_info": bg_info,
                            "status_file": status_file
                        }
                        
                        config_path = os.path.join(run_dir, 'process_config.json')
                        with open(config_path, 'w') as f: json.dump(config, f)
                        
                        # Initialize Status File
                        initial_status = {
                             "state": "initializing", 
                             "progress": 0.0, 
                             "message": "Starting worker subprocess...", 
                             "timestamp": time.time()
                        }
                        with open(status_file, 'w') as f: json.dump(initial_status, f)
                        
                        # Spawn Worker
                        worker_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'process_worker.py')
                        if sys.platform == "darwin":
                            subprocess.Popen(["caffeinate", "-i", sys.executable, worker_script, config_path])
                        else:
                            subprocess.Popen([sys.executable, worker_script, config_path])
                        
                        st.toast("Background Job Started!")
                        time.sleep(1)
                        st.rerun()

    # --- LOADER PANEL ---
    with col_left:
        st.subheader("Analysis Loader")
        st.write("Select a **Processed** dataset to analyze.")
        
        # Only show processed ones in this list for specific loading?
        # User said "Analysis is just a matter of selecting a pre-processed dataset"
        processed_opts = filtered_df[
            (filtered_df["Standard"] == "Processed") | (filtered_df["Dynamic"] == "Processed")
        ].apply(lambda x: f"{x['Date']} | {x['Description']} ({x['Sequence']})", axis=1).tolist()
        
        sel_load = st.selectbox("Select Dataset to Load", processed_opts, key="load_sel")
        
        if sel_load:
            # Decode selection
            parts = sel_load.split(" | ", 1)
            date_val = parts[0]
            # Format is Description (Sequence)
            rest = parts[1]
            seq_val = rest.rsplit(" (", 1)[-1][:-1]
            load_row = filtered_df[(filtered_df["Date"] == date_val) & (filtered_df["Sequence"] == seq_val)].iloc[0]
            
            st.info(f"Background: {load_row['Background Used']}")

            if st.button("Load into Interactive Analysis", type="primary"):
                 # Determine paths logic
                date_path = os.path.dirname(os.path.dirname(load_row['Path']))
                meas_path = load_row['Path']
                
                # Try to resolve bg_parent from stored info if possible, else guess
                bg_parent_guess = None
                bg_used = load_row['Background Used']
                if "/" in str(bg_used):
                    bg_parent_guess = bg_used.split("/")[0]

                st.session_state.target_load = {
                    "date": load_row['Date'],
                    "meas_parent": load_row['Parent'],
                    "meas_seq": load_row['Sequence'],
                    "meas_path": meas_path,
                    "bg_parent": bg_parent_guess,
                    "description": load_row['Description']
                }
                
                st.rerun()

