import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Mock Streamlit for imports
import types
sys.modules['streamlit'] = types.ModuleType('streamlit')
import streamlit as st
st.session_state = {}

# Robust Mocking for Decorators and Objects
def mock_cache(*args, **kwargs):
    def decorator(f):
        return f
    return decorator

st.cache_data = mock_cache
st.cache_resource = mock_cache
st.sidebar = types.SimpleNamespace()
st.sidebar.title = lambda *a: None
st.write = lambda *a: None
st.warning = lambda *a: print(f"WARNING: {a}")
st.error = lambda *a: print(f"ERROR: {a}")
st.info = lambda *a: print(f"INFO: {a}")

# Import Utils
from utils import figure_composer, app_utils, polarimeter_processing

# --- CONFIGURATION ---
TEST_CASE = "MATRIX" # "DEPOL" or "MATRIX"

PATHS = {
    "DEPOL": "/Users/andrew/Documents/Uni_Wuppertal/RESEARCH/HACP/analysis/output/IPT_WIND_TUNNEL/20251209/preprocessed_data/test_01_steps7to8_exp0_25s_gain0dB",
    "MATRIX": "/Users/andrew/Documents/Uni_Wuppertal/RESEARCH/HACP/analysis/output/IPT_WIND_TUNNEL/20251208/preprocessed_data/test_01_steps1to6_exp0_8s_gain0dB"
}

INPUT_DATA_PATH = PATHS[TEST_CASE]

# Directory to save the output figure
OUTPUT_DIR = "/Users/andrew/Documents/Uni_Wuppertal/RESEARCH/HACP/analysis/output/IPT_WIND_TUNNEL/test_figure_optimizer"
OUTPUT_FILENAME = f"optimized_figure_{TEST_CASE.lower()}.png"

# Standard hardware offset for angle model
PIXEL_OFFSET_X = 520

# OPTIONAL: Point this to a local raw image path if the remote drive is not connected.
# If this is None, the script will attempt to find a 'Parallel' image in the INPUT_DATA_PATH.
LOCAL_RAW_IMAGE_OVERRIDE = "/Users/andrew/Documents/Uni_Wuppertal/RESEARCH/HACP/analysis/output/IPT_WIND_TUNNEL/test_figure_optimizer/test_raw_img/20251209093319_Mono12_Iteration50_Step7_EP0_EW0_RP0_RW0.PNG"

# --- PALETTE OPTIONS ---
PALETTES = {
    "Tableau": {
        "primary": "#1f77b4", # Blue
        "comps": ["#ff7f0e", "#2ca02c", "#d62728"] # Orange, Green, Red
    },
    "Professional": {
        "primary": "#1565c0", # Rich Blue
        "comps": ["#2e7d32", "#c62828", "#ff8f00"] # Emerald, Ruby, Amber
    },
    "Vibrant": {
        "primary": "#3f51b5", # Indigo
        "comps": ["#009688", "#e91e63", "#9c27b0"] # Teal, Pink, Purple
    },
    "HighContrast": {
        "primary": "#000000", # Black (Very academic/classic)
        "comps": ["#d55e00", "#009e73", "#cc79a7"] # Vermillion, Bluish Green, Purple
    }
}
ACTIVE_PALETTE = PALETTES["Professional"]

# Mock Session State Dependencies
check_dependencies = {
    'measurements': {},
    'backgrounds': {},
    'comp_measurements': {},
    'comp_backgrounds': {},
    'precomputed_data': {},
    'comp_precomputed_data': {},
    'active_dataset_type': "Mueller Matrix", # Default, update on load
    'iterations': [],
    'global_comp_iters': []
}

def discover_symmetric_runs(primary_path):
    """Finds ES, RS, and RS-ES counterparts in the same directory."""
    parent = os.path.dirname(primary_path)
    base_name = os.path.basename(primary_path)
    
    # Identify the 'stem' and 'suffix'
    # Pattern: [STEM]_steps... or [STEM]-MOD_steps...
    # We find the first occurrence of '_steps' to split the base name
    if "_steps" in base_name:
        parts = base_name.split("_steps", 1)
        stem = parts[0]
        suffix = "_steps" + parts[1]
    else:
        # Fallback
        stem = base_name
        suffix = ""
        
    entries = os.listdir(parent)
    found = {}
    
    print(f"DEBUG Discovery: Base='{base_name}', SplitStem='{stem}', Suffix='{suffix}'")
    
    for entry in entries:
        if entry == base_name: continue
        # Look for the stem at the start and the EXACT suffix at the end
        if entry.startswith(stem) and entry.endswith(suffix):
            if '-RS-ES' in entry: found['RS-ES'] = os.path.join(parent, entry)
            elif '-ES' in entry: found['ES'] = os.path.join(parent, entry)
            elif '-RS' in entry: found['RS'] = os.path.join(parent, entry)
    
    print(f"DEBUG Discovery: Found {list(found.keys())}")
    return found

def load_data(test_case_mode="MATRIX"):
    """Loads data from the configured DATA_PATH_BASE and its symmetric counterparts."""
    data_path = DATA_PATH_BASE
    if not os.path.exists(data_path):
        print(f"Error: Input path not found: {data_path}")
        return False

    print(f"Loading Primary Data ({test_case_mode}) from: {data_path}...")
    data, meta, fmt = app_utils.try_load_precomputed(data_path)
    
    if data:
        check_dependencies['precomputed_data'] = data
        check_dependencies['iterations'] = sorted([int(k) if str(k).isdigit() else k for k in data.keys()])
        
        # Explicitly use argument
        if test_case_mode == "DEPOL":
             check_dependencies['active_dataset_type'] = "Depolarization Ratio"
        else:
             check_dependencies['active_dataset_type'] = "Mueller Matrix"
                   
        print(f"Loaded {len(data)} iterations. Mode: {check_dependencies['active_dataset_type']}")
        
        # Discover Symmetric Counterparts
        sym_paths = discover_symmetric_runs(data_path)
        check_dependencies['multi_comparisons'] = []
        
        # Dual Cache Loading for Primary
        print(f"Loading Primary Data ({TEST_CASE}) - Standard (Raw)...")
        std_data, std_meta, _ = app_utils.try_load_precomputed(data_path, mode="standard")
        print(f"Loading Primary Data ({TEST_CASE}) - Dynamic (Subtracted)...")
        dyn_data, dyn_meta, _ = app_utils.try_load_precomputed(data_path, mode="dynamic")
        
        check_dependencies['cache_primary'] = {
            'standard': (std_data, std_meta),
            'dynamic': (dyn_data, dyn_meta)
        }
        
        # Default to dynamic for main stages
        check_dependencies['precomputed_data'], check_dependencies['precomputed_meta'] = (dyn_data, dyn_meta) if dyn_data else (std_data, std_meta)
        
        comp_colors = ACTIVE_PALETTE["comps"]
        
        for i, (label, s_path) in enumerate(sym_paths.items()):
            print(f"Loading Symmetric Comparison ({label}) - Standard (Raw)...")
            cs_data, cs_meta, _ = app_utils.try_load_precomputed(s_path, mode="standard")
            print(f"Loading Symmetric Comparison ({label}) - Dynamic (Subtracted)...")
            cd_data, cd_meta, _ = app_utils.try_load_precomputed(s_path, mode="dynamic")
            
            # Use dynamic for comparison if available
            c_data, c_meta = (cd_data, cd_meta) if cd_data else (cs_data, cs_meta)
            
            if c_data:
                check_dependencies['multi_comparisons'].append({
                    "measurements": None,
                    "precomputed_data": c_data,
                    "precomputed_meta": c_meta, # Crucial for composer internal validation
                    "iterations": sorted([int(k) if str(k).isdigit() else k for k in c_data.keys()]),
                    "label": label,
                    "color": comp_colors[i % len(comp_colors)]
                })
        
        return True
    
    print("Failed to load valid precomputed data.")
    return False

def discover_first_raw_image(data_path):
    """Discovers the first 'Parallel' raw image in the sequence folder."""
    # Check override first
    if LOCAL_RAW_IMAGE_OVERRIDE and os.path.exists(LOCAL_RAW_IMAGE_OVERRIDE):
        print(f"Using local raw image override: {LOCAL_RAW_IMAGE_OVERRIDE}")
        return LOCAL_RAW_IMAGE_OVERRIDE

    # The folder contains processed_data.nc, etc. We need siblings.
    seq_folder = data_path
    if not os.path.exists(seq_folder): return None
    
    # Common parallel keywords from app_utils.robust_find_and_organize_files
    parallel_keywords = ['Depol_Parallel', 'Parallel']
    
    for f in sorted(os.listdir(seq_folder)):
        if f.lower().endswith(('.png', '.tif', '.tiff', '.bmp')):
            # Heuristic match for parallel
            if any(k in f for k in parallel_keywords):
                return os.path.join(seq_folder, f)
    return None

def build_specs_depol():
    layout_type = "Single Panel"
    width, height = 12, 3.5
    unit = "inch"
    
    angle_range = (100, 167)
    global_xlabel = "Scattering Angle [deg]"
    global_ylabel = "Depolarization Ratio"
    
    iters = check_dependencies['iterations']
    # Use ALL iterations for average/std
    selected_iters = iters if iters else []
    
    specs = {
        (1,1): {
            "type": "Computed Data", 
            "variable": "Depolarization Ratio", 
            "iters": selected_iters, 
            "show_std": True, 
            "show_comp": False, 
            "ylim": None
        }
    }
    return specs, layout_type, width, height, unit, global_xlabel, global_ylabel, angle_range

def build_specs_matrix():
    layout_type = "2x2 Grid"
    width, height = 12, 5.0
    unit = "inch"
    
    angle_range = (100, 167)
    global_xlabel = "Scattering Angle [deg]"
    global_ylabel = "Normalized Element Intensity"
    
    iters = check_dependencies['iterations']
    selected_iters = iters if iters else []
    
    specs = {
        (1,1): {"type": "Computed Data", "variable": "S11", "iters": selected_iters, "show_std": True, "show_comp": False, "ylim": None, "panel_title": "S11", "scale_factor": 1e-4, "ylabel": r"Intensity ($10^4$)"},
        (1,2): {"type": "Computed Data", "variable": "S12/S11", "iters": selected_iters, "show_std": True, "show_comp": False, "ylim": (-1, 1), "panel_title": "S12/S11"},
        (2,1): {"type": "Computed Data", "variable": "S33/S11", "iters": selected_iters, "show_std": True, "show_comp": False, "ylim": (-1, 1), "panel_title": "S33/S11"},
        (2,2): {"type": "Computed Data", "variable": "S34/S11", "iters": selected_iters, "show_std": True, "show_comp": False, "ylim": (-1, 1), "panel_title": "S34/S11"}
    }
    return specs, layout_type, width, height, unit, global_xlabel, global_ylabel, angle_range

def run_optimization_case(case_name):
    print(f"\n{'='*60}")
    print(f"RUNNING OPTIMIZATION CASE: {case_name}")
    print(f"{'='*60}")
    
    global DATA_PATH_BASE, DATA_PATH_STEM, TEST_CASE
    TEST_CASE = case_name
    
    if case_name == "MATRIX":
        # Matrix Data (Steps 1-6) - 20251208
        base_dir = "/Users/andrew/Documents/Uni_Wuppertal/RESEARCH/HACP/analysis/output/IPT_WIND_TUNNEL/20251208/preprocessed_data"
        folder = "test_01_steps1to6_exp0_8s_gain0dB"
        DATA_PATH_BASE = os.path.join(base_dir, folder)
        DATA_PATH_STEM = "optimized_matrix"
    elif case_name == "DEPOL":
        # Depolarization Data (Steps 7-8) - 20251209
        base_dir = "/Users/andrew/Documents/Uni_Wuppertal/RESEARCH/HACP/analysis/output/IPT_WIND_TUNNEL/20251209/preprocessed_data"
        folder = "test_01_steps7to8_exp0_25s_gain0dB"
        DATA_PATH_BASE = os.path.join(base_dir, folder)
        DATA_PATH_STEM = "optimized_depol"
    
    print(f"Loading Primary Data ({case_name}) from: {DATA_PATH_BASE}...")
    
    # Reload data according to case
    if not load_data(case_name): return

    if case_name == "DEPOL":
        specs, layout_type, w, h, unit, gx, gy, ar = build_specs_depol()
    else:
        specs, layout_type, w, h, unit, gx, gy, ar = build_specs_matrix()
    
    # --- VISUAL POLISH: LEGEND & PRECISION ---
    # Only show legend on the first panel (1,1)
    # Set y_precision: S11 -> 0 (Integer), Others -> 2
    for pos, spec in specs.items():
        r, c = pos
        spec["show_legend"] = (r == 1 and c == 1)
        
        if spec.get("variable") == "S11":
            spec["y_precision"] = 0
            # spec["ylim"] = [0, 8500]  <-- REMOVED to allow auto-scaling
        else:
            spec["y_precision"] = 2
            
    check_dependencies['primary_label'] = "Base" # Rename Primary -> Base
        
    angle_model = app_utils.load_angle_model_from_npz(app_utils.CALIBRATION_FILE_PATH)

    # --- STAGE 1: BASE FIGURE (Anchored Scaling) ---
    print(f"Generating BASE Figure ({TEST_CASE}) to anchor scaling...")
    
    # Stage 1 has NO comparisons so it anchors to primary only
    orig_multi = check_dependencies.get('multi_comparisons', [])
    check_dependencies['multi_comparisons'] = []
    
    # ENSURE STAGE 1 USES DYNAMIC (SUBTRACTED) CACHE
    cache_primary = check_dependencies.get('cache_primary', {})
    std_p, std_m = cache_primary.get('standard', (None, None))
    dyn_p, dyn_m = cache_primary.get('dynamic', (None, None))
    
    check_dependencies['precomputed_data'], check_dependencies['precomputed_meta'] = (dyn_p, dyn_m) if dyn_p else (std_p, std_m)
    
    fig_base, report_base = figure_composer.generate_composite_figure(
        specs, layout_type, w, h, unit, 
        check_dependencies=check_dependencies,
        common_labels=True,
        global_xlabel=gx, global_ylabel=gy,
        active_style=figure_composer.plotting.STYLES['Default'],
        colors=[ACTIVE_PALETTE["primary"]],
        angle_range=ar,
        angle_model=angle_model,
        pixel_offset=PIXEL_OFFSET_X,
        show_legend=True,
        disable_sci_angle=True,
        disable_sci_y=True,
        y_precision=3
    )
    
    # --- STAGE 2: SYMMETRIC COMPARISON FIGURE ---
    anchored_ylims = {}
    for i, ax in enumerate(fig_base.get_axes()):
        anchored_ylims[i] = ax.get_ylim()

    # --- STAGE 2: SYMMETRIC COMPARISON FIGURE ---
    print(f"Generating SYMMETRIC Comparison Figure ({TEST_CASE})...")
    check_dependencies['multi_comparisons'] = orig_multi
    
    # Update specs to show comparisons and disable shading for them
    for pos in specs:
        specs[pos]["show_comp"] = True
        specs[pos]["show_std_comp"] = False # SHADING OFF for comparisons
        # Apply anchored limits
        # Matplotlib axes in grid are ordered (1,1), (1,2)... which matches (rows, cols)
        idx = (pos[0]-1) * (2 if "2x" in layout_type else 1) + (pos[1]-1)
        if idx in anchored_ylims:
            specs[pos]["ylim"] = anchored_ylims[idx]

    fig_sym, report_sym = figure_composer.generate_composite_figure(
        specs, layout_type, w, h, unit, 
        check_dependencies=check_dependencies,
        common_labels=True,
        global_xlabel=gx, global_ylabel=gy,
        active_style=figure_composer.plotting.STYLES['Default'],
        colors=["#1f77b4"], # Primary blue
        angle_range=ar,
        angle_model=angle_model,
        pixel_offset=PIXEL_OFFSET_X,
        show_legend=True,
        disable_sci_angle=True,
        disable_sci_y=True,
        subtract_wall=True,
        y_precision=3
    )
    
    # --- STAGE 3: NO-WALL BASELINE (DEPOL ONLY) ---
    fig_no_wall = None
    if TEST_CASE == "DEPOL":
        print(f"Generating NO-WALL Baseline Figure ({TEST_CASE})...")
        # Reuse Stage 1 logic but switch to STANDARD (RAW) CACHE
        check_dependencies['multi_comparisons'] = [] 
        
        # SWITCH TO RAW DATA
        check_dependencies['precomputed_data'], check_dependencies['precomputed_meta'] = (std_p, std_m)
        
        # ENABLE AUTO-SCALING for No-Wall Baseline
        for pos in specs:
             specs[pos]["ylim"] = None
        
        fig_no_wall, _ = figure_composer.generate_composite_figure(
            specs, layout_type, w, h, unit, 
            check_dependencies=check_dependencies,
            common_labels=True,
            global_xlabel=gx, global_ylabel=gy,
            active_style=figure_composer.plotting.STYLES['Default'],
            colors=[ACTIVE_PALETTE["primary"]],
            angle_range=ar,
            angle_model=angle_model,
            pixel_offset=PIXEL_OFFSET_X,
            show_legend=True,
            disable_sci_angle=True,
            disable_sci_y=True,
            subtract_wall=False, # THE INTENT
            y_precision=3
        )

    # --- OUTPUT ORGANIZATION ---
    SUBDIRS = {
        "matrix": os.path.join(OUTPUT_DIR, "optimized_matrix"),
        "depol": os.path.join(OUTPUT_DIR, "optimized_depol"),
        "raw": os.path.join(OUTPUT_DIR, "optimized_raw_img"),
        "fitting": os.path.join(OUTPUT_DIR, "optimized_signal_fitting"),
        "reports": os.path.join(OUTPUT_DIR, "optimized_matrix" if TEST_CASE == "MATRIX" else "optimized_depol", "reports")
    }
    for p in SUBDIRS.values():
        if not os.path.exists(p): os.makedirs(p)
 
    # --- STAGE 4: METHODOLOGY FIGURES (Multi-Iteration Refinement) ---
    raw_images = [
        ("Iteration 1", "/Users/andrew/Documents/Uni_Wuppertal/RESEARCH/HACP/analysis/output/IPT_WIND_TUNNEL/test_figure_optimizer/test_raw_img/20251209092800_Mono12_Iteration1_Step7_EP0_EW0_RP0_RW0.PNG"),
        ("Iteration 50", "/Users/andrew/Documents/Uni_Wuppertal/RESEARCH/HACP/analysis/output/IPT_WIND_TUNNEL/test_figure_optimizer/test_raw_img/20251209093319_Mono12_Iteration50_Step7_EP0_EW0_RP0_RW0.PNG"),
        ("Iteration 100", "/Users/andrew/Documents/Uni_Wuppertal/RESEARCH/HACP/analysis/output/IPT_WIND_TUNNEL/test_figure_optimizer/test_raw_img/20251209093815_Mono12_Iteration100_Step7_EP0_EW0_RP0_RW0.PNG")
    ]
    
    # Calculate column for 140 degrees once if possible
    # We'll use a dummy load to get image dimensions if needed, or assume standard W
    target_angle = 140.0
    
    for label, img_path in raw_images:
        if not os.path.exists(img_path):
            print(f"Skipping methodology for {label} (File not found)")
            continue
            
        print(f"Processing methodology for {label}...")
        
        # Load image to get width for angle calculation
        img = np.array(Image.open(img_path))
        h_px, w_px = img.shape
        all_angles = angle_model(np.arange(w_px) + PIXEL_OFFSET_X)
        
        # Target column near 140 degrees
        col_idx = np.argmin(np.abs(all_angles - target_angle))
        actual_angle = all_angles[col_idx]
        print(f"  Target: {target_angle} deg -> Column: {col_idx} (Actual: {actual_angle:.2f} deg)")
        
        # 4a. RAW HEATMAP (Angle-based, Plasma, Plain)
        specs_raw = {
            (1,1): {
                "type": "Raw Image",
                "file": img_path,
                "cmap": "plasma",
                "zero_mode": "White",
                "zero_threshold": 0.11, # Gradient threshold (11%)
                "panel_title": "", # Empty string to suppress title
                "extent": [all_angles[0], all_angles[-1], h_px, 0],
                "disable_sci_x": True, "disable_sci_y": True
            }
        }
        fig_raw, _ = figure_composer.generate_composite_figure(
            specs_raw, "Single Panel", w, h, unit,
            check_dependencies={},
            active_style=figure_composer.plotting.STYLES['Default'],
            global_xlabel="Scattering Angle [deg]",
            global_ylabel="Y Pixel"
        )
        
        # 4b. FITTING PROFILE (Expanded Series: Raw -> Components -> Total)
        fitting_steps = [
            ("raw", ["raw"], "Raw Data"),
            ("wall_only", ["raw", "wall"], "Wall Fit"),
            ("components", ["raw", "wall", "signal"], "Signal Separation"),
            ("total", ["raw", "wall", "signal", "total"], "Full Fit")
        ]
        
        it_num = label.split()[-1]
        
        # Save Heatmap
        raw_out = os.path.join(SUBDIRS["raw"], f"optimized_figure_methodology_raw_iter{it_num}.png")
        fig_raw.savefig(raw_out, bbox_inches='tight', dpi=300)
        print(f"  Saved Heatmap: {os.path.basename(raw_out)}")

        for step_id, comps, step_title in fitting_steps:
             specs_decomp = {
                (1,1): {
                    "type": "Signal Decomposition",
                    "file": img_path,
                    "col_idx": col_idx,
                    "panel_title": f"Fit: {label} @ {actual_angle:.1f}°",
                    "internal_label_loc": "top right",
                    "show_legend": True,
                    "components": comps
                }
             }
             fig_decomp, _ = figure_composer.generate_composite_figure(
                specs_decomp, "Single Panel", w, h, unit,
                check_dependencies={},
                active_style=figure_composer.plotting.STYLES['Default'],
                global_xlabel="Intensity",
                global_ylabel="Vertical Pixel"
             )
             
             f_out = os.path.join(SUBDIRS["fitting"], f"optimized_figure_methodology_fitting_iter{it_num}_{step_id}.png")
             fig_decomp.savefig(f_out, bbox_inches='tight', dpi=300)
             print(f"    Saved Fit ({step_id}): {os.path.basename(f_out)}")

    # STAGE 5: SAVE STANDARD FIGURES TO NEW FOLDERS
    # STAGE 5: SAVE STANDARD FIGURES TO NEW FOLDERS
    target_dir = SUBDIRS["matrix"] if case_name == "MATRIX" else SUBDIRS["depol"]
    
    base_out = os.path.join(target_dir, f"optimized_figure_{case_name.lower()}_base.png")
    sym_out = os.path.join(target_dir, f"optimized_figure_{case_name.lower()}_symmetric.png")
    
    report_out = os.path.join(SUBDIRS["reports"], f"symmetry_report_{case_name.lower()}.txt")
    
    fig_base.savefig(base_out, bbox_inches='tight', dpi=300)
    fig_sym.savefig(sym_out, bbox_inches='tight', dpi=300)
    
    if fig_no_wall:
        no_wall_out = os.path.join(target_dir, f"optimized_figure_{case_name.lower()}_base_no_wall.png")
        fig_no_wall.savefig(no_wall_out, bbox_inches='tight', dpi=300)
        print(f"Saved NO-WALL figure to: {no_wall_out}")

    print(f"Saved BASE figure to: {base_out}")
    print(f"Saved SYMMETRIC figure to: {sym_out}")
    
    # Save Report to TEXT file
    if report_sym:
        report_text = "========================================\n"
        report_text += "QUANTITATIVE SYMMETRY REPORT\n"
        report_text += "========================================\n"
        for r in report_sym:
            bias_str = f"{r['Bias']:+.4f}" if abs(r['Bias']) < 0.1 else f"{r['Bias']:+.1%}"
            report_text += f"[{r['Panel']}] {r['Comparison']}: Bias={bias_str}, Match={r['Match (%)']:.2f}%\n"
        report_text += "========================================\n"
        
        with open(report_out, "w") as f:
            f.write(report_text)
        print(f"Saved SYMMETRY REPORT to: {report_out}")

    if report_sym:
        from utils import stats
        print("\n" + "="*40)
        print("QUANTITATIVE SYMMETRY REPORT")
        print("="*40)
        for r in report_sym:
            # Match formatting used in App
            bias_str = f"{r['Bias']:+.4f}" if abs(r['Bias']) < 0.1 else f"{r['Bias']:+.1%}"
            print(f"[{r['Panel']}] {r['Comparison']}: Bias={bias_str}, Match={r['Match (%)']:.2f}%")
        print("="*40 + "\n")

def main():
    # Run both cases to ensure all figures are updated/generated correctly
    for case in ["MATRIX", "DEPOL"]:
        run_optimization_case(case)

if __name__ == "__main__":
    main()
