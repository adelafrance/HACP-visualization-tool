import os
import sys
import numpy as np
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
TEST_CASE = "DEPOL" # Options: "DEPOL", "MATRIX"

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

def load_data():
    """Loads data from the configured INPUT_DATA_PATH and its symmetric counterparts."""
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"Error: Input path not found: {INPUT_DATA_PATH}")
        return False

    print(f"Loading Primary Data ({TEST_CASE}) from: {INPUT_DATA_PATH}...")
    data, meta, fmt = app_utils.try_load_precomputed(INPUT_DATA_PATH)
    
    if data:
        check_dependencies['precomputed_data'] = data
        check_dependencies['iterations'] = sorted([int(k) if str(k).isdigit() else k for k in data.keys()])
        
        if TEST_CASE == "DEPOL":
             check_dependencies['active_dataset_type'] = "Depolarization Ratio"
        else:
             check_dependencies['active_dataset_type'] = "Mueller Matrix"
                   
        print(f"Loaded {len(data)} iterations. Mode: {check_dependencies['active_dataset_type']}")
        
        # Discover Symmetric Counterparts
        sym_paths = discover_symmetric_runs(INPUT_DATA_PATH)
        check_dependencies['multi_comparisons'] = []
        
        comp_colors = ACTIVE_PALETTE["comps"]
        
        for i, (label, s_path) in enumerate(sym_paths.items()):
            print(f"Loading Symmetric Comparison ({label}) from: {s_path}...")
            s_data, s_meta, s_fmt = app_utils.try_load_precomputed(s_path)
            if s_data:
                check_dependencies['multi_comparisons'].append({
                    "measurements": None, # Using precomputed
                    "precomputed_data": s_data,
                    "iterations": sorted([int(k) if str(k).isdigit() else k for k in s_data.keys()]),
                    "label": label,
                    "color": comp_colors[i % len(comp_colors)]
                })
        
        return True
    
    print("Failed to load valid precomputed data.")
    return False

def build_specs_depol():
    layout_type = "Single Panel"
    width, height = 8, 5
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
    width, height = 12, 10
    unit = "inch"
    
    angle_range = (100, 167)
    global_xlabel = "Scattering Angle [deg]"
    global_ylabel = "Normalized Intensity / Elements"
    
    iters = check_dependencies['iterations']
    selected_iters = iters if iters else []
    
    specs = {
        (1,1): {"type": "Computed Data", "variable": "S11", "iters": selected_iters, "show_std": True, "show_comp": False, "ylim": None, "panel_title": "S11", "scale_factor": 1e-4, "ylabel": r"Intensity ($10^4$)"},
        (1,2): {"type": "Computed Data", "variable": "S12/S11", "iters": selected_iters, "show_std": True, "show_comp": False, "ylim": (-1, 1), "panel_title": "S12/S11"},
        (2,1): {"type": "Computed Data", "variable": "S33/S11", "iters": selected_iters, "show_std": True, "show_comp": False, "ylim": (-1, 1), "panel_title": "S33/S11"},
        (2,2): {"type": "Computed Data", "variable": "S34/S11", "iters": selected_iters, "show_std": True, "show_comp": False, "ylim": (-1, 1), "panel_title": "S34/S11"}
    }
    return specs, layout_type, width, height, unit, global_xlabel, global_ylabel, angle_range

def main():
    if not load_data(): return

    if TEST_CASE == "DEPOL":
        specs, layout_type, w, h, unit, gx, gy, ar = build_specs_depol()
    else:
        specs, layout_type, w, h, unit, gx, gy, ar = build_specs_matrix()
    
    angle_model = app_utils.load_angle_model_from_npz(app_utils.CALIBRATION_FILE_PATH)

    # --- STAGE 1: BASE FIGURE (Anchored Scaling) ---
    print(f"Generating BASE Figure ({TEST_CASE}) to anchor scaling...")
    
    # Stage 1 has NO comparisons so it anchors to primary only
    orig_multi = check_dependencies.get('multi_comparisons', [])
    check_dependencies['multi_comparisons'] = []
    
    fig_base = figure_composer.generate_composite_figure(
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
        disable_sci_y=True
    )
    
    # Extract Y-Limits from the base figure to anchor the comparison version
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

    fig_sym = figure_composer.generate_composite_figure(
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
        disable_sci_y=True
    )
    
    # OUTPUT HANDLING
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    base_out = os.path.join(OUTPUT_DIR, f"optimized_figure_{TEST_CASE.lower()}_base.png")
    sym_out = os.path.join(OUTPUT_DIR, f"optimized_figure_{TEST_CASE.lower()}_symmetric.png")
    
    fig_base.savefig(base_out, bbox_inches='tight', dpi=300)
    fig_sym.savefig(sym_out, bbox_inches='tight', dpi=300)
    
    print(f"Saved BASE figure to: {base_out}")
    print(f"Saved SYMMETRIC figure to: {sym_out}")

if __name__ == "__main__":
    main()
