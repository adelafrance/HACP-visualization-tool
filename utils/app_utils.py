"""
utils/app_utils.py
Helper functions for file scanning, configuration, and data I/O.
"""
import os
import re
import json
import numpy as np
from collections import defaultdict

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

STRATEGIES = {"Strategy A": ['I_PV', 'I_PH', 'I_PP', 'I_PM', 'I_RP', 'I_RM'], "Strategy B": ['I_PV', 'I_PH', 'I_PP', 'I_PM', 'I_PL', 'I_PR']}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_config(config_file, path):
    with open(config_file, 'w') as f: json.dump({'last_path': path}, f)

def load_config(config_file):
    try:
        with open(config_file, 'r') as f: return json.load(f).get('last_path', '')
    except (FileNotFoundError, json.JSONDecodeError): return ''

def get_required_measurements(measurements_data, analysis_type):
    if analysis_type == "Depolarization Ratio":
        return ['Depol_Parallel', 'Depol_Cross']
    if not measurements_data:
        return STRATEGIES["Strategy A"]
    first_iter = next(iter(measurements_data.values()))
    keys = set(first_iter.keys())
    if 'I_PL' in keys and 'I_PR' in keys:
        return STRATEGIES["Strategy B"]
    return STRATEGIES["Strategy A"]

def get_receiver_key_for_measurement(meas_type):
    reverse_map = {'I_PV': (0, 0), 'I_PH': (90, 90), 'I_PP': (45, 45), 'I_PM': (135, 135), 'I_PL': (90, 45), 'I_PR': (45, 90), 'I_RP': (45, 45), 'I_RM': (135, 135), 'Depol_Parallel': (0, 0), 'Depol_Cross': (90, 90)}
    return reverse_map.get(meas_type)

def get_folder_states(folder_path):
    states = set()
    if not os.path.exists(folder_path): return states
    try:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.tif', '.tiff', '.bmp'))]
    except OSError: return states
    pattern = re.compile(r"EP(\d+)_EW(\d+)_RP(\d+)_RW(\d+)", re.IGNORECASE)
    for f in files:
        m = pattern.search(f)
        if m: states.add(tuple(int(x) for x in m.groups()))
    return states

def find_best_background_folder(meas_name, meas_path, bg_parent_path):
    if not os.path.exists(bg_parent_path): return None
    bg_folders = sorted([d for d in os.listdir(bg_parent_path) if os.path.isdir(os.path.join(bg_parent_path, d))])
    if meas_name in bg_folders: return meas_name
    meas_states = get_folder_states(meas_path)
    if meas_states:
        best_bg, best_score = None, -1
        for bg_name in bg_folders:
            bg_path = os.path.join(bg_parent_path, bg_name)
            bg_states = get_folder_states(bg_path)
            if not bg_states: continue
            perfect_matches = sum(1 for m in meas_states if any(all((m[i] % 180) == (b[i] % 180) for i in range(4)) for b in bg_states))
            partial_matches = sum(1 for m in meas_states if any((m[2] % 180 == b[2] % 180) and (m[3] % 180 == b[3] % 180) for b in bg_states))
            total = len(meas_states)
            score = 3 if perfect_matches == total else (2 if partial_matches == total else (1 if perfect_matches > 0 else 0))
            if score > best_score: best_score, best_bg = score, bg_name
            elif score == best_score and score > 0 and meas_name in bg_name: best_bg = bg_name
        if best_bg and best_score > 0: return best_bg
    for bg in bg_folders:
        if meas_name in bg: return bg
    return bg_folders[0] if bg_folders else None

def get_angles_from_filename(filename):
    m = re.search(r"EP(\d+)_EW(\d+)_RP(\d+)_RW(\d+)", filename, re.IGNORECASE)
    return tuple(int(x) for x in m.groups()) if m else None

def get_smart_label(name, primary_meas, comp_meas):
    if not primary_meas or not comp_meas: return name
    try:
        p_iter = next(iter(primary_meas.values()))
        c_iter = next(iter(comp_meas.values()))
        tags = []
        p_keys, c_keys = set(p_iter.keys()), set(c_iter.keys())
        p_strat = "B" if ('I_PL' in p_keys and 'I_PR' in p_keys) else "A"
        c_strat = "B" if ('I_PL' in c_keys and 'I_PR' in c_keys) else "A"
        if p_strat != c_strat: tags.append("Matrix Sym")
        p_file, c_file = list(p_iter.values())[0], list(c_iter.values())[0]
        p_angles = get_angles_from_filename(os.path.basename(p_file))
        c_angles = get_angles_from_filename(os.path.basename(c_file))
        if p_angles and c_angles:
            ep1, ew1, rp1, rw1 = p_angles
            ep2, ew2, rp2, rw2 = c_angles
            def check_sym(a1, b1, a2, b2): return (a1 % 180 == a2 % 180) and (b1 % 180 == b2 % 180) and ((a1 != a2) or (b1 != b2))
            if check_sym(ep1, ew1, ep2, ew2): tags.append("Emi. Sym")
            if check_sym(rp1, rw1, rp2, rw2): tags.append("Rec. Sym")
        if tags: return f"{name} ({', '.join(tags)})"
    except (StopIteration, IndexError, AttributeError): pass
    return name

def robust_find_and_organize_files(meas_path, bg_path):
    measurements, backgrounds = defaultdict(dict), defaultdict(list)
    rec_map = {(0, 0): 'V', (90, 90): 'H', (45, 45): 'P', (135, 135): 'M', (90, 45): 'L', (45, 90): 'R'}
    def parse_info(filename):
        m = re.search(r"EP(\d+)_EW(\d+)_RP(\d+)_RW(\d+)", filename, re.IGNORECASE)
        if not m: return None
        ep, ew, rp, rw = [int(x) % 180 for x in m.groups()]
        iter_m = re.search(r"Iteration(\d+)", filename, re.IGNORECASE)
        iteration = int(iter_m.group(1)) if iter_m else 1
        return ep, ew, rp, rw, iteration

    if os.path.exists(meas_path):
        for f in sorted(os.listdir(meas_path)):
            if not f.lower().endswith(('.png', '.tif', '.tiff', '.bmp')): continue
            info = parse_info(f)
            if not info: continue
            ep, ew, rp, rw, iteration = info
            suffix = rec_map.get((rp, rw))
            if (ep, ew) == (45, 45) and suffix: measurements[iteration][f"I_P{suffix}"] = os.path.join(meas_path, f)
            elif (ep, ew) == (45, 90) and suffix: measurements[iteration][f"I_R{suffix}"] = os.path.join(meas_path, f)
            elif (ep, ew) == (0, 0):
                if (rp, rw) == (0, 0): measurements[iteration]["Depol_Parallel"] = os.path.join(meas_path, f)
                if (rp, rw) == (90, 90): measurements[iteration]["Depol_Cross"] = os.path.join(meas_path, f)

    if os.path.exists(bg_path):
        for f in sorted(os.listdir(bg_path)):
            if not f.lower().endswith(('.png', '.tif', '.tiff', '.bmp')): continue
            info = parse_info(f)
            if not info: continue
            _, _, rp, rw, _ = info
            backgrounds[(rp, rw)].append(os.path.join(bg_path, f))
            
    if not measurements: return {}, {}, "No valid files found (even with robust scan)."
    return measurements, backgrounds, None

def try_load_precomputed(meas_path, local_cache_root=None, mode="auto"):
    """
    Attempts to load precomputed data.
    mode: "auto" (try dynamic then standard), "standard" (only standard), "dynamic" (only dynamic)
    """
    seq_name = os.path.basename(meas_path)
    parent_dir = os.path.dirname(meas_path)
    date_dir = os.path.dirname(parent_dir)
    preproc_dir = os.path.join(date_dir, "preprocessed_data")
    try: date_name = os.path.basename(date_dir)
    except: date_name = "Unknown_Date"

    candidates = []
    
    # Define filenames based on mode
    filenames = []
    if mode in ["auto", "dynamic"]:
        filenames.append("processed_data_dynamic")
    if mode in ["auto", "standard"]:
        filenames.append("processed_data")
        
    for fname in filenames:
        # Priority 1: Local Cache
        if local_cache_root:
            local_path = os.path.join(local_cache_root, date_name, "preprocessed_data", seq_name, fname)
            if HAS_XARRAY: candidates.append((f"{local_path}.nc", "NetCDF"))
            candidates.append((f"{local_path}.json", "JSON"))
        
        # Priority 2: Measurement Folder
        if HAS_XARRAY:
            candidates.append((os.path.join(meas_path, f"{fname}.nc"), "NetCDF"))
        candidates.append((os.path.join(meas_path, f"{fname}.json"), "JSON"))

    for path, fmt in candidates:
        if os.path.exists(path):
            try:
                if fmt == "NetCDF":
                    ds = xr.open_dataset(path)
                    meta = ds.attrs
                    data = defaultdict(dict)
                    iters = ds.coords['iteration'].values
                    for key in ds.data_vars:
                        orig_key = str(key).replace('_div_', '/')
                        vals = ds[key].values
                        for i, it in enumerate(iters):
                            data[str(it)][orig_key] = vals[i]
                    return data, meta, "NetCDF"
                elif fmt == "JSON":
                    with open(path, 'r') as f: d = json.load(f)
                    meta = d.get('metadata', {})
                    results = {}
                    for it, res in d.get('results', {}).items():
                        results[it] = {k: np.array(v) for k, v in res.items()}
                    return results, meta, "JSON"
            except Exception as e:
                print(f"{fmt} load failed for {path}: {e}")
                continue
    return None, {}, None

def save_precomputed_data(meas_path, data, metadata, fmt="NetCDF", local_cache_root=None, is_dynamic=False):
    string_keyed_data = {str(k): v for k, v in data.items()}
    seq_name = os.path.basename(meas_path)
    parent_dir = os.path.dirname(meas_path)
    date_dir = os.path.dirname(parent_dir)
    
    if local_cache_root:
        try: date_name = os.path.basename(date_dir)
        except: date_name = "Unknown_Date"
        save_dir = os.path.join(local_cache_root, date_name, "preprocessed_data", seq_name)
    else:
        save_dir = os.path.join(date_dir, "preprocessed_data", seq_name)
    os.makedirs(save_dir, exist_ok=True)
    
    filename_base = "processed_data_dynamic" if is_dynamic else "processed_data"
    
    if fmt == "NetCDF" and HAS_XARRAY:
        save_path = os.path.join(save_dir, f"{filename_base}.nc")
        data_vars = defaultdict(list)
        iters = sorted(string_keyed_data.keys())
        if not iters: return None
        first_iter = string_keyed_data[iters[0]]
        first_val = next(iter(first_iter.values()))
        n_pixels = first_val.shape[0]
        pixels = np.arange(n_pixels)
        
        for it in iters:
            for k, v in string_keyed_data[it].items(): data_vars[k].append(v)
        
        ds_vars = {}
        for k, v_list in data_vars.items():
            if len(v_list) == len(iters):
                ds_vars[k.replace('/', '_div_')] = (["iteration", "angle"], np.stack(v_list))
        
        nc_attrs = {k: (json.dumps(v, cls=NumpyEncoder) if isinstance(v, dict) else v) for k, v in metadata.items()}
        ds = xr.Dataset(data_vars=ds_vars, coords={"iteration": iters, "pixel": pixels}, attrs=nc_attrs)
        try: ds.to_netcdf(save_path)
        except PermissionError: return None
        return save_path
    else:
        save_path = os.path.join(save_dir, f"{filename_base}.json")
        with open(save_path, 'w') as f: json.dump({"metadata": metadata, "results": string_keyed_data}, f, cls=NumpyEncoder, indent=4)
        return save_path

def get_linear_yrange(y_data, x_data, x_range):
    mask = (x_data >= x_range[0]) & (x_data <= x_range[1])
    visible_y = y_data[np.where(mask)]
    if visible_y.size == 0 or np.all(np.isnan(visible_y)): return None
    min_y, max_y = np.nanmin(visible_y), np.nanmax(visible_y)
    if min_y == max_y: return [min_y - 1, max_y + 1]
    padding = (max_y - min_y) * 0.10
    return [min_y - padding, max_y + padding]
