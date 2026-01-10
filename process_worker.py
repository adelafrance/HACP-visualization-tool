
import sys
import os
import json
import time
import argparse
import traceback

# Add script directory to path to allow imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from utils import app_computation, polarimeter_processing, app_utils, pixel_angle_tool

def write_status(status_file, state, progress, message="", error=None):
    """Writes status to a JSON file."""
    data = {
        "state": state,
        "progress": progress,
        "message": message,
        "timestamp": time.time(),
        "error": str(error) if error else None
    }
    try:
        with open(status_file, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Failed to write status: {e}")

def main():
    parser = argparse.ArgumentParser(description="Detached Data Processing Worker")
    parser.add_argument("config_file", help="Path to JSON configuration file")
    args = parser.parse_args()
    
    config_path = args.config_file
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    # Extract Parameters
    meas_path = config['meas_path']
    bg_path = config['bg_path']
    cache_root = config['cache_root']
    analysis_type = config['analysis_type']
    bg_on = config['bg_on']
    n_sigma = config['n_sigma']
    l_thresh = config['l_thresh']
    subtract_wall = config['subtract_wall']
    bg_info = config['bg_info']
    status_file = config['status_file']
    
    # Initialize Status
    write_status(status_file, "initializing", 0.0, "Worker started")
    
    
    try:
        # 0. CHECK FOR RESUME
        precomputed = {}
        
        # Determine load mode
        load_mode = "dynamic" if subtract_wall else "standard"
        existing_data, existing_meta, _ = app_utils.try_load_precomputed(meas_path, cache_root, mode=load_mode)
        
        if existing_data:
            # Validate Parameters
            params = existing_meta.get('parameters', {})
            
            # Check critical parameters
            match = True
            if params.get('subtract_wall') != subtract_wall: match = False
            if params.get('bg_subtraction') != bg_on: match = False
            if not subtract_wall:
                 if params.get('noise_sigma') != n_sigma: match = False
            
            if match:
                 precomputed = existing_data
                 write_status(status_file, "resuming", 0.0, f"Resuming from {len(precomputed)} iterations...")
            else:
                 write_status(status_file, "initializing", 0.0, "Parameters changed, starting fresh...")

        # Load Files (with retry logic for VPN stability)
        write_status(status_file, "loading_files", 0.05, "Loading measurements...")
        
        retry_count = 0
        max_retries = 5
        m, b = None, None
        
        while retry_count < max_retries:
            try:
                m, b, err = polarimeter_processing.find_and_organize_files(meas_path, bg_path)
                if not m: m, b, err = app_utils.robust_find_and_organize_files(meas_path, bg_path)
                if m: break
                else: raise Exception(err or "No files found")
            except OSError as e: # Catch network/disk errors
                retry_count += 1
                write_status(status_file, "retrying", 0.05, f"Network error, retrying ({retry_count}/{max_retries})...")
                time.sleep(5) # Wait before retry
            except Exception as e:
                raise e
        
        if not m:
             raise Exception("Failed to load files after retries.")

        # Load Angle Model
        write_status(status_file, "loading_model", 0.1, "Loading calibration...")
        cal_path = os.path.join(SCRIPT_DIR, 'utils', 'angle_model_data.npz')
        angle_model = pixel_angle_tool.load_angle_model_from_npz(cal_path)
        
        # Define Progress Callback
        def progress_cb(val):
            # Scale progress from 0.1 to 1.0
            total_prog = 0.1 + (val * 0.9)
            write_status(status_file, "processing", total_prog, f"Processing: {int(val*100)}%")

        # Run Process
        # We need to handle retries INSIDE the batch process loop ideally, but 
        # app_computation.run_batch_process does the loop. 
        # If the VPN drops during processing (file read), it will crash the function.
        # Ideally we'd wrap the file read inside app_computation, but that requires deeper refactoring.
        # For now, we will just run it. If it crashes, we fail. 
        # But since we are creating a robust worker, maybe we catch errors?
        
        # Re-implementing a simple retry wrapper around the WHOLE process is crude but better than nothing.
        # A better way would be modifying `calculate_curves_for_iteration` to retry.
        # I'll stick to the "Process level retry" implies entire job retry? No, that's bad.
        # Let's trust the OS/file system to handle transient drops slightly better, 
        # or implement file-read retries in `polarimeter_processing` later.
        # For now, the Worker Script isolation protects the UI at least.
        
        save_path, _ = app_computation.run_batch_process(
            m, b, sorted(m.keys()), precomputed, analysis_type, bg_on, n_sigma, l_thresh, 
            meas_path, angle_model, cache_root, export_fmt="NetCDF", 
            subtract_wall=subtract_wall, bg_info=bg_info, progress_callback=progress_cb
        )
        
        write_status(status_file, "completed", 1.0, "Processing Finished")
        
    except Exception as e:
        write_status(status_file, "failed", 0.0, "Error occurred", error=traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Fallback logging if main crashes before status file setup
        try:
            import sys, os, traceback
            log_dir = os.path.dirname(sys.argv[1]) if len(sys.argv) > 1 else "."
            with open(os.path.join(log_dir, "worker_crash.log"), "w") as f:
                f.write(traceback.format_exc())
            # Also try to update status file if possible
        except: pass
