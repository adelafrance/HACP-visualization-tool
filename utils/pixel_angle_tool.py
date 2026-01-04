# pixel_angle_tool.py
"""
Pixel-to-Angle Conversion Tool

Purpose:
  This script provides a function to convert the horizontal pixel coordinate from
  a camera image into a physical scattering angle in degrees.

How to Use:
  1. Assumes the 'angle_model_data.npz' file is in the same directory as this script.
  2. The self-test and example use can be run from this script.
  3. The `load_angle_model_from_npz` function can be imported to another script.

Requires:
  - numpy
  - scipy
"""
#%%
import numpy as np
from scipy.interpolate import interp1d
import os

# --- Core Tool ---

class PixelAngleModel:
    """
    A callable object that converts a global sensor pixel coordinate to a
    scattering angle in degrees.
    """
    def __init__(self, model_mm, sensor_width_px, pixel_size_mm, scattering_angle_offset=135.0):
        self.model_mm = model_mm
        self.sensor_center_px = sensor_width_px / 2.0
        self.pixel_size_mm = pixel_size_mm
        self.scattering_angle_offset = scattering_angle_offset

    def __call__(self, pixel_coord):
        """
        Converts one or more global pixel coordinates to scattering angles.

        Args:
            pixel_coord (int or array-like): The global horizontal pixel coordinate(s).

        Returns:
            float or np.ndarray: The corresponding scattering angle(s) in degrees.
        """
        pos_mm = (np.asanyarray(pixel_coord) - self.sensor_center_px) * self.pixel_size_mm
        base_angle = self.model_mm(pos_mm)
        return base_angle + self.scattering_angle_offset

def load_angle_model_from_npz(filepath):
    """
    Loads the calibration data from the .npz file and returns a ready-to-use
    PixelAngleModel object.
    """
    try:
        data = np.load(filepath)
        model_mm = interp1d(data['spline_x'], data['spline_y'], kind='cubic', bounds_error=False, fill_value="extrapolate")
        return PixelAngleModel(
            model_mm=model_mm,
            sensor_width_px=data['sensor_center_px'] * 2,
            pixel_size_mm=data['pixel_size_mm'],
            scattering_angle_offset=data['scattering_angle_offset']
        )
    except Exception as e:
        print(f"FATAL ERROR loading angle model from {filepath}: {e}")
        return None

# --- Example Usage ---

if __name__ == "__main__":
    print("--- Pixel-to-Angle Tool Self-Test ---")

    # --- CRITICAL PARAMETER ---
    # This is the horizontal offset of our image on the full sensor.
    # This MUST be added to local image coordinates before using the model.
    # It remains a question whether this should be hardcoded or flexibile since... 
    # ...it is likely to remain fixed in the instrument config file.
    PIXEL_OFFSET_X = 520

    # --- Step 1: Find and load the calibration file ---
    CALIBRATION_FILENAME = 'angle_model_data.npz'
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    calibration_file_path = os.path.join(SCRIPT_DIR, CALIBRATION_FILENAME)

    if not os.path.exists(calibration_file_path):
        print(f"\nERROR: Calibration file not found at '{calibration_file_path}'")
        print(f"Please make sure '{CALIBRATION_FILENAME}' is in the same folder as this script.")
    else:
        print(f"\nLoading angle model from '{CALIBRATION_FILENAME}'...")
        angle_model = load_angle_model_from_npz(calibration_file_path)

        if angle_model:
            print("Model loaded successfully.")

            # --- Step 2: Define local pixel coordinates from an image ---
            # Based on current, 3184-pixel wide image.
            local_pixel_coords = np.arange(3184)

            # --- Step 3: Convert to GLOBAL coordinates by adding the offset ---
            global_pixel_coords = local_pixel_coords + PIXEL_OFFSET_X

            # --- Step 4: Use the model with the GLOBAL coordinates ---
            scattering_angles = angle_model(global_pixel_coords)

            # --- Step 5: Display results ---
            print("\n--- Example Conversion ---")
            print(f"Using a pixel offset of: {PIXEL_OFFSET_X}")
            print("This offset MUST be added to image's pixel coordinates.")
            
            print("\nResulting scattering angles:")
            
            # --- Show the first 5 pixels ---
            print("  --- First 5 Pixels ---")
            for i in range(5):
                print(f"  Image Pixel {local_pixel_coords[i]:<4} (Sensor Pixel {global_pixel_coords[i]:<4}) -> {scattering_angles[i]:.4f} degrees")
            
            print("  ...")

            # --- Show the middle 5 pixels ---
            print("  --- Middle 5 Pixels ---")
            num_pixels = len(local_pixel_coords)
            mid_point = num_pixels // 2
            for i in range(mid_point - 2, mid_point + 3):
                print(f"  Image Pixel {local_pixel_coords[i]:<4} (Sensor Pixel {global_pixel_coords[i]:<4}) -> {scattering_angles[i]:.4f} degrees")

            print("  ...")

            # --- Show the last 5 pixels ---
            print("  --- Last 5 Pixels ---")
            for i in range(num_pixels - 5, num_pixels):
                print(f"  Image Pixel {local_pixel_coords[i]:<4} (Sensor Pixel {global_pixel_coords[i]:<4}) -> {scattering_angles[i]:.4f} degrees")
# %%
