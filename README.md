# HACP Visualization Tool

A specialized Streamlit application for analyzing and visualizing polarimeter data from the Hydrogen Atom Control Project (HACP). This tool processes raw measurement images to compute Stokes parameters, Mueller matrix elements, and signal decomposition metrics.

## Features

*   **Interactive Heatmaps**: View raw measurement images with clickable column selection.
*   **Mueller Matrix Calculation**: Automatically computes S11, S12, S33, S34 and normalized elements based on loaded measurement sets.
*   **Signal Decomposition**:
    *   **Background Subtraction**: Removes background noise using reference images.
    *   **Wall Subtraction**: Uses double-Gaussian fitting to separate signal from wall reflections.
    *   **Dynamic ROI**: Automatically detects beam path and width.
*   **Data Management**: Supports organizing thousands of measurement files into logical "Iterations" and "Steps".

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/HACP-visualization-tool.git
    cd HACP-visualization-tool
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install streamlit numpy pandas plotly pillow scipy
    ```
    *(Note: If you plan to export static items later, `kaleido` may also be required)*

## Usage

1.  **Start the App**:
    ```bash
    streamlit run app_v2.py
    ```

2.  **Load Data**:
    *   Point the tool to your `measurements` and `backgrounds` folders in the sidebar.
    *   Click "Load Data" to index the files.

3.  **Analyze**:
    *   Select an **Iteration** from the sidebar.
    *   Adjust **Log Threshold** to visualize faint signals.
    *   Toggle **"Subtract Wall"** for advanced signal isolation.
    *   Click on heatmaps to inspect specific column profiles.

## Structure

*   `app_v2.py`: Main Streamlit application entry point.
*   `utils/`:
    *   `polarimeter_processing.py`: Core physics/math logic (Gaussian fitting, Mueller calculus).
    *   `app_visualization.py`: Plotly chart generation.
    *   `pixel_angle_tool.py`: Geometry calibration for scattering angles.
