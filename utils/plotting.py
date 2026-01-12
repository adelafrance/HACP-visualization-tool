import plotly.graph_objects as go
import plotly.colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

class FigureStyle:
    """Defines the visual style for a figure."""
    def __init__(self, 
                 font_family="Arial", 
                 font_size=12, 
                 background_color="white",
                 grid_color="lightgrey",
                 show_grid=True,
                 ticks_in=True,
                 legend_size=None,
                 tick_size=None):
        self.font_family = font_family
        self.font_size = font_size
        self.background_color = background_color
        self.grid_color = grid_color
        self.show_grid = show_grid
        self.ticks_in = ticks_in
        self.legend_size = legend_size if legend_size is not None else max(8, self.font_size - 2)
        self.tick_size = tick_size if tick_size is not None else max(8, self.font_size - 2)
        
    def apply_to_mpl(self, ax):
        """Applies style to a Matplotlib Axes object."""
        ax.set_facecolor(self.background_color)
        if self.show_grid:
            ax.grid(True, which='major', color=self.grid_color, linestyle='--', linewidth=0.5)
            ax.grid(True, which='minor', color=self.grid_color, linestyle=':', linewidth=0.3)
        else:
            ax.grid(False)
        
        # Ticks Style
        if self.ticks_in:
            ax.tick_params(direction='in', top=True, right=True, which='both')
        
        # Font settings for labels (Tick labels handled via rcParams usually, or direct set)
        item = {'fontsize': self.font_size, 'fontname': self.font_family}
        ax.set_xlabel(ax.get_xlabel(), **item)
        ax.set_ylabel(ax.get_ylabel(), **item)
        ax.set_title(ax.get_title(), **item, fontweight='bold')
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(self.tick_size)
            label.set_fontname(self.font_family)

STYLES = {
    "Default": FigureStyle(),
    "Journal": FigureStyle(font_family="Times New Roman", font_size=14, background_color="white", grid_color="black"),
    "Publication": FigureStyle(font_family="Arial", font_size=14, background_color="white", grid_color="lightgrey"),
    "Presentation": FigureStyle(font_family="Arial", font_size=18, background_color="white", grid_color="dimgrey"),
    "Optimizer_Large": FigureStyle(font_family="Arial", font_size=18, background_color="white", grid_color="lightgrey", legend_size=10)
}

def get_mpl_modified_cmap(base_name="viridis", zero_mode="Default", zero_threshold=None):
    """Returns a Matplotlib colormap with 0 (or values below threshold) mapped to a specific color."""
    try:
        base_cmap = plt.get_cmap(base_name).copy()
    except:
        base_cmap = plt.get_cmap("viridis").copy()
    
    if zero_mode in ["White", "Black"]:
        from matplotlib.colors import ListedColormap
        colors = base_cmap(np.linspace(0, 1, 256))
        z_color = [1, 1, 1, 1] if zero_mode == "White" else [0, 0, 0, 1]
        
        if zero_threshold is not None:
            # Map the bottom portion of the LUT to the zero color
            # zero_threshold is assumed to be a fraction [0, 1] of the range
            num_indices = int(zero_threshold * 256)
            for i in range(max(1, num_indices)):
                colors[i] = z_color
        else:
            colors[0] = z_color
        return ListedColormap(colors)
    return base_cmap

def create_mpl_heatmap(img, ax, cmap='viridis', zmin=None, zmax=None, title="", style=None, xlabel="", ylabel="", show_x=True, show_y=True, extent=None, disable_sci_x=False, disable_sci_y=False, vline=None, vline_label=None, y_major_ticks=None, y_minor_ticks=None):
    """Matplotlib Heatmap on a specific Axes."""
    if style is None: style = STYLES["Default"]
    
    # If cmap is a string, we might need to modify it if the user wants standard 'White' zero
    # For now, we assume cmap could be a Colormap object already
    im = ax.imshow(img, cmap=cmap, vmin=zmin, vmax=zmax, aspect='auto', extent=extent)
    
    if vline is not None:
        ax.axvline(x=vline, color='red', linestyle='--', linewidth=1.5, alpha=0.9)
        if vline_label:
            # Place label slightly to the right of the line, near the top
            ax.text(vline + (extent[1]-extent[0])*0.01 if extent else vline + 1, 
                    extent[3] + (extent[2]-extent[3])*0.1 if extent else 10, 
                    vline_label, color='red', fontweight='bold', 
                    fontsize=style.font_size, 
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    if title: ax.set_title(title)
    if show_x: ax.set_xlabel(xlabel)
    if show_y: ax.set_ylabel(ylabel)
    
    # Tick Spacing Control
    if y_major_ticks is not None:
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(y_major_ticks))
    if y_minor_ticks is not None:
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor_ticks))

    style.apply_to_mpl(ax)
    
    # Enforce tick visibility and labeling
    ax.tick_params(axis='x', labelbottom=show_x)
    ax.tick_params(axis='y', labelleft=show_y)

    # Scientific Notation Logic
    try:
        if not disable_sci_x:
            ax.ticklabel_format(style='sci', scilimits=(-3,4), axis='x', useOffset=False)
        else:
            ax.ticklabel_format(style='plain', axis='x', useOffset=False)
        
        if not disable_sci_y:
            ax.ticklabel_format(style='sci', scilimits=(-3,4), axis='y', useOffset=False)
        else:
            ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    except (AttributeError, ValueError):
        pass
    return im

def create_mpl_line(x_data, y_data, ax, y_err=None, label="Data", color="blue", title="", style=None, xlabel="", ylabel="", show_x=True, show_y=True, y_log=False, disable_sci_x=False, disable_sci_y=False, linestyle="-", xlim=None, ylim=None, internal_label=None, internal_label_loc="top left", scale_factor=1.0, is_comparison=False, y_precision=None, show_legend=True, y_major_ticks=None, y_minor_ticks=None):
    """Matplotlib Line Plot on a specific Axes."""
    if style is None: style = STYLES["Default"]
    
    # Apply Scaling
    y_scaled = y_data * scale_factor
    y_err_scaled = y_err * scale_factor if y_err is not None else None

    # Plot Mean Curve
    ax.plot(x_data, y_scaled, label=label, color=color, linewidth=2, linestyle=linestyle)
    
    # Plot Standard Deviation / Error
    if y_err_scaled is not None:
        ax.fill_between(x_data, y_scaled - y_err_scaled, y_scaled + y_err_scaled, color=color, alpha=0.3, label=f"{label} ±Std")
    
    if y_log: ax.set_yscale('log')
    
    if title: ax.set_title(title)
    if show_x: ax.set_xlabel(xlabel)
    if show_y: ax.set_ylabel(ylabel)
    
    # Internal Panel Label
    if internal_label:
        lx, ly = 0.02, 0.96
        ha, va = 'left', 'top'
        if internal_label_loc == "top right":
            lx, ly = 0.98, 0.96
            ha = 'right'
        
        ax.text(lx, ly, internal_label, transform=ax.transAxes, 
                verticalalignment=va, horizontalalignment=ha, fontweight='bold', 
                fontsize=style.font_size, family=style.font_family,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # Tick Spacing Control
    if y_major_ticks is not None:
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(y_major_ticks))
    if y_minor_ticks is not None:
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor_ticks))

    style.apply_to_mpl(ax)

    # Handle Axis Visibility vs Tick Visibility (Applied LAST to prevent style overrides)
    if not is_comparison:
        # Enforce show_x/show_y
        ax.tick_params(axis='x', labelbottom=show_x)
        ax.tick_params(axis='y', labelleft=show_y)

        # Scientific Notation Logic
        try:
            if not disable_sci_x:
                ax.ticklabel_format(style='sci', scilimits=(-3,4), axis='x', useOffset=False)
            else:
                ax.ticklabel_format(style='plain', axis='x', useOffset=False)
            
            if not disable_sci_y:
                ax.ticklabel_format(style='sci', scilimits=(-3,4), axis='y', useOffset=False)
            else:
                ax.ticklabel_format(style='plain', axis='y', useOffset=False)
        except (AttributeError, ValueError):
            pass 
        
        # Enforce Fixed Precision if requested
        if y_precision is not None:
            from matplotlib.ticker import FormatStrFormatter
            ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{int(y_precision)}f'))

        if show_legend and label:
             ax.legend(fontsize=style.legend_size, loc='best')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

def create_mpl_decomposition(y_pixels, norm_profile, ax, popt=None, title="", style=None, xlabel="Intensity", ylabel="Pixel", show_x=True, show_y=True, internal_label=None, internal_label_loc="top left", show_legend=False, components=None, signal_color="magenta", total_color="orange", xlim=None, ylim=None, y_major_ticks=None, y_minor_ticks=None):
    """Matplotlib Decomposition Plot with selective component rendering."""
    if style is None: style = STYLES["Default"]
    from utils import polarimeter_processing
    
    # Default to all if not specified
    if components is None:
        components = ['raw', 'wall', 'signal', 'total']
    
    # Raw Data - Note: y_pixels is Y-AXIS, norm_profile is X-AXIS (Vertical Profile)
    # Matplotlib plot(x, y). So plot(norm_profile, y_pixels)
    if 'raw' in components:
        ax.plot(norm_profile, y_pixels, label='Raw Data', color='#1f77b4', linewidth=2)
    
    if internal_label:
        lx, ly = 0.02, 0.96
        ha, va = 'left', 'top'
        if internal_label_loc == "top right":
            lx, ly = 0.98, 0.96
            ha = 'right'

        ax.text(lx, ly, internal_label, transform=ax.transAxes, 
                verticalalignment=va, horizontalalignment=ha, fontweight='bold', 
                fontsize=style.font_size, family=style.font_family,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
    
    if popt is not None:
         # Generate curves
        narrow_curve = polarimeter_processing.gaussian_func(y_pixels, popt[1], popt[2], popt[3], 0) 
        broad_curve = polarimeter_processing.gaussian_func(y_pixels, popt[4], popt[5], popt[6], popt[0])
        total_curve = polarimeter_processing.double_gaussian_func(y_pixels, *popt)
        
        if 'wall' in components:
            ax.plot(broad_curve, y_pixels, label='Wall', color='gray', linestyle='--')
            ax.fill_betweenx(y_pixels, broad_curve, 0, color='gray', alpha=0.2)
        
        if 'signal' in components:
            ax.plot(narrow_curve, y_pixels, label='Signal', color=signal_color)
            ax.fill_betweenx(y_pixels, narrow_curve, 0, color=signal_color, alpha=0.2)
        
        if 'total' in components:
            ax.plot(total_curve, y_pixels, label='Total Fit', color='black', linestyle=':', linewidth=1.5)
            ax.fill_betweenx(y_pixels, total_curve, 0, color=total_color, alpha=0.2)
        
    ax.invert_yaxis() # Image Coordinates
    
    if title: ax.set_title(title)
    if show_x: ax.set_xlabel(xlabel)
    if show_y: ax.set_ylabel(ylabel)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Tick Spacing Control
    if y_major_ticks is not None:
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(y_major_ticks))
    if y_minor_ticks is not None:
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor_ticks))

    style.apply_to_mpl(ax)
    
    # Enforce tick visibility and labeling
    ax.tick_params(axis='both', which='major', labelsize=style.tick_size)
    ax.tick_params(axis='x', labelbottom=show_x)
    ax.tick_params(axis='y', labelleft=show_y)
    
    if show_legend:
        # User requested fixed 'lower right' for methodology refinement
        ax.legend(fontsize=style.legend_size, loc='lower right', framealpha=0.5)

    return ax


def suggest_ticks(mn, mx, target_count=5):
    """
    Suggests clever major and minor tick intervals based on a data range.
    Targets roughly target_count major intervals.
    """
    if mn is None or mx is None: return None, None
    span = mx - mn
    if span <= 0: return None, None
    
    # Calculate a raw interval
    raw_interval = span / target_count
    
    # Standard "Pretty" intervals
    # Since we are often dealing with small decimals (0.01, 0.05..), we scale to power of 10
    magnitude = 10 ** np.floor(np.log10(raw_interval))
    res = raw_interval / magnitude
    
    if res < 1.5: interval = 1.0 * magnitude
    elif res < 2.5: interval = 2.0 * magnitude
    elif res < 4.0: interval = 2.5 * magnitude # Or 3? User mentioned 3 as an option.
    elif res < 7.5: interval = 5.0 * magnitude
    else: interval = 10.0 * magnitude
    
    # Special overrides for the user's specific request
    # If the range is ~0.15, interval 0.03 (5 steps) or 0.05 (3 steps)
    # The math above would give 0.02 (7.5 steps) or 0.05
    if 0.12 <= span <= 0.18 and interval == 0.02:
        interval = 0.03 # Harmonize with user's specific "clever" example
        
    major = interval
    minor = interval / 2.0 # Standard 2:1 ratio for the minor grid request
    
    return major, minor


def get_mpl_modified_cmap(base_name="viridis", zero_mode="Default", zero_threshold=None):
    """Returns a Matplotlib colormap with 0 mapped to a specific color or gradient."""
    try:
        base_cmap = plt.get_cmap(base_name).copy()
    except:
        base_cmap = plt.get_cmap("viridis").copy()
    
    if zero_mode in ["White", "Black"]:
        from matplotlib.colors import ListedColormap
        # Get the colormapLUT
        colors = base_cmap(np.linspace(0, 1, 256))
        z_color = np.array([1, 1, 1, 1]) if zero_mode == "White" else np.array([0, 0, 0, 1])
        
        if zero_threshold is not None:
            # Gradient Transition Logic to mimic Plotly interpolation
            # We treat zero_threshold as the fraction of the map to blend from z_color to the map's color at that threshold.
            # E.g. 0.11 means indices 0..28 will be a gradient from White -> colors[28]
            
            num_indices = int(zero_threshold * 256)
            if num_indices > 0:
                target_color = colors[num_indices] # The color we blend TO
                
                for i in range(num_indices):
                    t = i / float(num_indices) # 0.0 at i=0, 1.0 at i=num
                    # Linear interpolation: (1-t)*Start + t*End
                    colors[i] = (1 - t) * z_color + t * target_color
        else:
             colors[0] = z_color
             
        return ListedColormap(colors)
    return base_cmap
    """
    Returns a colorscale where 0 is mapped to a specific color (Black/White/Default).
    """
    raw_colors = getattr(plotly.colors.sequential, name.capitalize())
    # Create a list of [pos, color]
    scale = [[i / (len(raw_colors) - 1), color] for i, color in enumerate(raw_colors)]
    
    if zero_mode == "Black":
        scale[0][1] = "#000000"
    elif zero_mode == "White":
        scale[0][1] = "#FFFFFF"
    # Else: Default leaves the original first color
    return scale

def create_heatmap_figure(img_float, cmap_name="Viridis", zero_mode="Default", zmin=0, zmax=None, title="", style=None, xlabel="", ylabel="", show_x_label=True, show_y_label=True):
    """
    Generates a Plotly Figure for a heatmap.
    """
    if style is None: style = STYLES["Default"]
    
    final_cmap = get_modified_colorscale(cmap_name, zero_mode)
    
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=img_float, 
        colorscale=final_cmap, 
        showscale=False, 
        zmin=zmin,
        zmax=zmax,
        hoverinfo='none'
    ))
    
    fig.update_layout(
        title=title,
        font=dict(family=style.font_family, size=style.font_size),
        paper_bgcolor=style.background_color,
        plot_bgcolor=style.background_color,
        margin=dict(l=60 if show_y_label else 20, r=20, t=40 if title else 20, b=50 if show_x_label else 20),
        yaxis=dict(
            autorange='reversed', 
            showticklabels=show_y_label, 
            fixedrange=True,
            title=ylabel if show_y_label else None,
            gridcolor=style.grid_color if style.show_grid else None,
            exponentformat='e'
        ),
        xaxis=dict(
            range=[0, img_float.shape[1]], 
            showticklabels=show_x_label, 
            fixedrange=True,
            title=xlabel if show_x_label else None,
            gridcolor=style.grid_color if style.show_grid else None,
            exponentformat='e'
        )
    )
    
    return fig

def create_line_figure(x_data, y_data, y_err=None, name="Data", title="", style=None, xlabel="", ylabel="", color=None, show_x_label=True, show_y_label=True, y_log=False):
    """Generates a line plot figure with optional shaded error bands."""
    if style is None: style = STYLES["Default"]
    base_color = color if color else "#1f77b4"
    
    fig = go.Figure()
    
    # 1. Error Band (Shaded)
    if y_err is not None:
        # We use a single trace with a 'toself' fill for the shaded region
        # Coordinates: [x_data forward, x_data backward], [y+err forward, y-err backward]
        x_err = np.concatenate([x_data, x_data[::-1]])
        y_upper = y_data + y_err
        y_lower = y_data - y_err
        y_err_combined = np.concatenate([y_upper, y_lower[::-1]])
        
        # Convert to list or handle NaNs for Plotly
        fig.add_trace(go.Scatter(
            x=x_err,
            y=y_err_combined,
            fill='toself',
            fillcolor=f"rgba{tuple(list(mcolors.to_rgb(base_color)) + [0.2])}",
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name=f"{name} ±Std"
        ))

    # 2. Mean Line
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data, 
        mode='lines', 
        name=name,
        line=dict(color=base_color, width=2)
    ))
    
    layout_args = dict(
        title=title,
        font=dict(family=style.font_family, size=style.font_size),
        paper_bgcolor=style.background_color,
        plot_bgcolor=style.background_color,
        margin=dict(l=60 if show_y_label else 20, r=20, t=40 if title else 20, b=50 if show_x_label else 20),
        yaxis=dict(
            showticklabels=show_y_label, 
            title=ylabel if show_y_label else None,
            gridcolor=style.grid_color if style.show_grid else None,
            exponentformat='e'
        ),
        xaxis=dict(
            showticklabels=show_x_label, 
            title=xlabel if show_x_label else None,
            gridcolor=style.grid_color if style.show_grid else None,
            exponentformat='e'
        )
    )
    if y_log: layout_args['yaxis']['type'] = 'log'
    
    fig.update_layout(**layout_args)
    return fig

def create_decomposition_figure(y_pixels, norm_profile, popt=None, roi_bounds=None, title="", style=None, xlabel="Intensity", ylabel="Pixel", show_x_label=True, show_y_label=True):
    """
    Generates a figure for signal decomposition (vertical profile analysis).
    """
    if style is None: style = STYLES["Default"]
    from utils import polarimeter_processing 

    fig = go.Figure()
    
    # 1. Raw Data
    fig.add_trace(go.Scatter(x=norm_profile, y=y_pixels, mode='lines', name='Raw Data', line=dict(color='#1f77b4', width=2)))

    if popt is not None:
        # Generate curves
        narrow_curve = polarimeter_processing.gaussian_func(y_pixels, popt[1], popt[2], popt[3], 0) 
        broad_curve = polarimeter_processing.gaussian_func(y_pixels, popt[4], popt[5], popt[6], popt[0])
        total_curve = polarimeter_processing.double_gaussian_func(y_pixels, *popt)
        
        # Wall (Broad)
        fig.add_trace(go.Scatter(x=broad_curve, y=y_pixels, mode='lines', name='Wall', line=dict(color='gray', dash='dash'), fill='tozerox', fillcolor='rgba(128, 128, 128, 0.2)'))
        # Signal (Narrow)
        fig.add_trace(go.Scatter(x=narrow_curve, y=y_pixels, mode='lines', name='Signal', line=dict(color='magenta', width=2), fill='tozerox', fillcolor='rgba(255, 0, 255, 0.2)'))
        # Total
        fig.add_trace(go.Scatter(x=total_curve, y=y_pixels, mode='lines', name='Total Fit', line=dict(color='black', width=1, dash='dot')))
    
    # ROI Bounds
    if roi_bounds:
        for y, color in [(roi_bounds[0], '#2ca02c'), (roi_bounds[1], '#2ca02c')]:
            if y is not None:
                fig.add_hline(y=y, line_color=color, line_width=1, line_dash="dash")

    fig.update_layout(
        title=title,
        font=dict(family=style.font_family, size=style.font_size),
        paper_bgcolor=style.background_color,
        plot_bgcolor=style.background_color,
        margin=dict(l=60 if show_y_label else 20, r=20, t=40 if title else 20, b=50 if show_x_label else 20),
        yaxis=dict(
            autorange='reversed',
            showticklabels=show_y_label, 
            title=ylabel if show_y_label else None,
            gridcolor=style.grid_color if style.show_grid else None,
            exponentformat='e'
        ),
        xaxis=dict(
            showticklabels=show_x_label, 
            title=xlabel if show_x_label else None,
            gridcolor=style.grid_color if style.show_grid else None,
            exponentformat='e'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def add_roi_traces(fig, roi_paths):
    """Adds ROI lines to an existing figure."""
    if roi_paths and 'roi_top' in roi_paths:
        x_vals = np.arange(len(roi_paths['roi_top']))
        fig.add_trace(go.Scatter(x=x_vals, y=roi_paths['roi_top'], mode='lines', line=dict(color='#00CC96', width=1, dash='dash'), name='ROI Top', hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=x_vals, y=roi_paths['roi_bottom'], mode='lines', line=dict(color='#00CC96', width=1, dash='dash'), name='ROI Bottom', hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=x_vals, y=roi_paths['roi_path'], mode='lines', line=dict(color='red', width=1), name='Center', hoverinfo='skip', showlegend=False))
    return fig

def add_click_capture_trace(fig, h, w):
    """Adds the invisible scatter trace for capturing clicks."""
    fig.add_trace(go.Scatter(
        x=np.arange(w), 
        y=np.full(w, h/2),
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)', size=1),
        hoverinfo='x',
        showlegend=False,
        name='ClickCapture'
    ))
    return fig
