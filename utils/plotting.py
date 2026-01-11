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
                 ticks_in=True):
        self.font_family = font_family
        self.font_size = font_size
        self.background_color = background_color
        self.grid_color = grid_color
        self.show_grid = show_grid
        self.ticks_in = ticks_in
        
    def apply_to_mpl(self, ax):
        """Applies style to a Matplotlib Axes object."""
        ax.set_facecolor(self.background_color)
        if self.show_grid:
            ax.grid(True, color=self.grid_color, linestyle='--', linewidth=0.5)
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
            label.set_fontsize(self.font_size - 2)
            label.set_fontname(self.font_family)

STYLES = {
    "Default": FigureStyle(),
    "Journal": FigureStyle(font_family="Times New Roman", font_size=14, background_color="white", grid_color="black"),
    "Publication": FigureStyle(font_family="Arial", font_size=14, background_color="white", grid_color="lightgrey"),
    "Presentation": FigureStyle(font_family="Arial", font_size=18, background_color="white", grid_color="dimgrey")  
}

def create_mpl_heatmap(img, ax, cmap='viridis', zmin=None, zmax=None, title="", style=None, xlabel="", ylabel="", show_x=True, show_y=True):
    """Matplotlib Heatmap on a specific Axes."""
    if style is None: style = STYLES["Default"]
    
    im = ax.imshow(img, cmap=cmap, vmin=zmin, vmax=zmax, aspect='auto')
    
    if title: ax.set_title(title)
    if show_x: ax.set_xlabel(xlabel)
    if show_y: ax.set_ylabel(ylabel)
    
    if not show_x: ax.set_xticklabels([])
    if not show_y: ax.set_yticklabels([])
    
    # Enforce Scientific Notation (only if ScalarFormatter)
    try:
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    except (AttributeError, ValueError):
        pass
    style.apply_to_mpl(ax)
    return im

def create_mpl_line(x_data, y_data, ax, y_err=None, label="Data", color="blue", title="", style=None, xlabel="", ylabel="", show_x=True, show_y=True, y_log=False, disable_sci_x=False, disable_sci_y=False, linestyle="-", ylim=None, internal_label=None, scale_factor=1.0):
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
    
    # Internal Panel Label (Top Left)
    if internal_label:
        ax.text(0.02, 0.96, internal_label, transform=ax.transAxes, 
                verticalalignment='top', fontweight='bold', 
                fontsize=style.font_size, family=style.font_family,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # Handle Axis Visibility vs Tick Visibility
    if not show_x:
        ax.tick_params(axis='x', labelbottom=False)
    
    if not show_y:
        ax.tick_params(axis='y', labelleft=False)

    # Scientific Notation Logic
    try:
        if not disable_sci_x:
            ax.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
        else:
            ax.ticklabel_format(style='plain', axis='x')
        
        if not disable_sci_y:
            ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
        else:
            ax.ticklabel_format(style='plain', axis='y')
    except (AttributeError, ValueError):
        pass # Log scale or incompatible formatter
        
    if ylim is not None:
        ax.set_ylim(ylim)

    style.apply_to_mpl(ax)

def create_mpl_decomposition(y_pixels, norm_profile, ax, popt=None, title="", style=None, xlabel="Intensity", ylabel="Pixel", show_x=True, show_y=True):
    """Matplotlib Decomposition Plot."""
    if style is None: style = STYLES["Default"]
    from utils import polarimeter_processing
    
    # Raw Data - Note: y_pixels is Y-AXIS, norm_profile is X-AXIS (Vertical Profile)
    # Matplotlib plot(x, y). So plot(norm_profile, y_pixels)
    ax.plot(norm_profile, y_pixels, label='Raw Data', color='#1f77b4', linewidth=2)
    
    if popt is not None:
         # Generate curves
        narrow_curve = polarimeter_processing.gaussian_func(y_pixels, popt[1], popt[2], popt[3], 0) 
        broad_curve = polarimeter_processing.gaussian_func(y_pixels, popt[4], popt[5], popt[6], popt[0])
        total_curve = polarimeter_processing.double_gaussian_func(y_pixels, *popt)
        
        ax.plot(broad_curve, y_pixels, label='Wall', color='gray', linestyle='--')
        ax.fill_betweenx(y_pixels, broad_curve, 0, color='gray', alpha=0.2)
        
        ax.plot(narrow_curve, y_pixels, label='Signal', color='magenta')
        ax.fill_betweenx(y_pixels, narrow_curve, 0, color='magenta', alpha=0.2)
        
        ax.plot(total_curve, y_pixels, label='Total Fit', color='black', linestyle=':')
        
    ax.invert_yaxis() # Image Coordinates
    
    if title: ax.set_title(title)
    if show_x: ax.set_xlabel(xlabel)
    if show_y: ax.set_ylabel(ylabel)
    
    if not show_x: ax.set_xticklabels([])
    if not show_y: ax.set_yticklabels([])

    try:
        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    except (AttributeError, ValueError):
        pass
    style.apply_to_mpl(ax)


def get_modified_colorscale(name, zero_mode):
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
