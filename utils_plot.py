import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# pio.renderers.default = "browser"


def plot_design(design_3D: np.ndarray, fig_title: str):
    """
    Create a 3D plot based on the given design.
    """
    num_contours = 10

    # Calculate ranges and sizes for contours
    range_x = design_3D[0, :, :].max() - design_3D[0, :, :].min()
    size_x = range_x / (2 * num_contours)
    range_y = design_3D[1, :, :].max() - design_3D[1, :, :].min()
    size_y = range_y / num_contours
    range_z = design_3D[2, :, :].max() - design_3D[2, :, :].min()
    size_z = range_z / num_contours

    gray_scale = [[0, '#808080'], [1, '#808080']]

    # Lighting and position settings
    lighting_settings = dict(ambient=0.4, diffuse=0.6, fresnel=2, specular=0.3, roughness=0.5)
    light_position = dict(x=100, y=100, z=1000)

    # Create surface plots
    hull_1 = go.Surface(
        x=design_3D[0, :, :],
        y=design_3D[1, :, :],
        z=design_3D[2, :, :],
        colorscale=gray_scale,
        contours={
            "x": {"show": True, "start": design_3D[0, :, :].min(), "end": design_3D[0, :, :].max(), "size": size_x},
            "y": {"show": True, "start": design_3D[1, :, :].min(), "end": design_3D[1, :, :].max(), "size": size_y},
            "z": {"show": True, "start": design_3D[2, :, :].min(), "end": design_3D[2, :, :].max(), "size": size_z}
        },
        showscale=False,
        lighting=lighting_settings,
        lightposition=light_position
    )

    hull_2 = go.Surface(
        x=design_3D[0, :, :],
        y=-design_3D[1, :, :],
        z=design_3D[2, :, :],
        colorscale=gray_scale,
        contours={
            "x": {"show": True, "start": design_3D[0, :, :].min(), "end": design_3D[0, :, :].max(), "size": size_x},
            "y": {"show": True, "start": -design_3D[1, :, :].max(), "end": -design_3D[1, :, :].min(), "size": size_y},
            "z": {"show": True, "start": design_3D[2, :, :].min(), "end": design_3D[2, :, :].max(), "size": size_z}
        },
        showscale=False,
        lighting=lighting_settings,
        lightposition=light_position
    )

    data = [hull_1, hull_2]  # Add the scatter plot data to the list of traces

    # Layout settings
    layout = go.Layout(
        title=fig_title,
        title_font=dict(size=24, family="Arial, bold"),
        title_x=0.5,
        title_y=0.95,
        scene=dict(
            aspectmode='data',
            xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',
                       backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',
                       backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0)"),
            zaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title='',
                       backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(0,0,0,0)"),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0.5, y=0, z=0), eye=dict(x=2, y=1.8, z=1.8))
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

    return fig


def plot_design_grid(designs: np.ndarray, titles: list = None, rows: int = 2, cols: int = 2):
    """
    Create a grid of 3D plots based on the given designs.
    
    Args:
        designs: numpy array of shape (N, 3, H, W) where N is number of designs
        titles: list of titles for each subplot
        rows: number of rows in the grid
        cols: number of columns in the grid
    """
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'surface'}]*cols]*rows,
        subplot_titles=titles if titles else [f'Design {i+1}' for i in range(len(designs))]
    )

    num_contours = 10
    gray_scale = [[0, '#808080'], [1, '#808080']]
    lighting_settings = dict(ambient=0.4, diffuse=0.6, fresnel=2, specular=0.3, roughness=0.5)
    light_position = dict(x=100, y=100, z=1000)

    for idx, design_3D in enumerate(designs):
        # Calculate row and column position
        row = idx // cols + 1
        col = idx % cols + 1

        # Calculate ranges and sizes for contours
        range_x = design_3D[0, :, :].max() - design_3D[0, :, :].min()
        size_x = range_x / (2 * num_contours)
        range_y = design_3D[1, :, :].max() - design_3D[1, :, :].min()
        size_y = range_y / num_contours
        range_z = design_3D[2, :, :].max() - design_3D[2, :, :].min()
        size_z = range_z / num_contours

        # Create both hull surfaces
        for mirror_y in [1, -1]:
            surface = go.Surface(
                x=design_3D[0, :, :],
                y=mirror_y * design_3D[1, :, :],
                z=design_3D[2, :, :],
                colorscale=gray_scale,
                contours={
                    "x": {"show": True, "start": design_3D[0, :, :].min(), "end": design_3D[0, :, :].max(), "size": size_x},
                    "y": {"show": True, "start": mirror_y * design_3D[1, :, :].min(), "end": mirror_y * design_3D[1, :, :].max(), "size": size_y},
                    "z": {"show": True, "start": design_3D[2, :, :].min(), "end": design_3D[2, :, :].max(), "size": size_z}
                },
                showscale=False,
                lighting=lighting_settings,
                lightposition=light_position
            )
            fig.add_trace(surface, row=row, col=col)

    # Update layout for each subplot
    for i in range(1, rows*cols + 1):
        fig.update_scenes(
            aspectmode='data',
            xaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False, title=''),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0.5, y=0, z=0), eye=dict(x=2, y=1.8, z=1.8)),
            row=((i-1)//cols + 1), col=((i-1)%cols + 1)
        )

    fig.update_layout(
        height=400*rows,
        width=500*cols,
        title_font=dict(size=24, family="Arial, bold"),
        showlegend=False
    )

    return fig



def plot_design_grid_mpl(designs: np.ndarray, titles: list = None, rows: int = 2, cols: int = 2, save_path: str = None):
    """
    Create a grid of 3D plots based on the given designs using Matplotlib.
    
    Args:
        designs: numpy array of shape (N, 3, H, W) where N is number of designs
        titles: list of titles for each subplot
        rows: number of rows in the grid
        cols: number of columns in the grid
        save_path: optional path to save the figure
    
    Returns:
        matplotlib figure object
    """
    # Determine the number of designs to plot
    num_designs = min(len(designs), rows * cols)
    
    # Create figure with specified layout
    fig = plt.figure(figsize=(6*cols, 5*rows), dpi=300)
    
    # Default titles if not provided
    if titles is None:
        titles = [f'Design {i+1}' for i in range(num_designs)]
    
    # Lighting and rendering parameters
    lighting_params = {
        'ambient': 0.4,
        'diffuse': 0.6,
        'specular': 0.3
    }
    
    # Camera view settings
    view_params = {
        'elev': 30,  # Elevation angle
        'azim': 45   # Azimuth angle
    }
    
    # Plot each design
    for idx in range(num_designs):
        # Create subplot
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        
        # Get current design
        design_3D = designs[idx]
        
        # Separate x, y, z coordinates
        x = design_3D[0, :, :]
        y = design_3D[1, :, :]
        z = design_3D[2, :, :]
        
        # Plot both positive and mirrored surfaces
        surfaces = [
            (y, 1),    # Positive y
            (y, -1)    # Mirrored y
        ]
        
        for surface_y, mirror in surfaces:
            # Create surface plot with gray colormap
            surf = ax.plot_surface(
                x, 
                mirror * surface_y, 
                z, 
                color='gray', 
                alpha=0.7,  # Slight transparency
                edgecolor='none',
                shade=True
            )
        
        # Customize the subplot
        ax.set_title(titles[idx], fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        # Set view angle
        ax.view_init(**view_params)
        
        # Tight axis limits
        ax.set_box_aspect((1,1,1))  # Equal aspect ratio

    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def main():
    design = np.loadtxt(r"D:\repositories\HullGAN_research\encoded_test\kcs_hull_SVA_encoding.txt", delimiter=',')
    design_3d = design.T.reshape(3, 40, 20)
    plot_design(design_3d, 'design')


if __name__ == '__main__':
    main()
