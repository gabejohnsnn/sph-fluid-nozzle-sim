import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from .nozzle import CylindricalNozzle, ConvergingDivergingNozzle

class CrossSectionVisualizer:
    """
    2D cross-section visualization for fluid flow through nozzles
    """
    def __init__(self, config):
        self.show_velocity = config.get('show_velocity', True)
        self.show_density = config.get('show_density', True)
        self.particle_scale = config.get('particle_scale', 10.0)  # Smaller particles
        self.cmap = cm.get_cmap(config.get('colormap', 'viridis'))
        self.slice_thickness = config.get('slice_thickness', 0.5)  # Thickness of the cross-section slice
        self.velocity_scale = config.get('velocity_scale', 0.1)  # Scale factor for velocity arrows
        self.nozzle_color = config.get('nozzle_color', 'gray')
        self.nozzle_alpha = config.get('nozzle_alpha', 0.3)
        self.background_color = config.get('background_color', '#f0f0f0')
        self.grid = config.get('grid', True)
        self.eps = 1e-6  # Small value to prevent division by zero
        
        # Display configuration
        self.update_freq = config.get('update_freq', 5)  # Update every N simulation steps
        self.pause_time = config.get('pause_time', 0.01)
        self.show_pressure = config.get('show_pressure', False)
        
        # Figure and axes
        self.fig = None
        self.ax = None
        self.particles_plot = None
        self.velocity_quiver = None
        self.title = None
        self.colorbar = None
        self.nozzle_outline = None
    
    def setup(self, simulation, nozzle):
        """
        Set up the visualization
        """
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        # Set background color
        self.fig.patch.set_facecolor(self.background_color)
        self.ax.set_facecolor(self.background_color)
        
        # Set axis limits based on simulation bounds with extra space
        x_padding = (simulation.bound_max[0] - simulation.bound_min[0]) * 0.05
        y_padding = (simulation.bound_max[1] - simulation.bound_min[1]) * 0.05
        
        self.ax.set_xlim(simulation.bound_min[0] - x_padding, simulation.bound_max[0] + x_padding)
        self.ax.set_ylim(simulation.bound_min[1] - y_padding, simulation.bound_max[1] + y_padding)
        
        # Axis labels and title
        self.ax.set_xlabel('X (Flow Direction)')
        self.ax.set_ylabel('Y')
        self.title = self.ax.set_title(f"Time: 0.000s | Particles: 0")
        
        # Initialize empty scatter plot for particles
        self.particles_plot = self.ax.scatter(
            [], [], s=self.particle_scale, c=[], cmap=self.cmap, alpha=0.8, edgecolors='none'
        )
        
        # Initialize empty quiver plot for velocities if enabled
        if self.show_velocity:
            self.velocity_quiver = self.ax.quiver(
                [], [], [], [], scale=1.0, scale_units='inches', width=0.002, 
                color='red', alpha=0.7
            )
        
        # Add colorbar
        if self.show_density:
            self.colorbar = self.fig.colorbar(self.particles_plot, ax=self.ax)
            if self.show_pressure:
                self.colorbar.set_label('Pressure')
            else:
                self.colorbar.set_label('Density')
        
        # Visualization of nozzle boundaries
        self._visualize_nozzle(nozzle)
        
        # Add grid if enabled
        if self.grid:
            self.ax.grid(True, linestyle='--', alpha=0.3)
        
        # Tight layout for better use of space
        self.fig.tight_layout()
        
        # Interactive mode on
        plt.ion()
        self.fig.show()
    
    def _visualize_nozzle(self, nozzle):
        """
        Add a 2D cross-section visualization of the nozzle
        """
        if isinstance(nozzle, (CylindricalNozzle, ConvergingDivergingNozzle)):
            # Number of points to define the nozzle outline
            n_points = 100
            
            # Generate x points along the nozzle length
            x_points = np.linspace(nozzle.position[0], 
                                  nozzle.position[0] + nozzle.length * nozzle.direction[0], 
                                  n_points)
            
            # Generate top and bottom y points
            top_y = []
            bottom_y = []
            
            for x in x_points:
                # Calculate position along the nozzle (0 to 1)
                axial_pos = x - nozzle.position[0]
                t = axial_pos / nozzle.length
                
                # Determine current radius based on nozzle type
                if isinstance(nozzle, ConvergingDivergingNozzle):
                    if t <= nozzle.throat_position:
                        t_section = t / nozzle.throat_position
                        current_diameter = (1 - t_section) * nozzle.inlet_diameter + t_section * nozzle.throat_diameter
                    else:
                        t_section = (t - nozzle.throat_position) / (1 - nozzle.throat_position)
                        current_diameter = (1 - t_section) * nozzle.throat_diameter + t_section * nozzle.outlet_diameter
                else:
                    # Cylindrical nozzle
                    current_diameter = (1 - t) * nozzle.inlet_diameter + t * nozzle.outlet_diameter
                
                current_radius = current_diameter / 2.0
                
                # Add points for top and bottom profiles
                top_y.append(current_radius)
                bottom_y.append(-current_radius)
            
            # Create polygon vertices for the nozzle outline
            vertices = list(zip(x_points, top_y))
            vertices.extend(list(zip(x_points[::-1], bottom_y[::-1])))
            
            # Draw nozzle outline as a filled polygon
            self.nozzle_outline = Polygon(vertices, closed=True, 
                                         facecolor=self.nozzle_color, 
                                         alpha=self.nozzle_alpha,
                                         edgecolor='black', linewidth=1.5)
            self.ax.add_patch(self.nozzle_outline)
            
            # Add markers for important points
            if isinstance(nozzle, ConvergingDivergingNozzle):
                # Mark throat position
                throat_x = nozzle.position[0] + nozzle.throat_position * nozzle.length
                self.ax.axvline(x=throat_x, color='blue', linestyle='--', alpha=0.5)
                self.ax.text(throat_x, 0, 'Throat', 
                          bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                          ha='center', va='center', rotation=90)
            
            # Mark inlet and outlet
            inlet_x = nozzle.position[0]
            outlet_x = nozzle.position[0] + nozzle.length
            
            self.ax.text(inlet_x, nozzle.inlet_diameter/2 + 0.1, 'Inlet',
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                      ha='center', va='bottom')
            
            self.ax.text(outlet_x, nozzle.outlet_diameter/2 + 0.1, 'Outlet',
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                      ha='center', va='bottom')
    
    def update(self, simulation_state):
        """
        Update the visualization with new simulation state
        """
        positions = simulation_state['positions']
        velocities = simulation_state['velocities']
        densities = simulation_state['densities']
        pressures = simulation_state.get('pressures', None)
        time = simulation_state['time']
        
        # Filter particles to show only those in the cross-section slice (near z=0)
        slice_mask = np.abs(positions[:, 2]) < self.slice_thickness
        slice_positions = positions[slice_mask]
        slice_velocities = velocities[slice_mask]
        
        if self.show_density:
            if self.show_pressure and pressures is not None:
                slice_values = pressures[slice_mask]
            else:
                slice_values = densities[slice_mask]
        
        # Update particle positions
        if len(slice_positions) > 0:
            self.particles_plot.set_offsets(slice_positions[:, :2])  # Use only x and y coordinates
            
            # Update color based on density or pressure
            if self.show_density and len(slice_values) > 0:
                min_val = np.min(slice_values)
                max_val = np.max(slice_values)
                if max_val > min_val:
                    norm_values = (slice_values - min_val) / (max_val - min_val)
                    self.particles_plot.set_array(norm_values)
            
            # Update particle size based on the actual number of particles
            # Decrease particle size if there are many particles to avoid overcrowding
            n_particles = len(slice_positions)
            adaptive_scale = max(1.0, min(self.particle_scale, self.particle_scale * (1000 / max(1000, n_particles))))
            self.particles_plot.set_sizes([adaptive_scale] * n_particles)
            
            # Update velocity quiver if enabled
            if self.show_velocity and self.velocity_quiver:
                # Downsample for quiver plot to avoid clutter
                if n_particles > 200:
                    step = n_particles // 200
                    self.velocity_quiver.set_offsets(slice_positions[::step, :2])
                    self.velocity_quiver.set_UVC(
                        slice_velocities[::step, 0] * self.velocity_scale,
                        slice_velocities[::step, 1] * self.velocity_scale
                    )
                else:
                    self.velocity_quiver.set_offsets(slice_positions[:, :2])
                    self.velocity_quiver.set_UVC(
                        slice_velocities[:, 0] * self.velocity_scale,
                        slice_velocities[:, 1] * self.velocity_scale
                    )
        else:
            # No particles in slice
            self.particles_plot.set_offsets(np.zeros((0, 2)))
            if self.show_velocity and self.velocity_quiver:
                self.velocity_quiver.set_offsets(np.zeros((0, 2)))
                self.velocity_quiver.set_UVC([], [])
        
        # Update title with simulation info
        total_particles = len(positions)
        slice_particles = len(slice_positions)
        self.title.set_text(f"Time: {time:.3f}s | Total Particles: {total_particles} | Slice Particles: {slice_particles}")
        
        # Update colorbar label if showing pressure
        if self.colorbar and self.show_pressure and pressures is not None:
            self.colorbar.set_label('Pressure')
        elif self.colorbar:
            self.colorbar.set_label('Density')
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(self.pause_time)
    
    def close(self):
        """
        Close the visualization
        """
        plt.close(self.fig)
