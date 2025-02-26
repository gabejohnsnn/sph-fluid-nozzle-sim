import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from .nozzle import CylindricalNozzle, ConvergingDivergingNozzle

class Visualizer:
    """
    Base class for visualization
    """
    def __init__(self, config):
        self.show_velocity = config.get('show_velocity', False)
        self.show_density = config.get('show_density', True)
        self.particle_scale = config.get('particle_scale', 100.0)
        self.cmap = cm.get_cmap(config.get('colormap', 'viridis'))
        self.eps = 1e-6  # Small value to prevent division by zero
    
    def setup(self, simulation, nozzle):
        """
        Set up the visualization
        """
        raise NotImplementedError("Subclasses must implement setup()")
    
    def update(self, simulation_state):
        """
        Update the visualization with new simulation state
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def close(self):
        """
        Clean up resources
        """
        raise NotImplementedError("Subclasses must implement close()")


class MatplotlibVisualizer(Visualizer):
    """
    Visualization using matplotlib
    """
    def __init__(self, config):
        super().__init__(config)
        self.fig = None
        self.ax = None
        self.particles_plot = None
        self.velocity_quiver = None
        self.title = None
        self.view_angle = 30
        self.animation = None
        self.colorbar = None
        
        # Display configuration
        self.update_freq = config.get('update_freq', 5)  # Update every N simulation steps
        self.pause_time = config.get('pause_time', 0.01)
    
    def setup(self, simulation, nozzle):
        # Create a figure and 3D axis
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set axis limits based on simulation bounds
        self.ax.set_xlim(simulation.bound_min[0], simulation.bound_max[0])
        self.ax.set_ylim(simulation.bound_min[1], simulation.bound_max[1])
        self.ax.set_zlim(simulation.bound_min[2], simulation.bound_max[2])
        
        # Axis labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Initialize empty scatter plot for particles
        self.particles_plot = self.ax.scatter(
            [], [], [], s=self.particle_scale, c=[], cmap=self.cmap, alpha=0.8
        )
        
        # Initialize empty quiver plot for velocities if enabled
        if self.show_velocity:
            self.velocity_quiver = self.ax.quiver(
                [], [], [], [], [], [], color='red', length=0.5, normalize=True
            )
        
        # Add colorbar
        if self.show_density:
            self.colorbar = self.fig.colorbar(self.particles_plot, ax=self.ax)
            self.colorbar.set_label('Density')
        
        # Title with simulation info
        self.title = self.ax.set_title(f"Time: 0.000s | Particles: 0")
        
        # Visualization of nozzle boundaries
        self._visualize_nozzle(nozzle)
        
        plt.ion()  # Interactive mode on
        self.fig.show()
    
    def _visualize_nozzle(self, nozzle):
        """
        Add a wire-frame visualization of the nozzle
        """
        if isinstance(nozzle, (CylindricalNozzle, ConvergingDivergingNozzle)):
            # Number of points to use for nozzle visualization
            n_slices = 20
            n_points_per_slice = 20
            
            # Create slices along the nozzle
            for i in range(n_slices + 1):
                t = i / n_slices
                axial_pos = nozzle.position + t * nozzle.length * nozzle.direction
                
                # Determine current radius
                if hasattr(nozzle, 'throat_position') and hasattr(nozzle, 'throat_diameter'):
                    # Converging-diverging nozzle
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
                
                # Create a circle at this slice
                angles = np.linspace(0, 2 * np.pi, n_points_per_slice, endpoint=False)
                x_circle = []
                y_circle = []
                z_circle = []
                
                for angle in angles:
                    offset = current_radius * (nozzle.right * np.cos(angle) + nozzle.up * np.sin(angle))
                    point = axial_pos + offset
                    x_circle.append(point[0])
                    y_circle.append(point[1])
                    z_circle.append(point[2])
                
                # Close the circle
                x_circle.append(x_circle[0])
                y_circle.append(y_circle[0])
                z_circle.append(z_circle[0])
                
                # Plot this circle
                self.ax.plot(x_circle, y_circle, z_circle, 'k-', alpha=0.3)
            
            # Plot axial lines
            for i in range(n_points_per_slice):
                x_line = []
                y_line = []
                z_line = []
                
                for j in range(n_slices + 1):
                    t = j / n_slices
                    axial_pos = nozzle.position + t * nozzle.length * nozzle.direction
                    
                    # Determine current radius
                    if hasattr(nozzle, 'throat_position') and hasattr(nozzle, 'throat_diameter'):
                        # Converging-diverging nozzle
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
                    angle = 2 * np.pi * i / n_points_per_slice
                    offset = current_radius * (nozzle.right * np.cos(angle) + nozzle.up * np.sin(angle))
                    point = axial_pos + offset
                    
                    x_line.append(point[0])
                    y_line.append(point[1])
                    z_line.append(point[2])
                
                # Plot this line
                self.ax.plot(x_line, y_line, z_line, 'k-', alpha=0.3)
    
    def update(self, simulation_state):
        positions = simulation_state['positions']
        velocities = simulation_state['velocities']
        densities = simulation_state['densities']
        time = simulation_state['time']
        
        # Update particle positions and colors
        self.particles_plot._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        
        # Update color based on density or velocity magnitude
        if self.show_density and len(densities) > 0:
            # Normalize densities to 0-1 range for colormap
            norm_densities = (densities - np.min(densities)) / (np.max(densities) - np.min(densities) + self.eps)
            self.particles_plot.set_array(norm_densities)
        
        # Update velocity quiver if enabled
        if self.show_velocity and self.velocity_quiver and len(positions) > 0:
            # Downsample for quiver plot to avoid clutter
            n_particles = len(positions)
            if n_particles > 100:
                step = n_particles // 100
                self.velocity_quiver.remove()
                self.velocity_quiver = self.ax.quiver(
                    positions[::step, 0], positions[::step, 1], positions[::step, 2],
                    velocities[::step, 0], velocities[::step, 1], velocities[::step, 2],
                    color='red', length=0.5, normalize=True
                )
            else:
                self.velocity_quiver.remove()
                self.velocity_quiver = self.ax.quiver(
                    positions[:, 0], positions[:, 1], positions[:, 2],
                    velocities[:, 0], velocities[:, 1], velocities[:, 2],
                    color='red', length=0.5, normalize=True
                )
        
        # Update title with simulation info
        self.title.set_text(f"Time: {time:.3f}s | Particles: {len(positions)}")
        
        # Rotate view slightly for 3D effect
        self.view_angle = (self.view_angle + 0.2) % 360
        self.ax.view_init(30, self.view_angle)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(self.pause_time)
    
    def close(self):
        plt.close(self.fig)


class ParticleDataExporter:
    """
    Export particle data for analysis
    """
    def __init__(self, config):
        self.export_folder = config.get('export_folder', 'output')
        self.export_interval = config.get('export_interval', 0.1)  # seconds
        self.last_export_time = 0.0
        
        # Create export folder if it doesn't exist
        import os
        if not os.path.exists(self.export_folder):
            os.makedirs(self.export_folder)
    
    def export(self, simulation_state):
        current_time = simulation_state['time']
        
        # Check if it's time to export
        if current_time - self.last_export_time >= self.export_interval:
            self.last_export_time = current_time
            
            # Create a filename with the current time
            filename = f"{self.export_folder}/particles_{current_time:.3f}.npz"
            
            # Save particle data
            np.savez(
                filename,
                positions=simulation_state['positions'],
                velocities=simulation_state['velocities'],
                densities=simulation_state['densities'],
                time=current_time
            )
            
            return True
        
        return False
