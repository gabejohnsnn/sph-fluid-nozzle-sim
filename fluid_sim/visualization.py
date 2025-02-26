import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from .nozzle import CylindricalNozzle, ConvergingDivergingNozzle

class Visualizer:
    """
    Base class for visualization
    """
    def __init__(self, config):
        self.show_velocity = config.get('show_velocity', True)
        self.show_density = config.get('show_density', True)
        self.particle_scale = config.get('particle_scale', 10.0)
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
    2D cross-section visualization using matplotlib
    """
    def __init__(self, config):
        super().__init__(config)
        self.fig = None
        self.ax = None
        self.particles_plot = None
        self.velocity_quiver = None
        self.title = None
        self.colorbar = None
        
        # Slicing parameters
        self.slice_thickness = config.get('slice_thickness', 0.3)  # Thickness of cross-section slice
        self.slice_axis = config.get('slice_axis', 'z')  # Axis to slice along ('x', 'y', or 'z')
        
        # Display configuration
        self.update_freq = config.get('update_freq', 5)  # Update every N simulation steps
        self.pause_time = config.get('pause_time', 0.01)
        self.nozzle_color = config.get('nozzle_color', '#555555')
        self.background_color = config.get('background_color', '#f8f8f8')
        self.show_annotations = config.get('show_annotations', True)
        self.show_grid = config.get('show_grid', True)
        
        # Color options
        self.use_velocity_color = config.get('use_velocity_color', True)
        self.show_pressure = config.get('show_pressure', False)
    
    def setup(self, simulation, nozzle):
        # Create a figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        
        # Set background
        self.fig.patch.set_facecolor(self.background_color)
        self.ax.set_facecolor(self.background_color)
        
        # Determine how to set up the view based on slice axis
        if self.slice_axis == 'z':  # Default, looking along z axis (xy plane)
            # Set axis limits based on simulation bounds
            self.ax.set_xlim(simulation.bound_min[0], simulation.bound_max[0])
            self.ax.set_ylim(simulation.bound_min[1], simulation.bound_max[1])
            
            # Axis labels
            self.ax.set_xlabel('X (Flow Direction)')
            self.ax.set_ylabel('Y')
            
            # Extract the primary axes for slicing
            self.primary_axes = [0, 1]  # x and y
            self.slice_axis_index = 2   # z
            
        elif self.slice_axis == 'y':  # Looking along y axis (xz plane)
            self.ax.set_xlim(simulation.bound_min[0], simulation.bound_max[0])
            self.ax.set_ylim(simulation.bound_min[2], simulation.bound_max[2])
            self.ax.set_xlabel('X (Flow Direction)')
            self.ax.set_ylabel('Z')
            self.primary_axes = [0, 2]  # x and z
            self.slice_axis_index = 1   # y
            
        elif self.slice_axis == 'x':  # Looking along x axis (yz plane)
            self.ax.set_xlim(simulation.bound_min[1], simulation.bound_max[1])
            self.ax.set_ylim(simulation.bound_min[2], simulation.bound_max[2])
            self.ax.set_xlabel('Y')
            self.ax.set_ylabel('Z')
            self.primary_axes = [1, 2]  # y and z
            self.slice_axis_index = 0   # x
        
        # Add grid if enabled
        if self.show_grid:
            self.ax.grid(True, linestyle='--', alpha=0.3)
        
        # Initialize empty scatter plot for particles
        self.particles_plot = self.ax.scatter(
            [], [], s=self.particle_scale, c=[], cmap=self.cmap, alpha=0.8, zorder=10
        )
        
        # Initialize empty quiver plot for velocities if enabled
        if self.show_velocity:
            self.velocity_quiver = self.ax.quiver(
                [], [], [], [], scale=1.0, scale_units='inches', 
                width=0.002, color='red', alpha=0.7, zorder=15
            )
        
        # Visualization of nozzle boundaries
        self._visualize_nozzle(nozzle)
        
        # Add colorbar
        self.colorbar = self.fig.colorbar(self.particles_plot, ax=self.ax)
        if self.use_velocity_color:
            self.colorbar.set_label('Velocity Magnitude')
        elif self.show_pressure:
            self.colorbar.set_label('Pressure')
        else:
            self.colorbar.set_label('Density')
        
        # Title with simulation info
        self.title = self.ax.set_title(f"Time: 0.000s | Particles: 0")
        
        # Tight layout
        self.fig.tight_layout()
        
        # Interactive mode on
        plt.ion()
        self.fig.show()
    
    def _visualize_nozzle(self, nozzle):
        """
        Add a 2D cross-section visualization of the nozzle
        """
        if not isinstance(nozzle, (CylindricalNozzle, ConvergingDivergingNozzle)):
            return
            
        # Different visualization based on slice axis
        if self.slice_axis == 'z':  # xy plane (default)
            self._visualize_nozzle_xy(nozzle)
        elif self.slice_axis == 'y':  # xz plane
            self._visualize_nozzle_xz(nozzle)
        elif self.slice_axis == 'x':  # yz plane
            self._visualize_nozzle_yz(nozzle)
    
    def _visualize_nozzle_xy(self, nozzle):
        """Draw nozzle in xy plane (default view)"""
        # Number of points to use for nozzle visualization
        n_points = 100
        
        # Generate x points along the nozzle length
        x_points = np.linspace(nozzle.position[0], 
                             nozzle.position[0] + nozzle.length * nozzle.direction[0], 
                             n_points)
        
        # Create top and bottom profiles
        top_points = []
        bottom_points = []
        
        for x in x_points:
            # Calculate position along the nozzle (0 to 1)
            t = (x - nozzle.position[0]) / nozzle.length
            
            # Calculate current radius based on nozzle type
            if isinstance(nozzle, ConvergingDivergingNozzle):
                if t <= nozzle.throat_position:
                    t_section = t / nozzle.throat_position
                    current_diameter = (1-t_section) * nozzle.inlet_diameter + t_section * nozzle.throat_diameter
                else:
                    t_section = (t - nozzle.throat_position) / (1 - nozzle.throat_position)
                    current_diameter = (1-t_section) * nozzle.throat_diameter + t_section * nozzle.outlet_diameter
            else:
                # Cylindrical nozzle
                current_diameter = (1-t) * nozzle.inlet_diameter + t * nozzle.outlet_diameter
            
            radius = current_diameter / 2.0
            
            top_points.append((x, radius))
            bottom_points.append((x, -radius))
        
        # Create polygon vertices (top, then bottom in reverse)
        vertices = top_points + bottom_points[::-1]
        
        # Draw nozzle outline
        nozzle_polygon = Polygon(vertices, closed=True, 
                               facecolor=self.nozzle_color, alpha=0.2,
                               edgecolor='black', linewidth=1.5, zorder=3)
        self.ax.add_patch(nozzle_polygon)
        
        # Add annotations if enabled
        if self.show_annotations:
            # Mark inlet, outlet, and throat (if applicable)
            inlet_x = nozzle.position[0]
            outlet_x = nozzle.position[0] + nozzle.length
            
            inlet_r = nozzle.inlet_diameter / 2
            outlet_r = nozzle.outlet_diameter / 2
            
            # Add inlet label
            self.ax.text(inlet_x, inlet_r + 0.2, "Inlet", 
                     ha='center', va='bottom', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
                     zorder=20)
            
            # Add outlet label
            self.ax.text(outlet_x, outlet_r + 0.2, "Outlet", 
                      ha='center', va='bottom', fontsize=9,
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
                      zorder=20)
            
            # Add throat label for converging-diverging nozzle
            if isinstance(nozzle, ConvergingDivergingNozzle):
                throat_x = nozzle.position[0] + nozzle.throat_position * nozzle.length
                throat_r = nozzle.throat_diameter / 2
                
                self.ax.text(throat_x, throat_r + 0.2, "Throat", 
                         ha='center', va='bottom', fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
                         zorder=20)
                
                # Add a small marker at the throat
                self.ax.axvline(x=throat_x, color='blue', linestyle='--', alpha=0.4, zorder=2)
    
    def _visualize_nozzle_xz(self, nozzle):
        """Draw nozzle in xz plane"""
        # Similar to _visualize_nozzle_xy but for xz plane
        n_points = 100
        x_points = np.linspace(nozzle.position[0], 
                             nozzle.position[0] + nozzle.length * nozzle.direction[0], 
                             n_points)
        
        top_points = []
        bottom_points = []
        
        for x in x_points:
            t = (x - nozzle.position[0]) / nozzle.length
            
            if isinstance(nozzle, ConvergingDivergingNozzle):
                if t <= nozzle.throat_position:
                    t_section = t / nozzle.throat_position
                    current_diameter = (1-t_section) * nozzle.inlet_diameter + t_section * nozzle.throat_diameter
                else:
                    t_section = (t - nozzle.throat_position) / (1 - nozzle.throat_position)
                    current_diameter = (1-t_section) * nozzle.throat_diameter + t_section * nozzle.outlet_diameter
            else:
                current_diameter = (1-t) * nozzle.inlet_diameter + t * nozzle.outlet_diameter
            
            radius = current_diameter / 2.0
            
            top_points.append((x, radius))
            bottom_points.append((x, -radius))
        
        vertices = top_points + bottom_points[::-1]
        nozzle_polygon = Polygon(vertices, closed=True, 
                               facecolor=self.nozzle_color, alpha=0.2,
                               edgecolor='black', linewidth=1.5, zorder=3)
        self.ax.add_patch(nozzle_polygon)
        
        # Add annotations similar to xy view if needed
        if self.show_annotations:
            # Similar annotations code as for xy plane
            pass
    
    def _visualize_nozzle_yz(self, nozzle):
        """Draw nozzle in yz plane (cross-section perpendicular to flow)"""
        # This would show circles at specific x positions
        # For simplicity, just draw the inlet, throat, and outlet
        x_positions = [nozzle.position[0]]
        labels = ["Inlet"]
        diameters = [nozzle.inlet_diameter]
        
        if isinstance(nozzle, ConvergingDivergingNozzle):
            throat_x = nozzle.position[0] + nozzle.throat_position * nozzle.length
            x_positions.append(throat_x)
            labels.append("Throat")
            diameters.append(nozzle.throat_diameter)
        
        outlet_x = nozzle.position[0] + nozzle.length
        x_positions.append(outlet_x)
        labels.append("Outlet")
        diameters.append(nozzle.outlet_diameter)
        
        # Draw circles for each position
        for x, label, diameter in zip(x_positions, labels, diameters):
            radius = diameter / 2.0
            circle = plt.Circle((0, 0), radius, fill=False, edgecolor='black', linestyle='-', alpha=0.7, zorder=3)
            self.ax.add_patch(circle)
            
            if self.show_annotations:
                self.ax.text(0, radius + 0.2, f"{label}: {diameter:.2f}m", 
                         ha='center', va='bottom', fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
                         zorder=20)
    
    def update(self, simulation_state):
        """
        Update the visualization with new simulation state
        """
        positions = simulation_state['positions']
        velocities = simulation_state['velocities']
        densities = simulation_state['densities']
        pressures = simulation_state.get('pressures', None)
        time = simulation_state['time']
        
        # Filter particles to only show those in the slice
        slice_mask = np.abs(positions[:, self.slice_axis_index]) < self.slice_thickness
        slice_positions = positions[slice_mask]
        slice_velocities = velocities[slice_mask]
        
        # Get correct values for coloring
        if self.use_velocity_color:
            color_values = np.sqrt(np.sum(slice_velocities**2, axis=1))
        elif self.show_pressure and pressures is not None:
            color_values = pressures[slice_mask]
        else:
            color_values = densities[slice_mask]
        
        # Extract the primary coordinates
        primary_coords = slice_positions[:, self.primary_axes]
        primary_vels = slice_velocities[:, self.primary_axes]
        
        if len(primary_coords) > 0:
            # Update particle positions
            self.particles_plot.set_offsets(primary_coords)
            
            # Update color based on appropriate values
            if len(color_values) > 0:
                vmin = np.min(color_values)
                vmax = np.max(color_values)
                if vmax > vmin:
                    self.particles_plot.set_array(color_values)
            
            # Update velocity quiver if enabled
            if self.show_velocity and self.velocity_quiver:
                # Downsample for quiver plot to avoid clutter
                n_particles = len(primary_coords)
                if n_particles > 100:
                    step = n_particles // 100
                    self.velocity_quiver.set_offsets(primary_coords[::step])
                    self.velocity_quiver.set_UVC(
                        primary_vels[::step, 0],
                        primary_vels[::step, 1]
                    )
                else:
                    self.velocity_quiver.set_offsets(primary_coords)
                    self.velocity_quiver.set_UVC(
                        primary_vels[:, 0],
                        primary_vels[:, 1]
                    )
        else:
            # No particles in slice - clear plots
            self.particles_plot.set_offsets(np.zeros((0, 2)))
            if self.show_velocity and self.velocity_quiver:
                self.velocity_quiver.set_offsets(np.zeros((0, 2)))
                self.velocity_quiver.set_UVC([], [])
        
        # Update title with simulation info
        total_particles = len(positions)
        slice_particles = len(slice_positions)
        self.title.set_text(f"Time: {time:.3f}s | Total: {total_particles} particles | Visible: {slice_particles} particles")
        
        # Update colorbar label
        if self.use_velocity_color:
            self.colorbar.set_label('Velocity Magnitude (m/s)')
        elif self.show_pressure and pressures is not None:
            self.colorbar.set_label('Pressure')
        else:
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
