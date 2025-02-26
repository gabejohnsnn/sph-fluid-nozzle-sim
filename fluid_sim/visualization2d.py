import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon, Circle
from matplotlib.colors import Normalize
from .nozzle import CylindricalNozzle, ConvergingDivergingNozzle

class CrossSectionVisualizer:
    """
    Enhanced 2D cross-section visualization for fluid flow through nozzles
    """
    def __init__(self, config):
        # Visualization parameters
        self.slice_thickness = config.get('slice_thickness', 0.3)  # Thickness of the cross-section slice
        self.particle_size_base = config.get('particle_size', 2.0)  # Base size for particles (will be adaptive)
        self.min_particle_size = config.get('min_particle_size', 1.0)  # Minimum particle size
        self.max_particles_at_full_size = config.get('max_particles_at_full_size', 500)  # Threshold for adaptive sizing
        
        # Color settings
        self.use_velocity_color = config.get('use_velocity_color', True)  # Color by velocity instead of density
        self.show_pressure = config.get('show_pressure', False)  # Alternative: color by pressure
        self.colormap = cm.get_cmap(config.get('colormap', 'coolwarm'))  # Default colormap
        self.colormap_density = cm.get_cmap(config.get('colormap_density', 'viridis'))  # Colormap for density
        self.nozzle_color = config.get('nozzle_color', '#555555')
        self.background_color = config.get('background_color', '#f8f8f8')
        self.bg_gradient = config.get('bg_gradient', True)  # Use gradient background
        
        # Velocity visualization
        self.show_velocity_vectors = config.get('show_velocity_vectors', False)  # Toggle velocity arrows
        self.velocity_vectors_scale = config.get('velocity_vectors_scale', 0.05)
        self.velocity_vectors_density = config.get('velocity_vectors_density', 50)  # Max number of arrows
        
        # Display settings
        self.show_streamlines = config.get('show_streamlines', True)  # Show flow streamlines
        self.streamline_density = config.get('streamline_density', 1.0)  # Density of streamlines
        self.streamline_color = config.get('streamline_color', 'white')  # Color of streamlines
        self.streamline_alpha = config.get('streamline_alpha', 0.7)  # Alpha of streamlines
        self.show_velocity_magnitude = config.get('show_velocity_magnitude', True)  # Show velocity magnitude labels
        self.show_annotations = config.get('show_annotations', True)  # Show nozzle annotations
        self.show_grid = config.get('show_grid', True)  # Show grid
        self.show_axis = config.get('show_axis', True)  # Show axis
        self.pause_time = config.get('pause_time', 0.01)
        
        # Initialize figure elements
        self.fig = None
        self.ax = None
        self.particles_scatter = None
        self.velocity_quiver = None
        self.streamlines = None
        self.colorbar = None
        self.title = None
        self.annotations = []
        self.grid_cache = None
        
        # Data for streamlines
        self.grid_resolution = config.get('grid_resolution', 20)  # Grid points for interpolation
        self.velocity_grid = None
        self.grid_points_x = None
        self.grid_points_y = None
        
        # For dynamic sizing
        self.frame_counter = 0
        self.update_streamlines_every = config.get('update_streamlines_every', 5)  # Update streamlines every N frames
        
    def setup(self, simulation, nozzle):
        """Set up the visualization with initial empty plots"""
        # Create figure and axis with appropriate size
        self.fig, self.ax = plt.subplots(figsize=(14, 7))
        
        # Set background
        self.fig.patch.set_facecolor(self.background_color)
        self.ax.set_facecolor(self.background_color)
        
        # Calculate appropriate axis limits
        x_min, x_max = simulation.bound_min[0], simulation.bound_max[0]
        y_min, y_max = simulation.bound_min[1], simulation.bound_max[1]
        
        # Add some padding
        x_padding = 0.1 * (x_max - x_min)
        y_padding = 0.1 * (y_max - y_min)
        
        self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
        self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Create gradient background if enabled
        if self.bg_gradient:
            self._create_background_gradient(x_min - x_padding, x_max + x_padding, 
                                            y_min - y_padding, y_max + y_padding)
        
        # Initialize empty scatter plot for particles
        self.particles_scatter = self.ax.scatter(
            [], [], s=self.particle_size_base, c=[], cmap=self.colormap, 
            alpha=0.9, edgecolors='none', zorder=10
        )
        
        # Initialize velocity vectors if enabled
        if self.show_velocity_vectors:
            self.velocity_quiver = self.ax.quiver(
                [], [], [], [], scale=1.0, scale_units='inches', 
                width=0.002, color='red', alpha=0.7, zorder=15
            )
        
        # Initialize streamlines (empty initially)
        if self.show_streamlines:
            self.streamlines = self.ax.streamplot(
                [0], [0], [0], [0], color=self.streamline_color,
                density=self.streamline_density, linewidth=1,
                arrowstyle='->', arrowsize=1.0, zorder=5,
                alpha=self.streamline_alpha
            )
            
            # Create grid for velocity interpolation
            self.grid_points_x = np.linspace(x_min, x_max, self.grid_resolution)
            self.grid_points_y = np.linspace(y_min, y_max, self.grid_resolution)
        
        # Draw the nozzle
        self._visualize_nozzle(nozzle)
        
        # Add colorbar
        self.colorbar = self.fig.colorbar(self.particles_scatter, ax=self.ax)
        if self.use_velocity_color:
            self.colorbar.set_label('Velocity Magnitude (m/s)')
        elif self.show_pressure:
            self.colorbar.set_label('Pressure')
        else:
            self.colorbar.set_label('Density')
        
        # Add title
        self.title = self.ax.set_title("Time: 0.000s | Particles: 0", fontsize=12)
        
        # Add grid if enabled
        if self.show_grid:
            self.ax.grid(True, linestyle='--', alpha=0.3)
        
        # Set labels
        if self.show_axis:
            self.ax.set_xlabel('X position (m)', fontsize=10)
            self.ax.set_ylabel('Y position (m)', fontsize=10)
        else:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        
        # Tight layout for better use of space
        self.fig.tight_layout()
        
        # Interactive mode on
        plt.ion()
        self.fig.show()
    
    def _create_background_gradient(self, x_min, x_max, y_min, y_max):
        """Create a subtle gradient background"""
        import matplotlib.colors as mcolors
        
        # Create gradient colors - from light to lighter
        color1 = mcolors.to_rgba(self.background_color)
        color2 = mcolors.to_rgba('#ffffff')
        
        # Create gradient
        gradient = np.zeros((100, 100, 4))
        for i in range(100):
            gradient[i, :, :] = mcolors.to_rgba(
                mcolors.rgb2hex((1-i/100)*color1 + (i/100)*color2)
            )
        
        # Add gradient as image
        self.ax.imshow(gradient, aspect='auto', extent=[x_min, x_max, y_min, y_max], 
                      origin='lower', alpha=0.3, zorder=0)
    
    def _visualize_nozzle(self, nozzle):
        """Visualize the nozzle as a 2D cross-section"""
        if isinstance(nozzle, (CylindricalNozzle, ConvergingDivergingNozzle)):
            # Number of points to define nozzle outline
            n_points = 100
            
            # Generate x points along the nozzle length
            x_points = np.linspace(nozzle.position[0], 
                                  nozzle.position[0] + nozzle.length * nozzle.direction[0], 
                                  n_points)
            
            # Calculate top and bottom profiles
            top_points = []
            bottom_points = []
            
            for x in x_points:
                # Position along nozzle (0 to 1)
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
                inlet_text = self.ax.text(inlet_x, inlet_r + 0.2, "Inlet", 
                                        ha='center', va='bottom', fontsize=9,
                                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
                                        zorder=20)
                self.annotations.append(inlet_text)
                
                # Add outlet label
                outlet_text = self.ax.text(outlet_x, outlet_r + 0.2, "Outlet", 
                                         ha='center', va='bottom', fontsize=9,
                                         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
                                         zorder=20)
                self.annotations.append(outlet_text)
                
                # Add throat label for converging-diverging nozzle
                if isinstance(nozzle, ConvergingDivergingNozzle):
                    throat_x = nozzle.position[0] + nozzle.throat_position * nozzle.length
                    throat_r = nozzle.throat_diameter / 2
                    
                    throat_text = self.ax.text(throat_x, throat_r + 0.2, "Throat", 
                                             ha='center', va='bottom', fontsize=9,
                                             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'),
                                             zorder=20)
                    self.annotations.append(throat_text)
                    
                    # Add a small marker at the throat
                    throat_line = self.ax.axvline(x=throat_x, color='blue', linestyle='--', alpha=0.4, zorder=2)
                    self.annotations.append(throat_line)
    
    def _interpolate_velocity_field(self, positions, velocities):
        """Interpolate velocities onto a regular grid for streamlines"""
        from scipy.interpolate import griddata
        
        if len(positions) < 10:  # Need enough points for interpolation
            return None, None
        
        # Create grid coordinates
        XX, YY = np.meshgrid(self.grid_points_x, self.grid_points_y)
        
        # Extract positions and velocity components
        x_pos = positions[:, 0]
        y_pos = positions[:, 1]
        vx = velocities[:, 0]
        vy = velocities[:, 1]
        
        # Interpolate velocity components onto grid
        try:
            U = griddata((x_pos, y_pos), vx, (XX, YY), method='linear', fill_value=0)
            V = griddata((x_pos, y_pos), vy, (XX, YY), method='linear', fill_value=0)
            return XX, YY, U, V
        except Exception:
            # If interpolation fails, return None
            return None, None, None, None
    
    def update(self, simulation_state):
        """Update the visualization with new simulation state"""
        positions = simulation_state['positions']
        velocities = simulation_state['velocities'] 
        densities = simulation_state['densities']
        pressures = simulation_state.get('pressures', None)
        time = simulation_state['time']
        
        # Filter particles to only show those in the cross-section slice
        slice_mask = np.abs(positions[:, 2]) < self.slice_thickness
        slice_positions = positions[slice_mask]
        slice_velocities = velocities[slice_mask]
        slice_densities = densities[slice_mask]
        
        if pressures is not None:
            slice_pressures = pressures[slice_mask]
        
        if len(slice_positions) > 0:
            # Calculate color values based on settings
            if self.use_velocity_color:
                # Color by velocity magnitude
                velocity_mag = np.sqrt(np.sum(slice_velocities**2, axis=1))
                color_values = velocity_mag
                self.particles_scatter.set_cmap(self.colormap)
            elif self.show_pressure and pressures is not None:
                # Color by pressure
                color_values = slice_pressures
                self.particles_scatter.set_cmap(self.colormap)
            else:
                # Color by density
                color_values = slice_densities
                self.particles_scatter.set_cmap(self.colormap_density)
            
            # Update scatter plot
            self.particles_scatter.set_offsets(slice_positions[:, :2])
            
            # Set color with proper normalization
            if len(color_values) > 0:
                vmin = np.min(color_values)
                vmax = np.max(color_values)
                if vmax > vmin:
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    self.particles_scatter.set_array(color_values)
                    self.particles_scatter.set_norm(norm)
            
            # Dynamically adjust particle size based on number of particles
            num_particles = len(slice_positions)
            if num_particles > self.max_particles_at_full_size:
                # Reduce size for large numbers of particles
                scale_factor = self.max_particles_at_full_size / num_particles
                size = max(self.min_particle_size, self.particle_size_base * scale_factor)
            else:
                size = self.particle_size_base
            
            self.particles_scatter.set_sizes([size] * num_particles)
            
            # Update velocity vectors if enabled
            if self.show_velocity_vectors and self.velocity_quiver:
                # Downsample for quiver plot to prevent overcrowding
                if num_particles > self.velocity_vectors_density:
                    step = num_particles // self.velocity_vectors_density
                    self.velocity_quiver.set_offsets(slice_positions[::step, :2])
                    self.velocity_quiver.set_UVC(
                        slice_velocities[::step, 0] * self.velocity_vectors_scale,
                        slice_velocities[::step, 1] * self.velocity_vectors_scale
                    )
                else:
                    self.velocity_quiver.set_offsets(slice_positions[:, :2])
                    self.velocity_quiver.set_UVC(
                        slice_velocities[:, 0] * self.velocity_vectors_scale,
                        slice_velocities[:, 1] * self.velocity_vectors_scale
                    )
            
            # Update streamlines periodically
            if self.show_streamlines and self.frame_counter % self.update_streamlines_every == 0:
                # Clear previous streamlines
                if hasattr(self.streamlines, 'lines') and self.streamlines.lines:
                    for line in self.streamlines.lines:
                        if line in self.ax.lines:
                            self.ax.lines.remove(line)
                
                # Interpolate velocity field for streamlines
                XX, YY, U, V = self._interpolate_velocity_field(slice_positions, slice_velocities)
                
                if XX is not None and U is not None:
                    # Create new streamlines
                    speed = np.sqrt(U**2 + V**2)
                    lw = 1 + speed / speed.max()
                    
                    self.streamlines = self.ax.streamplot(
                        XX, YY, U, V, color=self.streamline_color,
                        density=self.streamline_density, linewidth=lw,
                        arrowstyle='->', arrowsize=1.0, zorder=5,
                        alpha=self.streamline_alpha
                    )
                
                # Add velocity magnitude indicators at key points
                if self.show_velocity_magnitude:
                    # Clear previous velocity labels
                    for annotation in self.annotations:
                        if hasattr(annotation, 'remove'):
                            annotation.remove()
                    
                    self.annotations = []
                    
                    # Add new velocity indicators at strategic points
                    velocity_points = []
                    
                    # Get velocity at regular intervals
                    x_positions = np.linspace(
                        self.ax.get_xlim()[0] + 1, 
                        self.ax.get_xlim()[1] - 1,
                        5
                    )
                    
                    for x_pos in x_positions:
                        # Find nearby particles
                        nearby = np.where(np.abs(slice_positions[:, 0] - x_pos) < 0.5)[0]
                        if len(nearby) > 0:
                            # Get average velocity at this x position
                            avg_vx = np.mean(slice_velocities[nearby, 0])
                            avg_vy = np.mean(slice_velocities[nearby, 1])
                            magnitude = np.sqrt(avg_vx**2 + avg_vy**2)
                            
                            if magnitude > 0.1:  # Only show significant velocities
                                # Find y position with highest density of particles
                                y_coord = np.mean(slice_positions[nearby, 1])
                                velocity_points.append((x_pos, y_coord, magnitude))
                    
                    # Add velocity labels (but not too many)
                    for i, (x, y, vel) in enumerate(velocity_points):
                        if i % 2 == 0:  # Skip some points for clarity
                            vel_text = self.ax.text(
                                x, y + 0.3, f"{vel:.1f} m/s", 
                                ha='center', va='bottom', fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'),
                                zorder=20
                            )
                            self.annotations.append(vel_text)
        else:
            # No particles in slice - clear plots
            self.particles_scatter.set_offsets(np.zeros((0, 2)))
            if self.show_velocity_vectors and self.velocity_quiver:
                self.velocity_quiver.set_offsets(np.zeros((0, 2)))
                self.velocity_quiver.set_UVC([], [])
        
        # Update title
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
        
        # Increment frame counter
        self.frame_counter += 1
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(self.pause_time)
    
    def close(self):
        """Close the visualization"""
        plt.close(self.fig)
