import numpy as np

class Nozzle:
    """
    Base class for nozzle geometries
    """
    def __init__(self, config):
        self.position = np.array(config.get('position', [0.0, 0.0, 0.0]))
        self.direction = np.array(config.get('direction', [1.0, 0.0, 0.0]))
        self.direction = self.direction / np.linalg.norm(self.direction)
        
        # Inflow parameters
        self.flow_rate = config.get('flow_rate', 10.0)  # particles per second
        self.inflow_velocity = config.get('inflow_velocity', 5.0)  # initial velocity magnitude
        
        # Time tracking for particle emission
        self.last_emission_time = 0.0
        self.time_per_particle = 1.0 / self.flow_rate
    
    def is_inside(self, point):
        """
        Check if a point is inside the nozzle
        """
        raise NotImplementedError("Subclasses must implement is_inside()")
    
    def get_distance(self, point):
        """
        Get the signed distance from a point to the nozzle surface
        Negative values are inside the nozzle
        """
        raise NotImplementedError("Subclasses must implement get_distance()")
    
    def emit_particles(self, current_time, particle_radius):
        """
        Emit new particles based on flow rate and time elapsed
        Returns positions and velocities of new particles
        """
        raise NotImplementedError("Subclasses must implement emit_particles()")


class CylindricalNozzle(Nozzle):
    """
    Cylindrical nozzle with circular cross-section
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Nozzle geometry parameters
        self.inlet_diameter = config.get('inlet_diameter', 2.0)
        self.outlet_diameter = config.get('outlet_diameter', 1.0)
        self.length = config.get('length', 5.0)
        
        # Compute axis-aligned basis for the nozzle
        self.up = np.array([0.0, 1.0, 0.0])
        if np.abs(np.dot(self.direction, self.up)) > 0.99:
            self.up = np.array([1.0, 0.0, 0.0])
        
        self.right = np.cross(self.direction, self.up)
        self.right = self.right / np.linalg.norm(self.right)
        
        self.up = np.cross(self.right, self.direction)
        self.up = self.up / np.linalg.norm(self.up)
    
    def is_inside(self, point):
        return self.get_distance(point) < 0
    
    def get_distance(self, point):
        # Convert to local coordinates
        local_point = point - self.position
        
        # Project onto axis
        axis_projection = np.dot(local_point, self.direction)
        
        if axis_projection < 0 or axis_projection > self.length:
            return 1.0  # Outside the nozzle along the axis
        
        # Compute radial distance
        radial_vec = local_point - axis_projection * self.direction
        radial_distance = np.linalg.norm(radial_vec)
        
        # Interpolate diameter along the nozzle
        t = axis_projection / self.length
        current_diameter = (1 - t) * self.inlet_diameter + t * self.outlet_diameter
        current_radius = current_diameter / 2.0
        
        return radial_distance - current_radius
    
    def emit_particles(self, current_time, particle_radius):
        # Check if it's time to emit particles
        n_particles_to_emit = int((current_time - self.last_emission_time) / self.time_per_particle)
        
        if n_particles_to_emit <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        
        self.last_emission_time = current_time
        
        # Create positions in a circular pattern at the inlet
        positions = []
        velocities = []
        
        # Try to fit particles in a circle at the inlet
        inlet_radius = self.inlet_diameter / 2.0 - particle_radius
        circumference = 2 * np.pi * inlet_radius
        approx_particles_per_ring = int(circumference / (2 * particle_radius))
        
        # Adjust for actual number to emit
        particles_per_ring = min(approx_particles_per_ring, n_particles_to_emit)
        
        for i in range(particles_per_ring):
            angle = 2 * np.pi * i / particles_per_ring
            offset = inlet_radius * (self.right * np.cos(angle) + self.up * np.sin(angle))
            pos = self.position - particle_radius * self.direction + offset
            vel = self.direction * self.inflow_velocity
            
            positions.append(pos)
            velocities.append(vel)
        
        return np.array(positions), np.array(velocities)


class ConvergingDivergingNozzle(CylindricalNozzle):
    """
    Converging-diverging (de Laval) nozzle with variable diameter
    """
    def __init__(self, config):
        # Override outlet_diameter with three diameters
        self.inlet_diameter = config.get('inlet_diameter', 2.0)
        self.throat_diameter = config.get('throat_diameter', 0.5)
        self.outlet_diameter = config.get('outlet_diameter', 1.5)
        self.throat_position = config.get('throat_position', 0.5)  # as fraction of length
        
        super().__init__(config)
    
    def get_distance(self, point):
        # Convert to local coordinates
        local_point = point - self.position
        
        # Project onto axis
        axis_projection = np.dot(local_point, self.direction)
        
        if axis_projection < 0 or axis_projection > self.length:
            return 1.0  # Outside the nozzle along the axis
        
        # Compute radial distance
        radial_vec = local_point - axis_projection * self.direction
        radial_distance = np.linalg.norm(radial_vec)
        
        # Determine current diameter based on position
        t = axis_projection / self.length
        
        # Converging section
        if t <= self.throat_position:
            t_section = t / self.throat_position
            current_diameter = (1 - t_section) * self.inlet_diameter + t_section * self.throat_diameter
        # Diverging section
        else:
            t_section = (t - self.throat_position) / (1 - self.throat_position)
            current_diameter = (1 - t_section) * self.throat_diameter + t_section * self.outlet_diameter
        
        current_radius = current_diameter / 2.0
        
        return radial_distance - current_radius
