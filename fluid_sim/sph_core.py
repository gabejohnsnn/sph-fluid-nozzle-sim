import numpy as np
from numba import jit
import time

class SPHSimulation:
    """
    Smoothed Particle Hydrodynamics (SPH) simulation for fluid flow
    """
    def __init__(self, config):
        # Simulation parameters
        self.particle_radius = config.get('particle_radius', 0.1)
        self.h = config.get('smoothing_length', 4.0 * self.particle_radius)  # Smoothing length
        self.mass = config.get('particle_mass', 1.0)
        self.rest_density = config.get('rest_density', 1000.0)
        self.gas_constant = config.get('gas_constant', 2000.0)
        self.viscosity = config.get('viscosity', 0.1)
        self.gravity = np.array(config.get('gravity', [0.0, -9.81, 0.0]))
        self.dt = config.get('time_step', 0.001)
        self.eps = 1e-6  # Small value to prevent division by zero
        
        # Boundary parameters
        self.bound_min = np.array(config.get('bound_min', [-10.0, -10.0, -10.0]))
        self.bound_max = np.array(config.get('bound_max', [10.0, 10.0, 10.0]))
        self.bound_damping = config.get('bound_damping', -0.5)
        
        # Particle data
        self.positions = np.zeros((0, 3))
        self.velocities = np.zeros((0, 3))
        self.accelerations = np.zeros((0, 3))
        self.densities = np.zeros(0)
        self.pressures = np.zeros(0)
        
        # Timing
        self.simulation_time = 0.0
    
    def add_particles(self, new_positions, new_velocities=None):
        """
        Add particles to the simulation
        """
        n_new = new_positions.shape[0]
        
        if new_velocities is None:
            new_velocities = np.zeros((n_new, 3))
        
        self.positions = np.vstack((self.positions, new_positions))
        self.velocities = np.vstack((self.velocities, new_velocities))
        self.accelerations = np.zeros((self.positions.shape[0], 3))
        self.densities = np.zeros(self.positions.shape[0])
        self.pressures = np.zeros(self.positions.shape[0])
    
    @staticmethod
    @jit(nopython=True)
    def _compute_density_pressure(positions, h, mass, rest_density, gas_constant, n_particles):
        """
        Compute density and pressure for all particles
        """
        densities = np.zeros(n_particles)
        pressures = np.zeros(n_particles)
        
        # Kernel coefficient
        poly6_coeff = 315.0 / (64.0 * np.pi * h**9)
        
        for i in range(n_particles):
            # Self contribution to density
            densities[i] = mass * poly6_coeff * h**6
            
            for j in range(n_particles):
                if i == j:
                    continue
                    
                # Compute distance
                rij = positions[j] - positions[i]
                r2 = np.sum(rij**2)
                
                if r2 < h**2:
                    # W_poly6 contribution
                    densities[i] += mass * poly6_coeff * (h**2 - r2)**3
        
        # Compute pressure from density using equation of state
        for i in range(n_particles):
            pressures[i] = gas_constant * (densities[i] - rest_density)
        
        return densities, pressures
    
    @staticmethod
    @jit(nopython=True)
    def _compute_accelerations(positions, velocities, densities, pressures, h, mass, viscosity, gravity, n_particles):
        """
        Compute accelerations for all particles
        """
        accelerations = np.zeros((n_particles, 3))
        
        # Kernel derivatives coefficients
        spiky_grad_coeff = -45.0 / (np.pi * h**6)
        viscosity_lap_coeff = 45.0 / (np.pi * h**6)
        
        for i in range(n_particles):
            # Apply gravity
            accelerations[i] = gravity
            
            for j in range(n_particles):
                if i == j:
                    continue
                    
                # Compute distance and direction
                rij = positions[j] - positions[i]
                r = np.sqrt(np.sum(rij**2))
                
                if r < h:
                    # Normalized direction
                    rij_norm = rij / (r + 1e-10)
                    
                    # Pressure force (using spiky kernel gradient)
                    pressure_force = -mass * (
                        pressures[i] / (densities[i]**2 + 1e-10) + 
                        pressures[j] / (densities[j]**2 + 1e-10)
                    ) * spiky_grad_coeff * (h - r)**2 * rij_norm
                    
                    # Viscosity force (using viscosity laplacian)
                    viscosity_force = viscosity * mass * (
                        velocities[j] - velocities[i]
                    ) / (densities[j] + 1e-10) * viscosity_lap_coeff * (h - r)
                    
                    accelerations[i] += pressure_force + viscosity_force
        
        return accelerations
    
    def step(self):
        """
        Advance the simulation by one time step
        """
        n_particles = self.positions.shape[0]
        
        # Compute density and pressure
        self.densities, self.pressures = self._compute_density_pressure(
            self.positions, self.h, self.mass, self.rest_density, 
            self.gas_constant, n_particles
        )
        
        # Compute accelerations
        self.accelerations = self._compute_accelerations(
            self.positions, self.velocities, self.densities, self.pressures,
            self.h, self.mass, self.viscosity, self.gravity, n_particles
        )
        
        # Update velocities and positions (symplectic Euler)
        self.velocities += self.accelerations * self.dt
        self.positions += self.velocities * self.dt
        
        # Handle boundaries
        self._handle_boundaries()
        
        self.simulation_time += self.dt
        
        return {
            'positions': self.positions,
            'velocities': self.velocities,
            'densities': self.densities,
            'time': self.simulation_time
        }
    
    def _handle_boundaries(self):
        """
        Handle boundary collisions
        """
        for i in range(self.positions.shape[0]):
            for d in range(3):  # For each dimension (x, y, z)
                if self.positions[i, d] < self.bound_min[d]:
                    self.positions[i, d] = self.bound_min[d]
                    self.velocities[i, d] *= self.bound_damping
                
                if self.positions[i, d] > self.bound_max[d]:
                    self.positions[i, d] = self.bound_max[d]
                    self.velocities[i, d] *= self.bound_damping

    def get_state(self):
        """
        Get the current state of the simulation
        """
        return {
            'positions': self.positions,
            'velocities': self.velocities,
            'densities': self.densities,
            'pressures': self.pressures,
            'time': self.simulation_time
        }
