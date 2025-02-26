#!/usr/bin/env python

"""
Example of a cylindrical nozzle simulation
"""

import numpy as np
import time
from fluid_sim.sph_core import SPHSimulation
from fluid_sim.nozzle import CylindricalNozzle
from fluid_sim.fluid import Fluid
from fluid_sim.visualization import MatplotlibVisualizer
import matplotlib.pyplot as plt

# Simulation parameters
sim_config = {
    'particle_radius': 0.05,
    'time_step': 0.001,
    'gravity': [0.0, -9.81, 0.0],
    'bound_min': [-1.0, -5.0, -5.0],
    'bound_max': [15.0, 5.0, 5.0],
    'bound_damping': -0.5,
}

# Create fluid (water)
fluid = Fluid.water()
sim_config.update(fluid.get_simulation_parameters())

# Create simulation
simulation = SPHSimulation(sim_config)

# Create nozzle
nozzle_config = {
    'position': [0.0, 0.0, 0.0],
    'direction': [1.0, 0.0, 0.0],
    'inlet_diameter': 1.0,
    'outlet_diameter': 0.5,
    'length': 3.0,
    'flow_rate': 20.0,
    'inflow_velocity': 5.0,
}
nozzle = CylindricalNozzle(nozzle_config)

# Setup visualization
viz_config = {
    'show_velocity': True,
    'show_density': True,
    'particle_scale': 50.0,
    'colormap': 'viridis',
    'update_freq': 5,
    'pause_time': 0.01,
}
visualizer = MatplotlibVisualizer(viz_config)
visualizer.setup(simulation, nozzle)

# Simulation parameters
max_time = 5.0
max_particles = 2000

# Main simulation loop
try:
    start_time = time.time()
    while simulation.simulation_time < max_time and simulation.positions.shape[0] < max_particles:
        # Emit new particles from the nozzle
        new_positions, new_velocities = nozzle.emit_particles(simulation.simulation_time, sim_config['particle_radius'])
        if len(new_positions) > 0:
            simulation.add_particles(new_positions, new_velocities)
        
        # Advance simulation
        state = simulation.step()
        
        # Update visualization (every 5 steps)
        if int(simulation.simulation_time / sim_config['time_step']) % 5 == 0:
            visualizer.update(state)
        
        print(f"\rTime: {simulation.simulation_time:.2f}s, Particles: {simulation.positions.shape[0]}", end="")
    
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.2f} seconds")
    print(f"Simulated {simulation.simulation_time:.2f} seconds with {simulation.positions.shape[0]} particles")
    print(f"Performance: {int(simulation.simulation_time / sim_config['time_step']) / elapsed:.2f} steps/second")
    
    # Keep visualization window open
    plt.ioff()
    plt.show()

except KeyboardInterrupt:
    print("\nSimulation interrupted by user")

finally:
    visualizer.close()
