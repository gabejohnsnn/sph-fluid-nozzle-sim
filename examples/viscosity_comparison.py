#!/usr/bin/env python

"""
Example comparing the flow of fluids with different viscosities through the same nozzle
This script runs multiple simulations in sequence with different viscosity values
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fluid_sim.sph_core import SPHSimulation
from fluid_sim.nozzle import ConvergingDivergingNozzle
from fluid_sim.fluid import Fluid
from fluid_sim.visualization import ParticleDataExporter

# Create output directory
if not os.path.exists("output"):
    os.makedirs("output")

# Nozzle configuration (same for all simulations)
nozzle_config = {
    'position': [0.0, 0.0, 0.0],
    'direction': [1.0, 0.0, 0.0],
    'inlet_diameter': 1.2,
    'throat_diameter': 0.4,
    'outlet_diameter': 0.8,
    'throat_position': 0.4,
    'length': 3.0,
    'flow_rate': 15.0,
    'inflow_velocity': 6.0,
}

# Base simulation parameters
base_sim_config = {
    'particle_radius': 0.05,
    'time_step': 0.001,
    'gravity': [0.0, -1.0, 0.0],  # Reduced gravity for better visualization
    'bound_min': [-1.0, -3.0, -3.0],
    'bound_max': [10.0, 3.0, 3.0],
    'bound_damping': -0.5,
}

# Viscosity values to test
viscosities = [0.001, 0.01, 0.1, 1.0]  # From water-like to honey-like
fluid_names = ["Water", "Oil", "Syrup", "Honey"]

# Simulation parameters
max_time_per_sim = 2.0
max_particles = 1000

# Data collection for comparing results
all_results = {}

for i, (viscosity, name) in enumerate(zip(viscosities, fluid_names)):
    print(f"\nRunning simulation {i+1}/{len(viscosities)}: {name} (viscosity={viscosity})")
    
    # Create fluid
    fluid = Fluid({
        'name': name,
        'density': 1000.0,
        'viscosity': viscosity,
        'surface_tension': 0.072,
        'gas_constant': 2000.0
    })
    
    # Create simulation config with fluid properties
    sim_config = base_sim_config.copy()
    sim_config.update(fluid.get_simulation_parameters())
    
    # Create simulation
    simulation = SPHSimulation(sim_config)
    
    # Create nozzle
    nozzle = ConvergingDivergingNozzle(nozzle_config)
    
    # Setup data exporter
    exporter = ParticleDataExporter({
        'export_folder': f"output/{name.lower()}",
        'export_interval': 0.05,
    })
    
    # Store frames for analysis
    frames = []
    
    # Main simulation loop
    start_time = time.time()
    
    while simulation.simulation_time < max_time_per_sim and simulation.positions.shape[0] < max_particles:
        # Emit new particles from the nozzle
        new_positions, new_velocities = nozzle.emit_particles(simulation.simulation_time, sim_config['particle_radius'])
        if len(new_positions) > 0:
            simulation.add_particles(new_positions, new_velocities)
        
        # Advance simulation
        state = simulation.step()
        
        # Export data
        if exporter.export(state):
            # Store frame for analysis
            frames.append({
                'time': simulation.simulation_time,
                'positions': simulation.positions.copy(),
                'velocities': simulation.velocities.copy(),
                'densities': simulation.densities.copy(),
            })
        
        # Print progress
        print(f"\rTime: {simulation.simulation_time:.2f}s, Particles: {simulation.positions.shape[0]}", end="")
    
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.2f} seconds")
    
    # Store results for comparison
    all_results[name] = {
        'viscosity': viscosity,
        'frames': frames,
        'final_time': simulation.simulation_time,
        'final_particles': simulation.positions.shape[0],
    }

# Generate comparison visualizations
print("\nGenerating comparison visualizations...")

# Plot final particle positions for each viscosity
plt.figure(figsize=(15, 10))

for i, name in enumerate(fluid_names):
    results = all_results[name]
    final_frame = results['frames'][-1]
    positions = final_frame['positions']
    
    # Create subplot
    plt.subplot(2, 2, i+1)
    
    # Plot nozzle outline
    nozzle = ConvergingDivergingNozzle(nozzle_config)
    x_positions = np.linspace(0, nozzle.length, 100)
    y_top = []
    y_bottom = []
    
    for x in x_positions:
        # Calculate nozzle diameter at this position
        t = x / nozzle.length
        if t <= nozzle.throat_position:
            t_section = t / nozzle.throat_position
            diameter = (1 - t_section) * nozzle.inlet_diameter + t_section * nozzle.throat_diameter
        else:
            t_section = (t - nozzle.throat_position) / (1 - nozzle.throat_position)
            diameter = (1 - t_section) * nozzle.throat_diameter + t_section * nozzle.outlet_diameter
        
        radius = diameter / 2.0
        y_top.append(radius)
        y_bottom.append(-radius)
    
    plt.plot(x_positions, y_top, 'k-')
    plt.plot(x_positions, y_bottom, 'k-')
    
    # Plot particles (only show X and Y coordinates for 2D visualization)
    plt.scatter(positions[:, 0], positions[:, 1], s=20, alpha=0.6)
    
    plt.title(f"{name} (viscosity={results['viscosity']})")
    plt.xlim(-0.5, nozzle.length + 3)
    plt.ylim(-2, 2)
    plt.xlabel('X')
    plt.ylabel('Y')

plt.tight_layout()
plt.savefig("output/viscosity_comparison.png", dpi=300)
plt.close()

print("Comparison visualization saved to 'output/viscosity_comparison.png'")
print("Individual simulation data saved in 'output/' directory")
