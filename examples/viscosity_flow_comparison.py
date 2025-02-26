#!/usr/bin/env python

"""
Example demonstrating how different fluid viscosities affect flow through a converging-diverging nozzle.
This script runs multiple simulations in sequence, each with a different viscosity,
and displays them side by side with the enhanced 2D visualization.
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from fluid_sim.sph_core import SPHSimulation
from fluid_sim.nozzle import ConvergingDivergingNozzle
from fluid_sim.fluid import Fluid
from fluid_sim.visualization2d import CrossSectionVisualizer

# Ensure output directory exists
if not os.path.exists("output"):
    os.makedirs("output")

# Simulation parameters
NOZZLE_CONFIG = {
    'position': [0.0, 0.0, 0.0],
    'direction': [1.0, 0.0, 0.0],
    'inlet_diameter': 1.5,
    'throat_diameter': 0.4,
    'outlet_diameter': 1.0,
    'throat_position': 0.4,
    'length': 4.0,
    'flow_rate': 30.0,
    'inflow_velocity': 6.0,
}

SIM_CONFIG = {
    'particle_radius': 0.03,
    'time_step': 0.0005,
    'gravity': [0.0, -2.0, 0.0],
    'bound_min': [-1.0, -3.0, -3.0],
    'bound_max': [10.0, 3.0, 3.0],
    'bound_damping': -0.5,
}

# List of fluids to test with different viscosities
FLUIDS = [
    {'name': 'Water (Low Viscosity)', 'viscosity': 0.001, 'density': 1000.0, 'color': 'blue'},
    {'name': 'Oil (Medium Viscosity)', 'viscosity': 0.05, 'density': 900.0, 'color': 'orange'},
    {'name': 'Honey (High Viscosity)', 'viscosity': 1.0, 'density': 1400.0, 'color': 'brown'}
]

# Simulation control
MAX_TIME = 3.0  # Shorter time per simulation to keep total runtime reasonable
MAX_PARTICLES = 1500  # Fewer particles per simulation
UPDATE_FREQ = 5  # Update visualization every N steps
SLICE_THICKNESS = 0.3


def run_simulation(fluid_config, sim_config, nozzle_config, ax, index):
    """Run a single simulation for a specific fluid viscosity and display on the given axis"""
    # Create custom fluid
    fluid = Fluid.custom(
        fluid_config['name'], 
        fluid_config['density'], 
        fluid_config['viscosity']
    )
    
    # Update simulation params with fluid properties
    sim_params = sim_config.copy()
    sim_params.update(fluid.get_simulation_parameters())
    
    # Create simulation
    simulation = SPHSimulation(sim_params)
    
    # Create nozzle
    nozzle = ConvergingDivergingNozzle(nozzle_config)
    
    # Create visualization config
    viz_config = {
        'particle_size': 2.0,  # Smaller particles
        'colormap': 'coolwarm',
        'slice_thickness': SLICE_THICKNESS,
        'use_velocity_color': True,
        'show_streamlines': True,
        'streamline_density': 0.8,
        'show_velocity_vectors': False,
        'show_annotations': index == 0,  # Only show annotations on the first plot
        'show_velocity_magnitude': index == len(FLUIDS) - 1,  # Only show velocity labels on the last plot
        'pause_time': 0.01,
    }
    
    # Create visualizer that uses the provided axis
    visualizer = CrossSectionVisualizer(viz_config)
    visualizer.fig = plt.gcf()  # Use the current figure
    visualizer.ax = ax  # Use the provided axis
    
    # Set up the visualization manually since we're using a custom axis
    visualizer.setup(simulation, nozzle)
    
    # Set title with fluid info
    ax.set_title(f"{fluid_config['name']}\nViscosity: {fluid_config['viscosity']} Pa·s")
    
    print(f"\nRunning simulation for {fluid_config['name']} (viscosity={fluid_config['viscosity']} Pa·s)")
    
    # Create a progress bar for this simulation
    from tqdm import tqdm
    
    # Main simulation loop with progress bar
    sim_time = 0.0
    sim_steps = 0
    
    with tqdm(total=int(MAX_TIME / sim_params['time_step']), desc=f"Fluid {index+1}/{len(FLUIDS)}") as pbar:
        while sim_time < MAX_TIME and simulation.positions.shape[0] < MAX_PARTICLES:
            # Emit new particles
            new_positions, new_velocities = nozzle.emit_particles(sim_time, sim_params['particle_radius'])
            if len(new_positions) > 0:
                simulation.add_particles(new_positions, new_velocities)
            
            # Advance simulation
            state = simulation.step()
            sim_time = simulation.simulation_time
            sim_steps += 1
            
            # Update visualization occasionally
            if sim_steps % UPDATE_FREQ == 0:
                visualizer.update(state)
                plt.pause(0.001)  # Brief pause to update the UI
            
            pbar.update(1)
            pbar.set_description(f"Fluid {index+1}/{len(FLUIDS)}: t={sim_time:.2f}s, n={simulation.positions.shape[0]}")
    
    # Final visualization update
    visualizer.update(simulation.get_state())
    
    # Return final state data for comparison
    return {
        'time': sim_time,
        'particles': simulation.positions.shape[0],
        'velocities': simulation.velocities,
        'positions': simulation.positions,
        'densities': simulation.densities
    }


def main():
    # Set up a figure with subplots for each fluid
    fig = plt.figure(figsize=(len(FLUIDS) * 6, 8))
    gs = GridSpec(1, len(FLUIDS), figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(FLUIDS))]
    
    # Add overall title
    fig.suptitle("Comparison of Fluid Flow with Different Viscosities", fontsize=16, y=0.98)
    
    # Use interactive mode for real-time updates
    plt.ion()
    
    # Run simulations for each fluid
    results = []
    for i, fluid_config in enumerate(FLUIDS):
        result = run_simulation(fluid_config, SIM_CONFIG, NOZZLE_CONFIG, axes[i], i)
        results.append(result)
    
    # Final analysis and comparison
    print("\nSimulation Results:")
    for i, (fluid, result) in enumerate(zip(FLUIDS, results)):
        # Calculate average velocity magnitude at outlet
        positions = result['positions']
        velocities = result['velocities']
        
        # Find particles near outlet (x ≈ nozzle length)
        outlet_x = NOZZLE_CONFIG['position'][0] + NOZZLE_CONFIG['length']
        outlet_particles = np.where(np.abs(positions[:, 0] - outlet_x) < 0.5)[0]
        
        if len(outlet_particles) > 0:
            outlet_velocities = velocities[outlet_particles]
            avg_velocity = np.mean(np.sqrt(np.sum(outlet_velocities**2, axis=1)))
            max_velocity = np.max(np.sqrt(np.sum(outlet_velocities**2, axis=1)))
            
            print(f"{fluid['name']}:")
            print(f"  - Final particles: {result['particles']}")
            print(f"  - Avg velocity at outlet: {avg_velocity:.2f} m/s")
            print(f"  - Max velocity at outlet: {max_velocity:.2f} m/s")
            
            # Add this information to the plot
            axes[i].text(
                0.5, 0.02, 
                f"Avg Velocity: {avg_velocity:.2f} m/s\nMax Velocity: {max_velocity:.2f} m/s",
                ha='center', va='bottom', transform=axes[i].transAxes,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3')
            )
    
    # Make sure things look good
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for the suptitle
    
    # Save the final comparison image
    plt.savefig("output/viscosity_comparison.png", dpi=200, bbox_inches='tight')
    print("\nComparison image saved to 'output/viscosity_comparison.png'")
    
    # Keep the window open until closed
    plt.ioff()
    print("\nClose the figure window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
