#!/usr/bin/env python

"""
Example showing fluid flow through a nozzle using a 2D cross-section visualization
"""

import numpy as np
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from fluid_sim.sph_core import SPHSimulation
from fluid_sim.nozzle import CylindricalNozzle, ConvergingDivergingNozzle
from fluid_sim.fluid import Fluid
from fluid_sim.visualization2d import CrossSectionVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='2D Cross-section fluid flow visualization')
    parser.add_argument('--nozzle_type', choices=['cylindrical', 'converging-diverging'], 
                        default='converging-diverging', help='Type of nozzle')
    parser.add_argument('--particle_size', type=float, default=5.0, help='Visual size of particles')
    parser.add_argument('--fluid', choices=['water', 'oil', 'honey'], default='water', 
                        help='Type of fluid')
    parser.add_argument('--max_time', type=float, default=5.0, help='Maximum simulation time (s)')
    parser.add_argument('--max_particles', type=int, default=3000, help='Maximum number of particles')
    parser.add_argument('--colormap', default='viridis', help='Colormap for visualization')
    parser.add_argument('--show_pressure', action='store_true', help='Show pressure instead of density')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Simulation parameters
    sim_config = {
        'particle_radius': 0.03,  # Smaller particles for better visualization
        'time_step': 0.0005,
        'gravity': [0.0, -2.0, 0.0],  # Reduced gravity for better visualization
        'bound_min': [-1.0, -3.0, -3.0],
        'bound_max': [15.0, 3.0, 3.0],
        'bound_damping': -0.5,
    }
    
    # Create fluid based on argument
    if args.fluid == 'water':
        fluid = Fluid.water()
    elif args.fluid == 'oil':
        fluid = Fluid.oil()
    elif args.fluid == 'honey':
        fluid = Fluid.honey()
    else:
        fluid = Fluid.water()
    
    sim_config.update(fluid.get_simulation_parameters())
    
    # Create simulation
    simulation = SPHSimulation(sim_config)
    
    # Define nozzle parameters
    if args.nozzle_type == 'cylindrical':
        nozzle_config = {
            'position': [0.0, 0.0, 0.0],
            'direction': [1.0, 0.0, 0.0],
            'inlet_diameter': 1.2,
            'outlet_diameter': 0.6,
            'length': 4.0,
            'flow_rate': 30.0,  # Higher flow rate for better visualization
            'inflow_velocity': 5.0,
        }
        nozzle = CylindricalNozzle(nozzle_config)
        print(f"Created cylindrical nozzle (inlet: {nozzle_config['inlet_diameter']}, outlet: {nozzle_config['outlet_diameter']})")
    else:
        nozzle_config = {
            'position': [0.0, 0.0, 0.0],
            'direction': [1.0, 0.0, 0.0],
            'inlet_diameter': 1.5,
            'throat_diameter': 0.4,
            'outlet_diameter': 1.0,
            'throat_position': 0.4,
            'length': 4.0,
            'flow_rate': 30.0,  # Higher flow rate for better visualization
            'inflow_velocity': 5.0,
        }
        nozzle = ConvergingDivergingNozzle(nozzle_config)
        print(f"Created converging-diverging nozzle (throat diameter: {nozzle_config['throat_diameter']})")
    
    # Setup 2D cross-section visualization
    viz_config = {
        'show_velocity': True,
        'show_density': True,
        'particle_scale': args.particle_size,  # Small particles as requested
        'colormap': args.colormap,
        'slice_thickness': 0.5,  # Thickness of the cross-section slice
        'velocity_scale': 0.1,  # Scale factor for velocity arrows
        'update_freq': 5,
        'pause_time': 0.01,
        'show_pressure': args.show_pressure,
    }
    visualizer = CrossSectionVisualizer(viz_config)
    visualizer.setup(simulation, nozzle)
    
    # Simulation control parameters
    max_time = args.max_time
    max_particles = args.max_particles
    particle_radius = sim_config['particle_radius']
    
    print(f"Starting simulation with fluid: {args.fluid}")
    print(f"Max simulation time: {max_time}s, Max particles: {max_particles}")
    print(f"2D cross-section visualization enabled, particle size: {args.particle_size}")
    
    # Main simulation loop
    try:
        start_time = time.time()
        sim_steps = 0
        
        with tqdm(total=int(max_time / sim_config['time_step'])) as pbar:
            while simulation.simulation_time < max_time and simulation.positions.shape[0] < max_particles:
                # Emit new particles from the nozzle
                new_positions, new_velocities = nozzle.emit_particles(simulation.simulation_time, particle_radius)
                if len(new_positions) > 0:
                    simulation.add_particles(new_positions, new_velocities)
                
                # Advance simulation
                state = simulation.step()
                sim_steps += 1
                
                # Update visualization (every n steps)
                if sim_steps % viz_config['update_freq'] == 0:
                    visualizer.update(state)
                
                pbar.update(1)
                pbar.set_description(f"Time: {simulation.simulation_time:.2f}s, Particles: {simulation.positions.shape[0]}")
        
        elapsed = time.time() - start_time
        print(f"\nSimulation completed in {elapsed:.2f} seconds")
        print(f"Simulated {simulation.simulation_time:.2f} seconds with {simulation.positions.shape[0]} particles")
        print(f"Performance: {sim_steps / elapsed:.2f} steps/second")
        
        # Keep visualization window open
        print("Simulation finished. Close the visualization window to exit.")
        plt.ioff()
        plt.show()
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    finally:
        visualizer.close()

if __name__ == "__main__":
    main()
