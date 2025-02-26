#!/usr/bin/env python

"""
Example that demonstrates loading simulation parameters from a configuration file
"""

import sys
import os

# Add the project root to Python's path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from fluid_sim.sph_core import SPHSimulation
from fluid_sim.nozzle import CylindricalNozzle, ConvergingDivergingNozzle
from fluid_sim.fluid import Fluid
from fluid_sim.visualization import MatplotlibVisualizer, ParticleDataExporter
from fluid_sim.utilities import Config, DataAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description='Config-based Fluid Simulation')
    parser.add_argument('--config', type=str, default='config/default.json',
                        help='Configuration file path')
    parser.add_argument('--no_visualization', action='store_true', 
                        help='Disable visualization')
    parser.add_argument('--export_data', action='store_true', 
                        help='Export simulation data')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = Config(args.config)
    
    # Create fluid
    fluid_config = config.get_fluid_config()
    fluid = Fluid(fluid_config)
    print(f"Created fluid: {fluid_config['name']}")
    
    # Prepare simulation parameters
    sim_config = config.get_simulation_config()
    sim_config.update(fluid.get_simulation_parameters())
    
    # Create simulation
    simulation = SPHSimulation(sim_config)
    
    # Create nozzle
    nozzle_config = config.get_nozzle_config()
    if nozzle_config['type'] == 'converging-diverging':
        nozzle = ConvergingDivergingNozzle(nozzle_config)
        print(f"Created converging-diverging nozzle (throat diameter: {nozzle_config['throat_diameter']})")
    else:
        nozzle = CylindricalNozzle(nozzle_config)
        print(f"Created cylindrical nozzle (inlet: {nozzle_config['inlet_diameter']}, outlet: {nozzle_config['outlet_diameter']})")
    
    # Setup visualization if enabled
    visualizer = None
    if not args.no_visualization:
        viz_config = config.get_visualization_config()
        visualizer = MatplotlibVisualizer(viz_config)
        visualizer.setup(simulation, nozzle)
        print("Visualization enabled")
    
    # Setup data exporter if enabled
    exporter = None
    if args.export_data:
        export_config = config.get_export_config()
        exporter = ParticleDataExporter(export_config)
        print(f"Data export enabled (folder: {export_config['export_folder']})")
    
    # Data analyzer
    analyzer = DataAnalyzer()
    
    # Simulation control parameters
    control_config = config.get_control_config()
    max_time = control_config['max_time']
    max_particles = control_config['max_particles']
    
    # Main simulation loop
    particle_radius = sim_config['particle_radius']
    update_freq = config.get_visualization_config()['update_freq']
    
    print(f"Starting simulation (max time: {max_time}s, max particles: {max_particles})")
    
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
                
                # Update visualization
                if visualizer and sim_steps % update_freq == 0:
                    visualizer.update(state)
                
                # Export data if enabled
                if exporter:
                    exporter.export(state)
                
                # Store data for analysis
                if sim_steps % 10 == 0:
                    analyzer.add_snapshot(
                        simulation.simulation_time,
                        simulation.positions.copy(),
                        simulation.densities.copy(),
                        simulation.velocities.copy()
                    )
                
                pbar.update(1)
                pbar.set_description(f"Time: {simulation.simulation_time:.2f}s, Particles: {simulation.positions.shape[0]}")
        
        elapsed = time.time() - start_time
        print(f"\nSimulation completed in {elapsed:.2f} seconds")
        print(f"Simulated {simulation.simulation_time:.2f} seconds with {simulation.positions.shape[0]} particles")
        print(f"Performance: {sim_steps / elapsed:.2f} steps/second")
        
        # Calculate flow rates and velocity profiles
        if simulation.positions.shape[0] > 0:
            # Flow rate at outlet
            outlet_pos = nozzle.position[0] + nozzle.length * nozzle.direction[0]
            flow_rate = analyzer.get_flow_rate(outlet_pos)
            print(f"Average flow rate at outlet: {flow_rate:.2f} particles/s")
            
            # Velocity profile at outlet
            radial_pos, velocities = analyzer.get_velocity_profile(outlet_pos, nozzle.outlet_diameter / 2)
            if len(radial_pos) > 0:
                print(f"Maximum velocity at outlet: {max(velocities):.2f} m/s")
            
            # Generate simple velocity profile plot
            if len(radial_pos) > 0:
                plt.figure(figsize=(8, 6))
                plt.plot(radial_pos, velocities, 'o-', linewidth=2)
                plt.title(f"Velocity Profile at Outlet (t={simulation.simulation_time:.2f}s)")
                plt.xlabel("Radial Position (m)")
                plt.ylabel("Axial Velocity (m/s)")
                plt.grid(True)
                plt.savefig("velocity_profile.png", dpi=200)
                print("Velocity profile plot saved to 'velocity_profile.png'")
        
        # Allow for interactive viewing if visualization is enabled
        if visualizer:
            print("Simulation finished. Close the visualization window to exit.")
            plt.ioff()
            plt.show()
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    finally:
        if visualizer:
            visualizer.close()

if __name__ == "__main__":
    main()
