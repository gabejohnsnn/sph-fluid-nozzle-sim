#!/usr/bin/env python

import numpy as np
import argparse
import time
from tqdm import tqdm
from fluid_sim.sph_core import SPHSimulation
from fluid_sim.nozzle import CylindricalNozzle, ConvergingDivergingNozzle
from fluid_sim.fluid import Fluid
from fluid_sim.visualization import MatplotlibVisualizer, ParticleDataExporter
from fluid_sim.utilities import Config, DataAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description='Fluid Flow Simulation')
    
    # Fluid parameters
    parser.add_argument('--fluid_viscosity', type=float, help='Fluid viscosity (Pa.s)')
    parser.add_argument('--fluid_density', type=float, help='Fluid density (kg/m^3)')
    
    # Nozzle parameters
    parser.add_argument('--nozzle_type', choices=['cylindrical', 'converging-diverging'], 
                        help='Type of nozzle')
    parser.add_argument('--inlet_diameter', type=float, help='Inlet diameter (m)')
    parser.add_argument('--outlet_diameter', type=float, help='Outlet diameter (m)')
    parser.add_argument('--throat_diameter', type=float, 
                        help='Throat diameter for converging-diverging nozzle (m)')
    parser.add_argument('--flow_rate', type=float, help='Flow rate (particles/s)')
    
    # Simulation control
    parser.add_argument('--max_time', type=float, help='Maximum simulation time (s)')
    parser.add_argument('--max_particles', type=int, help='Maximum number of particles')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    # Visualization control
    parser.add_argument('--no_visualization', action='store_true', 
                        help='Disable visualization')
    parser.add_argument('--export_data', action='store_true', 
                        help='Export simulation data')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    config.update_from_args(args)
    
    # Set random seed
    np.random.seed(config.get_control_config()['seed'])
    
    # Create fluid
    fluid_config = config.get_fluid_config()
    fluid = Fluid(fluid_config)
    
    # Prepare simulation parameters
    sim_config = config.get_simulation_config()
    sim_config.update(fluid.get_simulation_parameters())
    
    # Create simulation
    simulation = SPHSimulation(sim_config)
    
    # Create nozzle
    nozzle_config = config.get_nozzle_config()
    if nozzle_config['type'] == 'converging-diverging':
        nozzle = ConvergingDivergingNozzle(nozzle_config)
    else:
        nozzle = CylindricalNozzle(nozzle_config)
    
    # Setup visualization if enabled
    visualizer = None
    if not args.no_visualization:
        viz_config = config.get_visualization_config()
        visualizer = MatplotlibVisualizer(viz_config)
        visualizer.setup(simulation, nozzle)
    
    # Setup data exporter if enabled
    exporter = None
    if args.export_data:
        export_config = config.get_export_config()
        exporter = ParticleDataExporter(export_config)
    
    # Data analyzer for post-processing
    analyzer = DataAnalyzer()
    
    # Simulation control parameters
    control_config = config.get_control_config()
    max_time = control_config['max_time']
    max_particles = control_config['max_particles']
    
    # Main simulation loop
    particle_radius = sim_config['particle_radius']
    update_freq = config.get_visualization_config()['update_freq']
    
    print(f"Starting simulation with {max_particles} max particles and {max_time}s duration")
    print(f"Fluid: {fluid_config['name']} (viscosity: {fluid_config['viscosity']}, density: {fluid_config['density']})")
    
    try:
        start_time = time.time()
        sim_steps = 0
        
        with tqdm(total=int(max_time / sim_config['time_step']), desc="Simulating") as pbar:
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
            
            # Velocity profile
            if nozzle_config['type'] == 'converging-diverging':
                # At throat
                throat_pos = nozzle.position[0] + nozzle.throat_position * nozzle.length * nozzle.direction[0]
                radial_pos, velocities = analyzer.get_velocity_profile(throat_pos, nozzle.throat_diameter / 2)
                if len(radial_pos) > 0:
                    print(f"Maximum velocity at throat: {np.max(velocities):.2f} m/s")
            
            # At outlet
            radial_pos, velocities = analyzer.get_velocity_profile(outlet_pos, nozzle.outlet_diameter / 2)
            if len(radial_pos) > 0:
                print(f"Maximum velocity at outlet: {np.max(velocities):.2f} m/s")
        
        # Allow for interactive viewing if visualization is enabled
        if visualizer:
            print("Simulation finished. Close the visualization window to exit.")
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.show()
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    finally:
        if visualizer:
            visualizer.close()

if __name__ == "__main__":
    main()
