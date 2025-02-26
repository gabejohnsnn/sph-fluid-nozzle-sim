#!/usr/bin/env python

"""
Example showing fluid flow through a nozzle using an enhanced 2D cross-section visualization
with streamlines, velocity coloring, and improved aesthetics
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
    parser = argparse.ArgumentParser(description='Enhanced 2D Cross-section fluid flow visualization')
    
    # Nozzle options
    parser.add_argument('--nozzle_type', choices=['cylindrical', 'converging-diverging'], 
                        default='converging-diverging', help='Type of nozzle')
    parser.add_argument('--inlet_diameter', type=float, default=1.5, help='Inlet diameter (m)')
    parser.add_argument('--outlet_diameter', type=float, default=1.0, help='Outlet diameter (m)')
    parser.add_argument('--throat_diameter', type=float, default=0.4, help='Throat diameter (m) for converging-diverging nozzle')
    parser.add_argument('--throat_position', type=float, default=0.4, help='Position of throat as fraction of length')
    parser.add_argument('--nozzle_length', type=float, default=4.0, help='Length of nozzle (m)')
    
    # Fluid options
    parser.add_argument('--fluid', choices=['water', 'oil', 'honey', 'custom'], default='water', 
                        help='Type of fluid')
    parser.add_argument('--viscosity', type=float, default=None, 
                        help='Custom fluid viscosity (Pa.s), use with --fluid custom')
    parser.add_argument('--density', type=float, default=None, 
                        help='Custom fluid density (kg/m^3), use with --fluid custom')
    
    # Simulation options
    parser.add_argument('--gravity', type=float, default=-2.0, help='Gravity in y-direction (m/s^2)')
    parser.add_argument('--particle_radius', type=float, default=0.03, help='Radius of fluid particles (m)')
    parser.add_argument('--flow_rate', type=float, default=30.0, help='Flow rate (particles/second)')
    parser.add_argument('--inflow_velocity', type=float, default=5.0, help='Initial velocity (m/s)')
    parser.add_argument('--max_time', type=float, default=5.0, help='Maximum simulation time (s)')
    parser.add_argument('--max_particles', type=int, default=3000, help='Maximum number of particles')
    
    # Visualization options
    parser.add_argument('--particle_size', type=float, default=3.0, help='Visual size of particles')
    parser.add_argument('--colormap', default='coolwarm', help='Colormap for velocity visualization')
    parser.add_argument('--density_colormap', default='viridis', help='Colormap for density visualization')
    parser.add_argument('--color_by', choices=['velocity', 'density', 'pressure'], default='velocity',
                      help='Property to use for particle coloring')
    parser.add_argument('--streamlines', action='store_true', default=True, help='Show streamlines')
    parser.add_argument('--velocity_vectors', action='store_true', help='Show velocity vectors')
    parser.add_argument('--slice_thickness', type=float, default=0.3, help='Thickness of the cross-section slice')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Simulation parameters
    sim_config = {
        'particle_radius': args.particle_radius,
        'time_step': 0.0005,
        'gravity': [0.0, args.gravity, 0.0],
        'bound_min': [-1.0, -3.0, -3.0],
        'bound_max': [args.nozzle_length + 10.0, 3.0, 3.0],
        'bound_damping': -0.5,
    }
    
    # Create fluid based on argument
    if args.fluid == 'water':
        fluid = Fluid.water()
    elif args.fluid == 'oil':
        fluid = Fluid.oil()
    elif args.fluid == 'honey':
        fluid = Fluid.honey()
    elif args.fluid == 'custom' and args.viscosity is not None and args.density is not None:
        fluid = Fluid.custom('Custom Fluid', args.density, args.viscosity)
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
            'inlet_diameter': args.inlet_diameter,
            'outlet_diameter': args.outlet_diameter,
            'length': args.nozzle_length,
            'flow_rate': args.flow_rate,
            'inflow_velocity': args.inflow_velocity,
        }
        nozzle = CylindricalNozzle(nozzle_config)
        print(f"Created cylindrical nozzle (inlet: {nozzle_config['inlet_diameter']}, outlet: {nozzle_config['outlet_diameter']})")
    else:
        nozzle_config = {
            'position': [0.0, 0.0, 0.0],
            'direction': [1.0, 0.0, 0.0],
            'inlet_diameter': args.inlet_diameter,
            'throat_diameter': args.throat_diameter,
            'outlet_diameter': args.outlet_diameter,
            'throat_position': args.throat_position,
            'length': args.nozzle_length,
            'flow_rate': args.flow_rate,
            'inflow_velocity': args.inflow_velocity,
        }
        nozzle = ConvergingDivergingNozzle(nozzle_config)
        print(f"Created converging-diverging nozzle (throat diameter: {nozzle_config['throat_diameter']})")
    
    # Setup 2D cross-section visualization
    viz_config = {
        'particle_size': args.particle_size,
        'colormap': args.colormap,
        'colormap_density': args.density_colormap,
        'slice_thickness': args.slice_thickness,
        'use_velocity_color': args.color_by == 'velocity',
        'show_pressure': args.color_by == 'pressure',
        'show_streamlines': args.streamlines,
        'show_velocity_vectors': args.velocity_vectors,
        'velocity_vectors_scale': 0.05,
        'update_streamlines_every': 5,
        'streamline_density': 1.0,
        'streamline_color': 'white',
        'streamline_alpha': 0.7,
        'show_annotations': True,
        'show_velocity_magnitude': True,
        'bg_gradient': True,
        'pause_time': 0.01,
    }
    visualizer = CrossSectionVisualizer(viz_config)
    visualizer.setup(simulation, nozzle)
    
    # Simulation control parameters
    max_time = args.max_time
    max_particles = args.max_particles
    particle_radius = sim_config['particle_radius']
    
    print(f"\nStarting simulation:")
    print(f"  - Fluid: {fluid.name}")
    print(f"  - Viscosity: {fluid.viscosity:.6f} Pa·s")
    print(f"  - Density: {fluid.density:.1f} kg/m³")
    print(f"  - Nozzle type: {args.nozzle_type}")
    print(f"  - Max time: {max_time}s, Max particles: {max_particles}")
    print(f"  - Flow rate: {args.flow_rate} particles/s")
    print(f"  - Visualization: Coloring by {args.color_by}, {'with' if args.streamlines else 'without'} streamlines")
    
    # Main simulation loop
    try:
        start_time = time.time()
        sim_steps = 0
        update_freq = 5  # Visualization update frequency
        
        with tqdm(total=int(max_time / sim_config['time_step']), desc="Simulating") as pbar:
            while simulation.simulation_time < max_time and simulation.positions.shape[0] < max_particles:
                # Emit new particles from the nozzle
                new_positions, new_velocities = nozzle.emit_particles(simulation.simulation_time, particle_radius)
                if len(new_positions) > 0:
                    simulation.add_particles(new_positions, new_velocities)
                
                # Advance simulation
                state = simulation.step()
                sim_steps += 1
                
                # Update visualization (every n steps)
                if sim_steps % update_freq == 0:
                    visualizer.update(state)
                
                pbar.update(1)
                pbar.set_description(f"Time: {simulation.simulation_time:.2f}s, Particles: {simulation.positions.shape[0]}")
        
        # Final update to make sure we see the last state
        visualizer.update(simulation.get_state())
        
        elapsed = time.time() - start_time
        print(f"\nSimulation completed in {elapsed:.2f} seconds")
        print(f"Simulated {simulation.simulation_time:.2f} seconds with {simulation.positions.shape[0]} particles")
        print(f"Performance: {sim_steps / elapsed:.2f} steps/second")
        
        # Keep visualization window open until closed by user
        print("Simulation finished. Close the visualization window to exit.")
        plt.ioff()
        plt.show()
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    finally:
        visualizer.close()

if __name__ == "__main__":
    main()
