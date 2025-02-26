import numpy as np
import json
import os

class Config:
    """
    Configuration manager for simulation parameters
    """
    def __init__(self, config_file=None):
        # Default configuration
        self.config = {
            # Simulation parameters
            'simulation': {
                'particle_radius': 0.1,
                'time_step': 0.001,
                'gravity': [0.0, -9.81, 0.0],
                'bound_min': [-10.0, -10.0, -10.0],
                'bound_max': [10.0, 10.0, 10.0],
                'bound_damping': -0.5,
            },
            
            # Fluid parameters
            'fluid': {
                'name': 'Water',
                'density': 1000.0,
                'viscosity': 0.001,
                'surface_tension': 0.072,
                'gas_constant': 2000.0,
            },
            
            # Nozzle parameters
            'nozzle': {
                'type': 'cylindrical',
                'position': [0.0, 0.0, 0.0],
                'direction': [1.0, 0.0, 0.0],
                'inlet_diameter': 2.0,
                'outlet_diameter': 1.0,
                'length': 5.0,
                'flow_rate': 10.0,
                'inflow_velocity': 5.0,
                'throat_diameter': 0.5,  # For converging-diverging nozzle
                'throat_position': 0.5,  # For converging-diverging nozzle
            },
            
            # Visualization parameters
            'visualization': {
                'show_velocity': False,
                'show_density': True,
                'particle_scale': 100.0,
                'colormap': 'viridis',
                'update_freq': 5,
                'pause_time': 0.01,
            },
            
            # Export parameters
            'export': {
                'export_folder': 'output',
                'export_interval': 0.1,
            },
            
            # Simulation control
            'control': {
                'max_time': 10.0,
                'max_particles': 5000,
                'seed': 42,
            },
        }
        
        # Load from file if provided
        if config_file is not None and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                self._update_nested_dict(self.config, loaded_config)
    
    def _update_nested_dict(self, d, u):
        """
        Update nested dictionary d with values from u
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def get_simulation_config(self):
        return self.config['simulation']
    
    def get_fluid_config(self):
        return self.config['fluid']
    
    def get_nozzle_config(self):
        return self.config['nozzle']
    
    def get_visualization_config(self):
        return self.config['visualization']
    
    def get_export_config(self):
        return self.config['export']
    
    def get_control_config(self):
        return self.config['control']
    
    def save(self, filename):
        """
        Save configuration to a file
        """
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def update_from_args(self, args):
        """
        Update configuration from command-line arguments
        """
        # Fluid parameters
        if hasattr(args, 'fluid_viscosity') and args.fluid_viscosity is not None:
            self.config['fluid']['viscosity'] = args.fluid_viscosity
        if hasattr(args, 'fluid_density') and args.fluid_density is not None:
            self.config['fluid']['density'] = args.fluid_density
        
        # Nozzle parameters
        if hasattr(args, 'nozzle_type') and args.nozzle_type is not None:
            self.config['nozzle']['type'] = args.nozzle_type
        if hasattr(args, 'inlet_diameter') and args.inlet_diameter is not None:
            self.config['nozzle']['inlet_diameter'] = args.inlet_diameter
        if hasattr(args, 'outlet_diameter') and args.outlet_diameter is not None:
            self.config['nozzle']['outlet_diameter'] = args.outlet_diameter
        if hasattr(args, 'throat_diameter') and args.throat_diameter is not None:
            self.config['nozzle']['throat_diameter'] = args.throat_diameter
        if hasattr(args, 'flow_rate') and args.flow_rate is not None:
            self.config['nozzle']['flow_rate'] = args.flow_rate
        
        # Simulation control
        if hasattr(args, 'max_time') and args.max_time is not None:
            self.config['control']['max_time'] = args.max_time
        if hasattr(args, 'max_particles') and args.max_particles is not None:
            self.config['control']['max_particles'] = args.max_particles
        if hasattr(args, 'seed') and args.seed is not None:
            self.config['control']['seed'] = args.seed


class DataAnalyzer:
    """
    Analyze simulation data
    """
    def __init__(self):
        self.data = {}
    
    def add_snapshot(self, time, particles, densities, velocities):
        self.data[time] = {
            'particles': particles,
            'densities': densities,
            'velocities': velocities,
        }
    
    def get_flow_rate(self, x_position, time_window=0.1):
        """
        Calculate flow rate through a plane at x_position
        """
        flow_rates = []
        times = sorted(self.data.keys())
        
        for i in range(1, len(times)):
            t1, t2 = times[i-1], times[i]
            dt = t2 - t1
            
            if dt > time_window:
                continue
            
            # Count particles crossing the plane
            pos1 = self.data[t1]['particles']
            pos2 = self.data[t2]['particles']
            
            # Find particles that crossed from left to right
            crossings = 0
            for j in range(len(pos1)):
                if j < len(pos2):
                    if pos1[j, 0] < x_position and pos2[j, 0] >= x_position:
                        crossings += 1
            
            flow_rate = crossings / dt
            flow_rates.append(flow_rate)
        
        return np.mean(flow_rates) if flow_rates else 0.0
    
    def get_velocity_profile(self, x_position, radius, n_bins=10):
        """
        Get velocity profile at a specific x-position
        """
        # Combine all snapshots and find particles near the plane
        all_positions = []
        all_velocities = []
        
        for time in self.data:
            positions = self.data[time]['particles']
            velocities = self.data[time]['velocities']
            
            # Find particles near the plane
            mask = np.abs(positions[:, 0] - x_position) < 0.5
            all_positions.extend(positions[mask])
            all_velocities.extend(velocities[mask])
        
        all_positions = np.array(all_positions)
        all_velocities = np.array(all_velocities)
        
        if len(all_positions) == 0:
            return [], []
        
        # Calculate radial distance from axis
        radial_distances = np.sqrt(all_positions[:, 1]**2 + all_positions[:, 2]**2)
        axial_velocities = all_velocities[:, 0]
        
        # Create radial bins
        bin_edges = np.linspace(0, radius, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        # Calculate average velocity in each bin
        bin_velocities = []
        for i in range(n_bins):
            mask = (radial_distances >= bin_edges[i]) & (radial_distances < bin_edges[i+1])
            if np.any(mask):
                bin_velocities.append(np.mean(axial_velocities[mask]))
            else:
                bin_velocities.append(0.0)
        
        return bin_centers, np.array(bin_velocities)
