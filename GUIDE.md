# Fluid Nozzle Simulation Guide

This guide provides information on how to use the SPH-based fluid nozzle simulation framework to study fluid flow through various nozzle geometries.

## Core Components

The framework consists of several core modules:

1. **SPH Simulation Core** (`fluid_sim/sph_core.py`): Implements the Smoothed Particle Hydrodynamics algorithm
2. **Nozzle Geometries** (`fluid_sim/nozzle.py`): Defines nozzle shapes and handles particle emission
3. **Fluid Properties** (`fluid_sim/fluid.py`): Manages fluid properties like viscosity and density
4. **Visualization** (`fluid_sim/visualization.py`): Provides real-time visualization and data export
5. **Utilities** (`fluid_sim/utilities.py`): Configuration management and data analysis tools

## Running Simulations

### Basic Usage

The simplest way to run a simulation is using the main runner script:

```bash
python run_simulation.py
```

This will run with default parameters. You can customize the simulation using command-line arguments:

```bash
python run_simulation.py --nozzle_type converging-diverging --inlet_diameter 2.0 --outlet_diameter 1.0 --throat_diameter 0.5 --fluid_viscosity 0.01
```

### Using Configuration Files

You can also use JSON configuration files:

```bash
python examples/config_based_simulation.py --config config/nozzle_test.json
```

### Example Scripts

The framework includes several example scripts:

- `examples/cylindrical_nozzle.py`: Simple cylindrical nozzle simulation
- `examples/converging_diverging_nozzle.py`: De Laval (converging-diverging) nozzle simulation
- `examples/viscosity_comparison.py`: Compares flow behavior with different fluid viscosities
- `examples/config_based_simulation.py`: Demonstrates loading parameters from a config file

## Parameter Configuration

### Simulation Parameters

- `particle_radius`: Size of the SPH particles
- `time_step`: Time step for numerical integration
- `gravity`: Gravity vector [x, y, z]
- `bound_min`/`bound_max`: Simulation domain boundaries

### Fluid Properties

- `density`: Fluid density (kg/m³)
- `viscosity`: Fluid viscosity (Pa·s)
- `surface_tension`: Surface tension coefficient (N/m)
- `gas_constant`: Used in the equation of state for pressure calculation

### Nozzle Parameters

#### Common Parameters
- `position`: Starting position of the nozzle [x, y, z]
- `direction`: Direction vector of the nozzle [x, y, z]
- `inlet_diameter`: Diameter at the inlet (m)
- `outlet_diameter`: Diameter at the outlet (m)
- `length`: Length of the nozzle (m)
- `flow_rate`: Particle emission rate (particles/s)
- `inflow_velocity`: Initial velocity of particles (m/s)

#### Converging-Diverging Nozzle
- `throat_diameter`: Diameter at the narrowest point (m)
- `throat_position`: Position of the throat as a fraction of total length (0-1)

## Data Analysis

The framework provides tools for analyzing flow behavior:

### Flow Rate Calculation

Calculate the flow rate at any cross-section:

```python
flow_rate = analyzer.get_flow_rate(x_position)
```

### Velocity Profile

Get the velocity profile at a specific cross-section:

```python
radial_positions, velocities = analyzer.get_velocity_profile(x_position, radius)
```

## Visualization Options

- `show_velocity`: Show velocity vectors (boolean)
- `show_density`: Color particles by density (boolean)
- `particle_scale`: Visual size of particles
- `colormap`: Colormap for density visualization (e.g., 'viridis', 'coolwarm')

## Extending the Framework

### Adding New Nozzle Geometries

Create a new class that inherits from the `Nozzle` base class:

```python
class CustomNozzle(Nozzle):
    def __init__(self, config):
        super().__init__(config)
        # Add custom parameters
        
    def get_distance(self, point):
        # Implement signed distance function
        
    def emit_particles(self, current_time, particle_radius):
        # Implement particle emission
```

### Creating Custom Fluids

Use the provided factory methods or create custom fluids:

```python
# Built-in fluids
water = Fluid.water()
oil = Fluid.oil()
honey = Fluid.honey()

# Custom fluid
custom_fluid = Fluid.custom("Glycerin", density=1260.0, viscosity=1.5)
```

## Performance Considerations

- Particle count significantly affects performance
- The `particle_radius` parameter controls resolution and particle count
- Higher viscosity values generally improve numerical stability
- Use `numba` JIT compilation for performance-critical functions

## Output and Data Export

Simulation data can be exported to NumPy .npz files for further analysis:

```bash
python run_simulation.py --export_data
```

Files will be saved in the `output` directory by default.

## Troubleshooting

- **Simulation instability**: Try reducing the time step or increasing fluid viscosity
- **Particles clumping**: Check the smoothing length and rest density parameters
- **Low performance**: Reduce the number of particles or disable visualization for faster runs
- **Visualization issues**: Try different colormaps or adjust the particle scale

## References

- Monaghan, J.J. (1992). "Smoothed Particle Hydrodynamics"
- Müller, M., Charypar, D., & Gross, M. (2003). "Particle-based fluid simulation for interactive applications"
