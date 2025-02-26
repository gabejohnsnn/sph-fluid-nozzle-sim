# Fluid Nozzle Simulation

A Python-based fluid simulation engine for testing flow through nozzles of varying sizes with different fluid properties using Smoothed Particle Hydrodynamics (SPH).

## Features

- Smoothed Particle Hydrodynamics (SPH) based fluid simulation
- Multiple nozzle geometries (cylindrical and converging-diverging)
- Support for fluids with different viscosities and densities
- Both 3D and 2D cross-section visualizations
- Data export and analysis tools

## Installation

```bash
git clone https://github.com/gabejohnsnn/sph-fluid-nozzle-sim.git
cd sph-fluid-nozzle-sim
pip install -r requirements.txt
```

## Quick Start

Try the 2D cross-section visualization for a clearer view of fluid flow:

```bash
python examples/cross_section_visualization.py
```

Or run a standard simulation:

```bash
python run_simulation.py
```

## Advanced Usage

You can customize the simulation by modifying the parameters in the config files or by passing command-line arguments:

```bash
python run_simulation.py --nozzle_type converging-diverging --inlet_diameter 2.0 --outlet_diameter 1.0 --throat_diameter 0.5 --fluid_viscosity 0.01
```

For the 2D cross-section visualization:

```bash
python examples/cross_section_visualization.py --nozzle_type converging-diverging --particle_size 3.0 --fluid oil --colormap coolwarm
```

## Examples

Check the `examples` directory for sample simulations:

```bash
python examples/cylindrical_nozzle.py                # Simple cylindrical nozzle
python examples/converging_diverging_nozzle.py       # De Laval nozzle
python examples/viscosity_comparison.py              # Compare different fluid viscosities
python examples/config_based_simulation.py           # Load parameters from config file
python examples/cross_section_visualization.py       # 2D cross-section visualization
```

## Documentation

See [GUIDE.md](GUIDE.md) for comprehensive documentation on using and extending the simulation framework.

## License

MIT
