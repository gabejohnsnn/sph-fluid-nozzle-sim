# Fluid Nozzle Simulation

A Python-based fluid simulation engine for testing flow through nozzles of varying sizes with different fluid properties using Smoothed Particle Hydrodynamics (SPH).

## Features

- Smoothed Particle Hydrodynamics (SPH) based fluid simulation
- Multiple nozzle geometries (cylindrical and converging-diverging)
- Support for fluids with different viscosities and densities
- Enhanced 2D cross-section visualization with:
  - Flow streamlines
  - Velocity and density color mapping
  - Automatic particle sizing
  - Velocity indicators
  - Clear nozzle geometry representation
- Data export and analysis tools

## Installation

Clone the repository and install the package:

```bash
# Clone the repository
git clone https://github.com/gabejohnsnn/sph-fluid-nozzle-sim.git
cd sph-fluid-nozzle-sim

# Install the package in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

If you choose to only install dependencies without installing the package, you'll need to run scripts from the project root directory.

## Quick Start

Try the enhanced 2D visualization for a clearer view of fluid flow:

```bash
python examples/cross_section_visualization.py
```

Compare how different fluids flow through the same nozzle:

```bash
python examples/viscosity_flow_comparison.py
```

Or run a standard simulation:

```bash
python run_simulation.py
```

## Multiple Ways to Run Examples

We've provided several ways to run the examples:

1. **Install as a package** (recommended):
   ```bash
   pip install -e .
   python examples/cross_section_visualization.py
   ```

2. **Use the helper script**:
   ```bash
   python run_example.py examples/cross_section_visualization.py
   ```

3. **Run from project root**:
   ```bash
   # Make sure you're in the project root directory
   cd sph-fluid-nozzle-sim
   python examples/cross_section_visualization.py
   ```

The example scripts now include path fixing code, so they should work regardless of how you run them.

## Advanced Usage

You can customize the simulation by modifying the parameters in the config files or by passing command-line arguments:

```bash
# Full simulation customization
python run_simulation.py --nozzle_type converging-diverging --inlet_diameter 2.0 --outlet_diameter 1.0 --throat_diameter 0.5 --fluid_viscosity 0.01
```

For the enhanced 2D cross-section visualization:

```bash
# Color particles by velocity with streamlines
python examples/cross_section_visualization.py --nozzle_type converging-diverging --particle_size 3.0 --fluid oil --colormap coolwarm --color_by velocity

# Color particles by density without streamlines
python examples/cross_section_visualization.py --color_by density --streamlines False

# Show velocity vectors
python examples/cross_section_visualization.py --velocity_vectors
```

## Examples

Check the `examples` directory for sample simulations:

```bash
python examples/cylindrical_nozzle.py                # Simple cylindrical nozzle
python examples/converging_diverging_nozzle.py       # De Laval nozzle
python examples/viscosity_comparison.py              # Compare different fluid viscosities
python examples/config_based_simulation.py           # Load parameters from config file
python examples/cross_section_visualization.py       # Enhanced 2D visualization
python examples/viscosity_flow_comparison.py         # Side-by-side viscosity comparison
```

## Troubleshooting

### ModuleNotFoundError: No module named 'fluid_sim'

If you get this error, it means Python can't find the `fluid_sim` module. Fix it by:

1. Make sure you're running the scripts from the project root directory, OR
2. Install the package in development mode with `pip install -e .`, OR
3. Use the included `run_example.py` helper script

## Documentation

See [GUIDE.md](GUIDE.md) for comprehensive documentation on using and extending the simulation framework.

## License

MIT
