class Fluid:
    """
    Class to define fluid properties
    """
    def __init__(self, config):
        self.name = config.get('name', 'DefaultFluid')
        self.density = config.get('density', 1000.0)  # kg/m^3
        self.viscosity = config.get('viscosity', 0.001)  # Pa.s
        self.surface_tension = config.get('surface_tension', 0.072)  # N/m
        self.gas_constant = config.get('gas_constant', 2000.0)
    
    def get_simulation_parameters(self):
        """
        Get simulation parameters based on fluid properties
        """
        return {
            'rest_density': self.density,
            'viscosity': self.viscosity,
            'gas_constant': self.gas_constant,
        }
    
    @classmethod
    def water(cls):
        """Factory method for water"""
        return cls({
            'name': 'Water',
            'density': 1000.0,
            'viscosity': 0.001,
            'surface_tension': 0.072,
            'gas_constant': 2000.0
        })
    
    @classmethod
    def oil(cls):
        """Factory method for oil"""
        return cls({
            'name': 'Oil',
            'density': 900.0,
            'viscosity': 0.03,
            'surface_tension': 0.03,
            'gas_constant': 2000.0
        })
    
    @classmethod
    def honey(cls):
        """Factory method for honey"""
        return cls({
            'name': 'Honey',
            'density': 1400.0,
            'viscosity': 10.0,
            'surface_tension': 0.07,
            'gas_constant': 2000.0
        })
    
    @classmethod
    def custom(cls, name, density, viscosity, surface_tension=0.072, gas_constant=2000.0):
        """Factory method for custom fluid"""
        return cls({
            'name': name,
            'density': density,
            'viscosity': viscosity,
            'surface_tension': surface_tension,
            'gas_constant': gas_constant
        })
