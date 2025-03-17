class Species:
    """
    Class representing a plasma species with physical parameters.
    """
    def __init__(self, name, mass, charge, density, vx0, vy0=0.0, vz0=0.0, vth_x=0.01, vth_y=0.01):
        self.name = name          # Species name
        self.mass = mass          # Normalized to electron mass
        self.charge = charge      # Normalized to elementary charge
        self.density = density    # Normalized density
        self.vx0 = vx0            # Initial bulk velocity x (c)
        self.vy0 = vy0            # Initial bulk velocity y (c)
        self.vz0 = vz0            # Initial bulk velocity z (c)
        self.vth_x = vth_x        # Thermal velocity x
        self.vth_y = vth_y        # Thermal velocity y
        self.qm_ratio = charge/mass  # q/m ratio
