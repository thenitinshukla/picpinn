import torch
import torch.nn as nn

class PlasmaNet(nn.Module):
    """
    Neural network model for plasma simulation with boundary conditions.
    """
    def __init__(self, species_list, hidden_layers=3, neurons_per_layer=128):
        super(PlasmaNet, self).__init__()
        self.species = species_list
        self.num_species = len(species_list)

        # Input: x, y, t
        # Output: For each species: n, ux, uy, uz + fields: Ex, Ey, Ez, Bx, By, Bz
        self.output_per_species = 4  # n, ux, uy, uz
        self.field_components = 6    # Ex, Ey, Ez, Bx, By, Bz
        self.output_size = self.num_species * self.output_per_species + self.field_components

        # Create network layers
        self.input_layer = nn.Linear(3, neurons_per_layer)
        
        # Create hidden layers with skip connections
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(neurons_per_layer, neurons_per_layer),
                nn.Tanh(),
                nn.Linear(neurons_per_layer, neurons_per_layer)
            ))
        
        # Output layer
        self.output_layer = nn.Linear(neurons_per_layer, self.output_size)
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
        # System dimensions (will be set during training)
        self.Lx = None
        self.Ly = None

    def _initialize_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_domain_size(self, Lx, Ly):
        """Set the domain size for boundary conditions."""
        self.Lx = Lx
        self.Ly = Ly

    def apply_boundary_conditions(self, x, y, t, outputs):
        """
        Apply boundary conditions:
        - Absorbing in x-direction (flow direction)
        - Periodic in y-direction (transverse)
        - Zero fields at t=0
        
        Args:
            x, y, t: Spatial and temporal coordinates
            outputs: Raw network outputs
            
        Returns:
            Modified outputs with boundary conditions applied
        """
        if self.Lx is None or self.Ly is None:
            return outputs  # Cannot apply boundary conditions without domain size
            
        species_outputs, fields = outputs
        
        # Apply periodic boundary conditions in y-direction
        # Map y to [0, Ly] for periodic boundary
        y_periodic = y % self.Ly
        
        # Apply absorbing boundary conditions in x-direction
        # Use a smooth transition function near boundaries
        boundary_width = 0.1 * self.Lx
        left_boundary = torch.sigmoid((x - boundary_width) / (0.01 * self.Lx))
        right_boundary = torch.sigmoid((self.Lx - boundary_width - x) / (0.01 * self.Lx))
        boundary_factor = left_boundary * right_boundary
        
        # Apply to fields (damping near boundaries)
        Ex, Ey, Ez, Bx, By, Bz = torch.split(fields, 1, dim=1)
        
        # Apply zero field condition at t=0
        # Use a stronger, more explicit approach to ensure fields are zero at t=0
        # This is a key fix for the issue mentioned
        t_factor = torch.where(t < 1e-6, 
                              torch.zeros_like(t), 
                              1.0 - torch.exp(-10.0 * t))  # Sharper transition
        t_factor = t_factor.unsqueeze(1)  # Ensure correct broadcasting
        
        # Apply both boundary and time factors
        Ex = Ex * boundary_factor * t_factor
        Ey = Ey * boundary_factor * t_factor
        Ez = Ez * boundary_factor * t_factor
        Bx = Bx * boundary_factor * t_factor
        By = By * boundary_factor * t_factor
        Bz = Bz * boundary_factor * t_factor
        
        # Recombine fields
        fields_modified = torch.cat([Ex, Ey, Ez, Bx, By, Bz], dim=1)
        
        # Apply to species outputs
        species_outputs_modified = []
        for species_output in species_outputs:
            n, vx, vy, vz = torch.split(species_output, 1, dim=1)
            # Apply boundary conditions
            n = n * boundary_factor
            species_outputs_modified.append(torch.cat([n, vx, vy, vz], dim=1))
            
        return species_outputs_modified, fields_modified

    def forward(self, x, y, t):
        """Forward pass through the network with boundary conditions."""
        # Ensure inputs are 2D tensors
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Combine inputs
        inputs = torch.cat([x, y, t], dim=1)
        
        # Forward pass with skip connections
        h = torch.tanh(self.input_layer(inputs))
        
        for layer in self.hidden_layers:
            h_new = layer(h)
            h = h + h_new  # Skip connection
            h = torch.tanh(h)  # Non-linearity after skip connection
            
        outputs_raw = self.output_layer(h)
        
        # Split outputs into species data and fields
        species_outputs = []
        offset = 0
        for _ in range(self.num_species):
            species_outputs.append(outputs_raw[:, offset:offset+self.output_per_species])
            offset += self.output_per_species
        
        fields = outputs_raw[:, offset:offset+self.field_components]
        
        # Apply boundary conditions
        return self.apply_boundary_conditions(x, y, t, (species_outputs, fields))