import torch
import torch.nn as nn

class PlasmaNet(nn.Module):
    """
    Neural network model for plasma simulation.
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

    def _initialize_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        """Forward pass through the network."""
        # Combine inputs
        inputs = torch.cat([x, y, t], dim=1)
        
        # Forward pass with skip connections
        h = torch.tanh(self.input_layer(inputs))
        
        for layer in self.hidden_layers:
            h_new = layer(h)
            h = h + h_new  # Skip connection
            h = torch.tanh(h)  # Non-linearity after skip connection
            
        outputs = self.output_layer(h)
        
        # Split outputs into species data and fields
        species_outputs = []
        offset = 0
        for _ in range(self.num_species):
            species_outputs.append(outputs[:, offset:offset+self.output_per_species])
            offset += self.output_per_species
            
        fields = outputs[:, offset:offset+self.field_components]
        
        return species_outputs, fields


