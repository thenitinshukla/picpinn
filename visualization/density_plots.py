import torch
import numpy as np
import matplotlib.pyplot as plt
import os

@torch.no_grad()
def plot_density_at_t0(model, device, Lx, Ly, output_dir, Nx=100, Ny=70):
    """
    Plot the density distribution for each species at t=0.
    
    Args:
        model: The trained PlasmaNet model
        device: Computation device (CPU/GPU)
        Lx, Ly: System dimensions
        output_dir: Directory to save results
        Nx, Ny: Grid resolution
    """
    model.eval()
    
    # Set domain size for boundary conditions
    model.set_domain_size(Lx, Ly)
    
    # Create directory for density plots
    density_dir = f'{output_dir}/density_plots'
    os.makedirs(density_dir, exist_ok=True)
    
    # Generate spatial grid at t=0
    x = torch.linspace(0, Lx, Nx, device=device)
    y = torch.linspace(0, Ly, Ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)
    T_flat = torch.zeros_like(X_flat)  # t=0
    
    # Process in batches to avoid memory issues
    batch_size = 1024
    num_batches = (X_flat.shape[0] + batch_size - 1) // batch_size
    
    # Initialize arrays to store density values for each species
    density_values = [np.zeros(X_flat.shape[0]) for _ in range(model.num_species)]
    
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, X_flat.shape[0])
        
        x_batch = X_flat[start_idx:end_idx]
        y_batch = Y_flat[start_idx:end_idx]
        t_batch = T_flat[start_idx:end_idx]
        
        species_outputs, _ = model(x_batch, y_batch, t_batch)
        
        # Extract density for each species
        for s in range(model.num_species):
            n, _, _, _ = torch.split(species_outputs[s], 1, dim=1)
            density_values[s][start_idx:end_idx] = n.cpu().numpy().flatten()
    
    # Create a combined figure for all species
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot density for each species
    for s in range(model.num_species):
        # Reshape density values to 2D grid
        density_2d = density_values[s].reshape(Nx, Ny)
        
        # Plot in the combined figure
        im = axes[s].imshow(density_2d.T, extent=[0, Lx, 0, Ly], 
                           origin='lower', aspect='auto', cmap='viridis')
        axes[s].set_xlabel('x')
        axes[s].set_ylabel('y')
        axes[s].set_title(f'Density of {model.species[s].name} at t=0')
        plt.colorbar(im, ax=axes[s], label='n')
        
        # Also create individual high-resolution plots
        fig_single, ax_single = plt.subplots(figsize=(10, 8))
        im_single = ax_single.imshow(density_2d.T, extent=[0, Lx, 0, Ly], 
                                    origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(im_single, ax=ax_single, label='n')
        ax_single.set_xlabel('x')
        ax_single.set_ylabel('y')
        ax_single.set_title(f'Density of {model.species[s].name} at t=0')
        
        # Add contour lines
        contour_levels = np.linspace(density_2d.min(), density_2d.max(), 10)
        ax_single.contour(density_2d.T, levels=contour_levels, colors='k', alpha=0.3, 
                         extent=[0, Lx, 0, Ly])
        
        plt.tight_layout()
        plt.savefig(f'{density_dir}/density_{model.species[s].name}_t0.png', dpi=300)
        plt.close(fig_single)
    
    # Save the combined figure
    plt.tight_layout()
    plt.savefig(f'{output_dir}/density_all_species_t0.png', dpi=300)
    plt.close(fig)
    
    # Create a total density plot (sum of all species)
    total_density = np.zeros_like(density_values[0])
    for s in range(model.num_species):
        total_density += density_values[s]
    
    total_density_2d = total_density.reshape(Nx, Ny)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(total_density_2d.T, extent=[0, Lx, 0, Ly], 
                  origin='lower', aspect='auto', cmap='plasma')
    plt.colorbar(im, ax=ax, label='Total Density')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Total Plasma Density at t=0')
    
    # Add contour lines
    contour_levels = np.linspace(total_density_2d.min(), total_density_2d.max(), 10)
    ax.contour(total_density_2d.T, levels=contour_levels, colors='k', alpha=0.3, 
              extent=[0, Lx, 0, Ly])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/total_density_t0.png', dpi=300)
    plt.close()
    
    print(f"Density plots at t=0 saved to '{output_dir}'")