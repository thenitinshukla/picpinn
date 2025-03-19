import torch
import numpy as np
from scipy.optimize import curve_fit
from utils.fitting import exponential_fit
from utils.filtering import filter_fields

@torch.no_grad()
def evaluate_model(model, device, Lx, Ly, Tmax, Nx=32, Ny=32, Nt=100):
    """
    Evaluate the model on a grid and return results for analysis.
    
    Args:
        model: The trained PlasmaNet model
        device: Computation device (CPU/GPU)
        Lx, Ly: System dimensions
        Tmax: Maximum simulation time
        Nx, Ny, Nt: Grid resolution
        
    Returns:
        Tuple containing time values, field/velocity histories, and full Bz field data
    """
    model.eval()
    
    # Set domain size for boundary conditions
    model.set_domain_size(Lx, Ly)
    
    # Generate evaluation grid
    x = torch.linspace(0, Lx, Nx, device=device)
    y = torch.linspace(0, Ly, Ny, device=device)
    t = torch.linspace(0, Tmax, Nt, device=device)
    
    # Store results for each time step
    bx_avg_history = np.zeros(Nt)
    by_avg_history = np.zeros(Nt)
    bz_avg_history = np.zeros(Nt)
    vx_avg_history = np.zeros((model.num_species, Nt))
    vy_avg_history = np.zeros((model.num_species, Nt))
    
    # Store full Bz field data for each time step
    bz_field_history = np.zeros((Nt, Ny, Nx))
    
    # Evaluate at each time step to save memory
    for i, t_val in enumerate(t):
        # Create 2D spatial grid at this time
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Initialize arrays to store field values
        Ex_2d = np.zeros((Nx, Ny))
        Ey_2d = np.zeros((Nx, Ny))
        Ez_2d = np.zeros((Nx, Ny))
        Bx_2d = np.zeros((Nx, Ny))
        By_2d = np.zeros((Nx, Ny))
        Bz_2d = np.zeros((Nx, Ny))
        
        # Initialize sums for averages
        bx_sum = 0.0
        by_sum = 0.0
        bz_sum = 0.0
        vx_sum = np.zeros(model.num_species)
        vy_sum = np.zeros(model.num_species)
        
        # Total number of points
        total_points = 0
        
        # Evaluate model at each grid point
        for ix in range(Nx):
            for iy in range(Ny):
                # Create input tensors
                x_val = x[ix].view(1, 1)
                y_val = y[iy].view(1, 1)
                t_val_tensor = torch.tensor([[t_val]], device=device)
                
                # Evaluate model
                species_outputs, fields = model(x_val, y_val, t_val_tensor)
                
                # Extract field components
                Ex, Ey, Ez, Bx, By, Bz = torch.chunk(fields, 6, dim=1)
                
                # Update sums for averages
                bx_sum += torch.abs(Bx).item()
                by_sum += torch.abs(By).item()
                bz_sum += torch.abs(Bz).item()
                
                # Store field values
                Ex_2d[ix, iy] = Ex.item()
                Ey_2d[ix, iy] = Ey.item()
                Ez_2d[ix, iy] = Ez.item()
                Bx_2d[ix, iy] = Bx.item()
                By_2d[ix, iy] = By.item()
                Bz_2d[ix, iy] = Bz.item()
                
                # Extract velocities for each species
                for s in range(model.num_species):
                    n, vx, vy, vz = torch.split(species_outputs[s], 1, dim=1)
                    vx_sum[s] += vx.item()
                    vy_sum[s] += vy.item()
                
                total_points += 1
        
        # Convert to torch tensors for filtering
        Ex_tensor = torch.tensor(Ex_2d, device='cpu')
        Ey_tensor = torch.tensor(Ey_2d, device='cpu')
        Ez_tensor = torch.tensor(Ez_2d, device='cpu')
        Bx_tensor = torch.tensor(Bx_2d, device='cpu')
        By_tensor = torch.tensor(By_2d, device='cpu')
        Bz_tensor = torch.tensor(Bz_2d, device='cpu')
        
        # Apply filtering
        Ex_filtered, Ey_filtered, Ez_filtered, Bx_filtered, By_filtered, Bz_filtered = filter_fields(
            Ex_tensor, Ey_tensor, Ez_tensor, Bx_tensor, By_tensor, Bz_tensor, num_passes=5
        )
        
        # Convert back to numpy for storage
        Bz_filtered = Bz_filtered.numpy()
        
        # Compute averages
        bx_avg_history[i] = bx_sum / total_points
        by_avg_history[i] = by_sum / total_points
        bz_avg_history[i] = bz_sum / total_points
        
        for s in range(model.num_species):
            vx_avg_history[s, i] = vx_sum[s] / total_points
            vy_avg_history[s, i] = vy_sum[s] / total_points
        
        # Store the filtered Bz field data
        bz_field_history[i] = Bz_filtered.T
    
    return t.cpu().numpy(), bx_avg_history, by_avg_history, bz_avg_history, vx_avg_history, vy_avg_history, bz_field_history

@torch.no_grad()
def calculate_energy_components(model, device, Lx, Ly, Tmax, Nx=64, Ny=64, Nt=200):
    """
    Calculate electromagnetic energy and kinetic energy components over time.
    
    Args:
        model: The trained PlasmaNet model
        device: Computation device (CPU/GPU)
        Lx, Ly: System dimensions
        Tmax: Maximum simulation time
        Nx, Ny, Nt: Grid resolution
        
    Returns:
        Tuple containing time values and energy components
    """
    model.eval()
    
    # Set domain size for boundary conditions
    model.set_domain_size(Lx, Ly)
    
    # Generate evaluation grid
    x = torch.linspace(0, Lx, Nx, device=device)
    y = torch.linspace(0, Ly, Ny, device=device)
    t = torch.linspace(0, Tmax, Nt, device=device)
    
    # Store energy components over time
    ex_energy = np.zeros(Nt)  # E_x energy
    ey_energy = np.zeros(Nt)  # E_y energy
    ez_energy = np.zeros(Nt)  # E_z energy
    bx_energy = np.zeros(Nt)  # B_x energy
    by_energy = np.zeros(Nt)  # B_y energy
    bz_energy = np.zeros(Nt)  # B_z energy
    
    # Kinetic energy components for each species
    k_long = np.zeros((model.num_species, Nt))  # Longitudinal KE (vx direction)
    k_trans_y = np.zeros((model.num_species, Nt))  # Transverse KE (vy direction)
    k_trans_z = np.zeros((model.num_species, Nt))  # Transverse KE (vz direction)
    
    # Evaluate at each time step
    for i, t_val in enumerate(t):
        # Initialize energy sums
        ex_sum = 0.0
        ey_sum = 0.0
        ez_sum = 0.0
        bx_sum = 0.0
        by_sum = 0.0
        bz_sum = 0.0
        
        # Initialize kinetic energy sums
        k_long_sum = np.zeros(model.num_species)
        k_trans_y_sum = np.zeros(model.num_species)
        k_trans_z_sum = np.zeros(model.num_species)
        
        # Total number of points
        total_points = 0
        
        # Evaluate model at each grid point
        for ix in range(Nx):
            for iy in range(Ny):
                # Create input tensors
                x_val = x[ix].view(1, 1)
                y_val = y[iy].view(1, 1)
                t_val_tensor = torch.tensor([[t_val]], device=device)
                
                # Evaluate model
                species_outputs, fields = model(x_val, y_val, t_val_tensor)
                
                # Extract field components
                Ex, Ey, Ez, Bx, By, Bz = torch.chunk(fields, 6, dim=1)
                
                # Calculate electromagnetic energy densities (E²/2 and B²/2)
                ex_sum += (Ex**2).item()
                ey_sum += (Ey**2).item()
                ez_sum += (Ez**2).item()
                bx_sum += (Bx**2).item()
                by_sum += (By**2).item()
                bz_sum += (Bz**2).item()
                
                # Calculate kinetic energy for each species
                for s in range(model.num_species):
                    n, vx, vy, vz = torch.split(species_outputs[s], 1, dim=1)
                    
                    # Kinetic energy density = n * m * v²/2 (normalized to m_e*c²)
                    # For electrons, mass = 1, for ions use their mass
                    mass = model.species[s].mass
                    
                    # Longitudinal KE (in flow direction)
                    k_long_sum[s] += (n * mass * vx**2 / 2).item()
                    
                    # Transverse KE
                    k_trans_y_sum[s] += (n * mass * vy**2 / 2).item()
                    k_trans_z_sum[s] += (n * mass * vz**2 / 2).item()
                
                total_points += 1
        
        # Compute averages
        ex_energy[i] = ex_sum / (2 * total_points)  # E²/2
        ey_energy[i] = ey_sum / (2 * total_points)
        ez_energy[i] = ez_sum / (2 * total_points)
        bx_energy[i] = bx_sum / (2 * total_points)  # B²/2
        by_energy[i] = by_sum / (2 * total_points)
        bz_energy[i] = bz_sum / (2 * total_points)
        
        for s in range(model.num_species):
            k_long[s, i] = k_long_sum[s] / total_points
            k_trans_y[s, i] = k_trans_y_sum[s] / total_points
            k_trans_z[s, i] = k_trans_z_sum[s] / total_points
    
    return t.cpu().numpy(), ex_energy, ey_energy, ez_energy, bx_energy, by_energy, bz_energy, k_long, k_trans_y, k_trans_z