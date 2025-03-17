import torch
import numpy as np
from scipy.optimize import curve_fit
from utils.fitting import exponential_fit

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
        Tuple containing time values and field/velocity histories
    """
    model.eval()
    
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
    
    # Evaluate at each time step to save memory
    for i, t_val in enumerate(t):
        # Create 2D spatial grid at this time
        X, Y = torch.meshgrid(x, y, indexing='ij')
        X_flat = X.reshape(-1, 1)
        Y_flat = Y.reshape(-1, 1)
        T_flat = torch.full_like(X_flat, t_val)
        
        # Process in batches to avoid memory issues
        batch_size = 1024
        num_batches = (X_flat.shape[0] + batch_size - 1) // batch_size
        
        bx_sum = 0.0
        by_sum = 0.0
        bz_sum = 0.0
        vx_sum = np.zeros(model.num_species)
        vy_sum = np.zeros(model.num_species)
        
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, X_flat.shape[0])
            
            x_batch = X_flat[start_idx:end_idx]
            y_batch = Y_flat[start_idx:end_idx]
            t_batch = T_flat[start_idx:end_idx]
            
            species_outputs, fields = model(x_batch, y_batch, t_batch)
            
            # Extract B fields (assuming fields contains [Ex, Ey, Ez, Bx, By, Bz])
            Ex, Ey, Ez, Bx, By, Bz = torch.split(fields, 1, dim=1)
            bx_sum += torch.sum(torch.abs(Bx)).item()
            by_sum += torch.sum(torch.abs(By)).item()
            bz_sum += torch.sum(torch.abs(Bz)).item()
            
            # Extract velocities for each species
            for s in range(model.num_species):
                n, vx, vy, vz = torch.split(species_outputs[s], 1, dim=1)
                vx_sum[s] += torch.sum(vx).item()
                vy_sum[s] += torch.sum(vy).item()
        
        # Compute averages
        num_points = X_flat.shape[0]
        bx_avg_history[i] = bx_sum / num_points
        by_avg_history[i] = by_sum / num_points
        bz_avg_history[i] = bz_sum / num_points
        
        for s in range(model.num_species):
            vx_avg_history[s, i] = vx_sum[s] / num_points
            vy_avg_history[s, i] = vy_sum[s] / num_points
    
    return t.cpu().numpy(), bx_avg_history, by_avg_history, bz_avg_history, vx_avg_history, vy_avg_history

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
        # Create 2D spatial grid at this time
        X, Y = torch.meshgrid(x, y, indexing='ij')
        X_flat = X.reshape(-1, 1)
        Y_flat = Y.reshape(-1, 1)
        T_flat = torch.full_like(X_flat, t_val)
        
        # Process in batches to avoid memory issues
        batch_size = 1024
        num_batches = (X_flat.shape[0] + batch_size - 1) // batch_size
        
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
        
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, X_flat.shape[0])
            
            x_batch = X_flat[start_idx:end_idx]
            y_batch = Y_flat[start_idx:end_idx]
            t_batch = T_flat[start_idx:end_idx]
            
            species_outputs, fields = model(x_batch, y_batch, t_batch)
            
            # Extract field components
            Ex, Ey, Ez, Bx, By, Bz = torch.split(fields, 1, dim=1)
            
            # Calculate electromagnetic energy densities (E²/2 and B²/2)
            ex_sum += torch.sum(Ex**2).item()
            ey_sum += torch.sum(Ey**2).item()
            ez_sum += torch.sum(Ez**2).item()
            bx_sum += torch.sum(Bx**2).item()
            by_sum += torch.sum(By**2).item()
            bz_sum += torch.sum(Bz**2).item()
            
            # Calculate kinetic energy for each species
            for s in range(model.num_species):
                n, vx, vy, vz = torch.split(species_outputs[s], 1, dim=1)
                
                # Kinetic energy density = n * m * v²/2 (normalized to m_e*c²)
                # For electrons, mass = 1, for ions use their mass
                mass = model.species[s].mass
                
                # Longitudinal KE (in flow direction)
                k_long_sum[s] += torch.sum(n * mass * vx**2 / 2).item()
                
                # Transverse KE
                k_trans_y_sum[s] += torch.sum(n * mass * vy**2 / 2).item()
                k_trans_z_sum[s] += torch.sum(n * mass * vz**2 / 2).item()
        
        # Compute averages
        num_points = X_flat.shape[0]
        ex_energy[i] = ex_sum / (2 * num_points)  # E²/2
        ey_energy[i] = ey_sum / (2 * num_points)
        ez_energy[i] = ez_sum / (2 * num_points)
        bx_energy[i] = bx_sum / (2 * num_points)  # B²/2
        by_energy[i] = by_sum / (2 * num_points)
        bz_energy[i] = bz_sum / (2 * num_points)
        
        for s in range(model.num_species):
            k_long[s, i] = k_long_sum[s] / num_points
            k_trans_y[s, i] = k_trans_y_sum[s] / num_points
            k_trans_z[s, i] = k_trans_z_sum[s] / num_points
    
    return t.cpu().numpy(), ex_energy, ey_energy, ez_energy, bx_energy, by_energy, bz_energy, k_long, k_trans_y, k_trans_z


