import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import os
from functools import partial
from datetime import datetime

# Create output directory for results
output_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

class Species:
    def __init__(self, name, mass, charge, density, vx0, vy0=0.0, vz0=0.0, vth_x=0.01, vth_y=0.01):
        self.name = name
        self.mass = mass          # Normalized to electron mass
        self.charge = charge      # Normalized to elementary charge
        self.density = density    # Normalized density
        self.vx0 = vx0            # Initial bulk velocity x (c)
        self.vy0 = vy0            # Initial bulk velocity y (c)
        self.vz0 = vz0            # Initial bulk velocity z (c)
        self.vth_x = vth_x        # Thermal velocity x
        self.vth_y = vth_y        # Thermal velocity y
        self.qm_ratio = charge/mass  # q/m ratio


class PlasmaNet(nn.Module):
    def __init__(self, species_list, hidden_layers=3, neurons_per_layer=128):
        super(PlasmaNet, self).__init__()
        self.species = species_list
        self.num_species = len(species_list)

        # Input: x, y, t
        # Output: For each species: n, ux, uy, uz + fields: Ex, Ey, Ez, Bx, By, Bz
        self.output_per_species = 4  # n, ux, uy, uz
        self.output_size = self.num_species*self.output_per_species + 6

        # Create network layers
        layers = []
        layers.append(nn.Linear(3, neurons_per_layer))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers-1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ))

         # Initialize weights for better convergence
        layers.append(nn.Linear(neurons_per_layer, self.output_size))
        
        self.network = nn.Sequential(*layers)

     def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
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
    
    def exponential_fit(t, a, b, c):
        """Exponential function for fitting growth rate: a * exp(b * t) + c"""
        return a * np.exp(b * t) + c


@torch.no_grad()
def evaluate_model(model, device, Lx, Ly, Tmax, Nx=32, Ny=32, Nt=100):
    """Evaluate the model on a grid and return results for analysis."""
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
            
            # Extract B fields
            _, _, _, Bx, By, Bz = fields.split(1, dim=1)
            bx_sum += torch.sum(torch.abs(Bx)).item()
            by_sum += torch.sum(torch.abs(By)).item()
            bz_sum += torch.sum(torch.abs(Bz)).item()
            
            # Extract velocities for each species
            for s in range(model.num_species):
                _, vx, vy, _ = species_outputs[s].split(1, dim=1)
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
    """Calculate electromagnetic energy and kinetic energy components over time."""
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
            Ex, Ey, Ez, Bx, By, Bz = fields.split(1, dim=1)
            
            # Calculate electromagnetic energy densities (E²/2 and B²/2)
            ex_sum += torch.sum(Ex**2).item()
            ey_sum += torch.sum(Ey**2).item()
            ez_sum += torch.sum(Ez**2).item()
            bx_sum += torch.sum(Bx**2).item()
            by_sum += torch.sum(By**2).item()
            bz_sum += torch.sum(Bz**2).item()
            
            # Calculate kinetic energy for each species
            for s in range(model.num_species):
                n, vx, vy, vz = species_outputs[s].split(1, dim=1)
                
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

def plot_energy_evolution(model, device, Lx, Ly, Tmax, suffix=""):
    """Generate plots of electromagnetic energy and kinetic energy evolution."""
    print("Calculating energy components...")
    t_vals, ex_energy, ey_energy, ez_energy, bx_energy, by_energy, bz_energy, k_long, k_trans_y, k_trans_z = calculate_energy_components(
        model, device, Lx, Ly, Tmax, Nx=32, Ny=32, Nt=200
    )
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # Panel (a): Electromagnetic energy evolution (log scale)
    ax1.semilogy(t_vals, ex_energy, 'b-', linewidth=2, label=r'$\alpha_{E_1}$')
    ax1.semilogy(t_vals, ey_energy, 'g-', linewidth=2, label=r'$\alpha_{E_2}$')
    ax1.semilogy(t_vals, bz_energy, 'r-', linewidth=2, label=r'$\alpha_{B_3}$')
    
    # Try to fit exponential growth to early phase of Bz
    growth_end_idx = len(t_vals) // 4  # Use first quarter for growth phase
    try:
        # Add small offset to avoid log(0)
        bz_fit_data = bz_energy[:growth_end_idx] + 1e-10
        t_fit = t_vals[:growth_end_idx]
        
        # Initial guess for parameters
        p0 = [min(bz_fit_data), 0.1, 0]
        
        # Fit exponential growth
        popt, _ = curve_fit(exponential_growth, t_fit, bz_fit_data, p0=p0)
        A, gamma, C = popt
        
        # Plot theoretical growth rate
        ax1.semilogy(t_fit, exponential_growth(t_fit, A, gamma, C), 'r:', linewidth=1.5)
        
        # Add growth rate annotation
        ax1.annotate(f'$\\gamma = {gamma:.4f}\\omega_p$', 
                    xy=(t_fit[-1], exponential_growth(t_fit[-1], A, gamma, C)),
                    xytext=(t_fit[-1]+5, exponential_growth(t_fit[-1], A, gamma, C)*2),
                    arrowprops=dict(arrowstyle='->'))
    except Exception as e:
        print(f"Warning: Could not fit exponential growth curve: {e}")
    
    ax1.set_ylabel(r'$\alpha_{E,B}$', fontsize=12)
    ax1.set_title('(a) Electromagnetic Energy Evolution', fontsize=14)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='best', fontsize=10)
    ax1.set_ylim(1e-8, 1e-1)
    
    # Panel (b): Kinetic energy evolution
    # Focus on electron species (typically first two species in the list)
    electron_indices = [i for i, species in enumerate(model.species) if species.mass == 1.0]
    
    if electron_indices:
        # Combine longitudinal KE for electron species
        k_long_electrons = np.sum(k_long[electron_indices], axis=0)
        
        # Combine transverse KE for electron species (y and z components)
        k_trans_electrons = np.sum(k_trans_y[electron_indices] + k_trans_z[electron_indices], axis=0)
        
        ax2.plot(t_vals, k_long_electrons, 'r-', linewidth=2, label=r'$K_1$ (Longitudinal KE)')
        ax2.plot(t_vals, k_trans_electrons, 'b-', linewidth=2, label=r'$K_2=K_3$ (Transverse KE)')
        
        ax2.set_ylabel(r'Kinetic energy density [$m_e n_0 c^2$]', fontsize=12)
        ax2.set_title('(b) Electron Kinetic Energy Evolution', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
    
    ax2.set_xlabel(r'$t$ [$\omega_p^{-1}$]', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'results/energy_evolution{suffix}.png', dpi=300)
    
    # Save data for further analysis
    np.savez(f'results/energy_data{suffix}.npz', 
             t=t_vals, ex=ex_energy, ey=ey_energy, ez=ez_energy,
             bx=bx_energy, by=by_energy, bz=bz_energy,
             k_long=k_long, k_trans_y=k_trans_y, k_trans_z=k_trans_z)
    
    print(f"Energy evolution plots saved to 'results/energy_evolution{suffix}.png'")
    plt.close()

    def run_simulation(species_list, Lx=40.0, Ly=40.0, Tmax=50.0, Nx=64, Ny=64, Nt=100, 
                   epochs=5000, batch_size=10000, lr=1e-3, device=None):
    """
    Run the Weibel instability simulation with optimizations.
    
    Args:
        species_list: List of Species objects
        Lx, Ly: System dimensions
        Tmax: Maximum simulation time
        Nx, Ny, Nt: Grid resolution
        epochs: Number of training epochs
        batch_size: Mini-batch size for training
        lr: Learning rate
        device: Computation device (CPU/GPU)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    start_time = time.time()
    
    # Initialize model with optimized architecture
    model = PlasmaNetWeibel(species_list, hidden_layers=3, hidden_dim=128).to(device)
    
    # Use Adam optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=False
    )
    
    # Training loop with progress tracking
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    patience = 100  # Early stopping patience
    
    # Pre-generate random points for training
    num_points = batch_size * epochs
    x_points = torch.rand(num_points, 1, device=device) * Lx
    y_points = torch.rand(num_points, 1, device=device) * Ly
    t_points = torch.rand(num_points, 1, device=device) * Tmax
    
    # Add more points at t=0 for initial conditions
    t0_ratio = 0.1  # 10% of points at t=0
    t0_indices = torch.randint(0, num_points, (int(num_points * t0_ratio),))
    t_points[t0_indices] = 0.0
    
    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        
        # Get batch for this epoch
        start_idx = epoch * batch_size
        end_idx = (epoch + 1) * batch_size
        if end_idx > num_points:
            # Wrap around if we reach the end
            end_idx = num_points
            
        # Create tensors with requires_grad=True for autograd
        x_batch = x_points[start_idx:end_idx].clone().requires_grad_(True)
        y_batch = y_points[start_idx:end_idx].clone().requires_grad_(True)
        t_batch = t_points[start_idx:end_idx].clone().requires_grad_(True)
        
        # Forward pass
        species_outputs, fields = model(x_batch, y_batch, t_batch)
        
        # Split fields into components
        Ex, Ey, Ez, Bx, By, Bz = fields.split(1, dim=1)
        
        # Initialize loss
        total_loss = 0.0
        
        # Initial conditions: t=0
        mask_t0 = (t_batch == 0).squeeze()
        if mask_t0.any():
            for idx, species in enumerate(species_list):
                n, vx, vy, vz = species_outputs[idx][mask_t0].split(1, dim=1)
                
                # Density initialization
                n_init = species.density * (1.0 + 0.01 * torch.sin(2 * np.pi * x_batch[mask_t0] / Lx))
                total_loss += 10.0 * torch.mean((n - n_init)**2)
                
                # Velocity initialization with thermal perturbation
                vx_init = species.vx + species.vthx * torch.randn_like(vx)
                vy_init = species.vy + species.vthy * torch.randn_like(vy)
                
                total_loss += 10.0 * torch.mean((vx - vx_init)**2)
                total_loss += 10.0 * torch.mean((vy - vy_init)**2)
                total_loss += 10.0 * torch.mean(vz**2)  # vz should be 0 initially
        
        # Compute derivatives for Maxwell's equations
        dEx_dt = torch.autograd.grad(Ex, t_batch, grad_outputs=torch.ones_like(Ex), 
                                     create_graph=True, retain_graph=True)[0]
        dEy_dt = torch.autograd.grad(Ey, t_batch, grad_outputs=torch.ones_like(Ey), 
                                     create_graph=True, retain_graph=True)[0]
        dEz_dt = torch.autograd.grad(Ez, t_batch, grad_outputs=torch.ones_like(Ez), 
                                     create_graph=True, retain_graph=True)[0]
        
        dBx_dt = torch.autograd.grad(Bx, t_batch, grad_outputs=torch.ones_like(Bx), 
                                     create_graph=True, retain_graph=True)[0]
        dBy_dt = torch.autograd.grad(By, t_batch, grad_outputs=torch.ones_like(By), 
                                     create_graph=True, retain_graph=True)[0]
        dBz_dt = torch.autograd.grad(Bz, t_batch, grad_outputs=torch.ones_like(Bz), 
                                     create_graph=True, retain_graph=True)[0]
        
        # Spatial derivatives
        dEx_dx = torch.autograd.grad(Ex, x_batch, grad_outputs=torch.ones_like(Ex), 
                                     create_graph=True, retain_graph=True)[0]
        dEy_dy = torch.autograd.grad(Ey, y_batch, grad_outputs=torch.ones_like(Ey), 
                                     create_graph=True, retain_graph=True)[0]
        dEz_dz = torch.zeros_like(Ez)  # 2D simulation, no z-dependence
        
        dBx_dx = torch.autograd.grad(Bx, x_batch, grad_outputs=torch.ones_like(Bx), 
                                     create_graph=True, retain_graph=True)[0]
        dBy_dy = torch.autograd.grad(By, y_batch, grad_outputs=torch.ones_like(By), 
                                     create_graph=True, retain_graph=True)[0]
        dBz_dz = torch.zeros_like(Bz)  # 2D simulation, no z-dependence
        
        dEz_dx = torch.autograd.grad(Ez, x_batch, grad_outputs=torch.ones_like(Ez), 
                                     create_graph=True, retain_graph=True)[0]
        dEz_dy = torch.autograd.grad(Ez, y_batch, grad_outputs=torch.ones_like(Ez), 
                                     create_graph=True, retain_graph=True)[0]
        
        dBz_dx = torch.autograd.grad(Bz, x_batch, grad_outputs=torch.ones_like(Bz), 
                                     create_graph=True, retain_graph=True)[0]
        dBz_dy = torch.autograd.grad(Bz, y_batch, grad_outputs=torch.ones_like(Bz), 
                                     create_graph=True, retain_graph=True)[0]
        
        dEx_dy = torch.autograd.grad(Ex, y_batch, grad_outputs=torch.ones_like(Ex), 
                                     create_graph=True, retain_graph=True)[0]
        dEy_dx = torch.autograd.grad(Ey, x_batch, grad_outputs=torch.ones_like(Ey), 
                                     create_graph=True, retain_graph=True)[0]
        
        dBx_dy = torch.autograd.grad(Bx, y_batch, grad_outputs=torch.ones_like(Bx), 
                                     create_graph=True, retain_graph=True)[0]
        dBy_dx = torch.autograd.grad(By, x_batch, grad_outputs=torch.ones_like(By), 
                                     create_graph=True, retain_graph=True)[0]
        
        # Calculate current densities
        Jx = torch.zeros_like(Ex)
        Jy = torch.zeros_like(Ey)
        Jz = torch.zeros_like(Ez)
        
        for idx, species in enumerate(species_list):
            n, vx, vy, vz = species_outputs[idx].split(1, dim=1)
            Jx += species.charge * n * vx
            Jy += species.charge * n * vy
            Jz += species.charge * n * vz
        
        # Ampere's law
        ampere_x = dEx_dt - (dBz_dy) + Jx  # Simplified: dBy_dz = 0 in 2D
        ampere_y = dEy_dt + dBz_dx + Jy    # Simplified: dBx_dz = 0 in 2D
        ampere_z = dEz_dt - (dBy_dx - dBx_dy) + Jz
        
        # Faraday's law
        faraday_x = dBx_dt + dEz_dy        # Simplified: dEy_dz = 0 in 2D
        faraday_y = dBy_dt - dEz_dx        # Simplified: dEx_dz = 0 in 2D
        faraday_z = dBz_dt + (dEy_dx - dEx_dy)
        
        # Gauss's law
        total_charge_density = sum(species.charge * n for idx, species in enumerate(species_list) 
                                for n, _, _, _ in [species_outputs[idx].split(1, dim=1)])
        gauss_electric = dEx_dx + dEy_dy - total_charge_density  # Simplified: dEz_dz = 0 in 2D
        gauss_magnetic = dBx_dx + dBy_dy  # Simplified: dBz_dz = 0 in 2D
        
        # Add Maxwell's equations to loss with appropriate weights
        maxwell_weight = 1.0
        total_loss += maxwell_weight * (
            torch.mean(ampere_x**2) + torch.mean(ampere_y**2) + torch.mean(ampere_z**2) +
            torch.mean(faraday_x**2) + torch.mean(faraday_y**2) + torch.mean(faraday_z**2) +
            torch.mean(gauss_electric**2) + torch.mean(gauss_magnetic**2)
        )
        
        # Continuity and momentum equations for each species
        for idx, species in enumerate(species_list):
            n, vx, vy, vz = species_outputs[idx].split(1, dim=1)
            
            # Continuity equation
            dn_dt = torch.autograd.grad(n, t_batch, grad_outputs=torch.ones_like(n), 
                                       create_graph=True, retain_graph=True)[0]
            dn_dx = torch.autograd.grad(n, x_batch, grad_outputs=torch.ones_like(n), 
                                       create_graph=True, retain_graph=True)[0]
            dn_dy = torch.autograd.grad(n, y_batch, grad_outputs=torch.ones_like(n), 
                                       create_graph=True, retain_graph=True)[0]
            
            dvx_dx = torch.autograd.grad(vx, x_batch, grad_outputs=torch.ones_like(vx), 
                                        create_graph=True, retain_graph=True)[0]
            dvy_dy = torch.autograd.grad(vy, y_batch, grad_outputs=torch.ones_like(vy), 
                                        create_graph=True, retain_graph=True)[0]
            
            continuity = dn_dt + n * (dvx_dx + dvy_dy) + vx * dn_dx + vy * dn_dy
            total_loss += torch.mean(continuity**2)
            
            # Momentum equation
            dvx_dt = torch.autograd.grad(vx, t_batch, grad_outputs=torch.ones_like(vx), 
                                        create_graph=True, retain_graph=True)[0]
            dvy_dt = torch.autograd.grad(vy, t_batch, grad_outputs=torch.ones_like(vy), 
                                        create_graph=True, retain_graph=True)[0]
            dvz_dt = torch.autograd.grad(vz, t_batch, grad_outputs=torch.ones_like(vz), 
                                        create_graph=True, retain_graph=True)[0]
            
            qm = species.qm_ratio
            
            # Lorentz force
            Fx = qm * (Ex + vy * Bz - vz * By)
            Fy = qm * (Ey + vz * Bx - vx * Bz)
            Fz = qm * (Ez + vx * By - vy * Bx)
            
            # Convective derivative terms
            dvx_dx = torch.autograd.grad(vx, x_batch, grad_outputs=torch.ones_like(vx), 
                                        create_graph=True, retain_graph=True)[0]
            dvx_dy = torch.autograd.grad(vx, y_batch, grad_outputs=torch.ones_like(vx), 
                                        create_graph=True, retain_graph=True)[0]
            
            dvy_dx = torch.autograd.grad(vy, x_batch, grad_outputs=torch.ones_like(vy), 
                                        create_graph=True, retain_graph=True)[0]
            dvy_dy = torch.autograd.grad(vy, y_batch, grad_outputs=torch.ones_like(vy), 
                                        create_graph=True, retain_graph=True)[0]
            
            dvz_dx = torch.autograd.grad(vz, x_batch, grad_outputs=torch.ones_like(vz), 
                                        create_graph=True, retain_graph=True)[0]
            dvz_dy = torch.autograd.grad(vz, y_batch, grad_outputs=torch.ones_like(vz), 
                                        create_graph=True, retain_graph=True)[0]
            
            # Full momentum equations
            momentum_x = dvx_dt + vx * dvx_dx + vy * dvx_dy - Fx
            momentum_y = dvy_dt + vx * dvy_dx + vy * dvy_dy - Fy
            momentum_z = dvz_dt + vx * dvz_dx + vy * dvz_dy - Fz
            
            total_loss += torch.mean(momentum_x**2) + torch.mean(momentum_y**2) + torch.mean(momentum_z**2)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Track loss
        current_loss = total_loss.item()
        losses.append(current_loss)
        
        # Update learning rate
        scheduler.step(current_loss)
        
        # Early stopping check
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'results/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Print progress
        if epoch % 100 == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{epochs}, Loss: {current_loss:.6f}, Time: {elapsed:.2f}s")
            
            # Evaluate and plot intermediate results every 1000 epochs
            if epoch % 1000 == 0 and epoch > 0:
                plot_results(model, device, Lx, Ly, Tmax, epoch)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('results/best_model.pt'))
    
    # Final evaluation and plotting
    print("Training completed. Generating final results...")
    plot_results(model, device, Lx, Ly, Tmax)
    
    # Generate energy evolution plots
    plot_energy_evolution(model, device, Lx, Ly, Tmax)
    
    elapsed = time.time() - start_time
    print(f"Total simulation time: {elapsed:.2f} seconds")
    
    return model, losses

 Slightly higher learning rate with scheduler
        device=device
    )
    
    # Plot final loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/training_loss.png', dpi=300)
    
    print("Simulation completed successfully!")
    print("Results saved in the 'results' directory")

if __name__ == "__main__":
    main()