import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from datetime import datetime
from visualization.plotting import plot_results, plot_energy_evolution

def run_simulation(species_list, model_class, output_dir, Lx=40.0, Ly=40.0, Tmax=50.0, Nx=64, Ny=64, Nt=100, 
                   epochs=5000, batch_size=10000, lr=1e-3, device=None):
    """
    Run the plasma simulation with optimizations.
    
    Args:
        species_list: List of Species objects
        model_class: The neural network model class to use (e.g. PlasmaNet)
        output_dir: Directory to save results
        Lx, Ly: System dimensions
        Tmax: Maximum simulation time
        Nx, Ny, Nt: Grid resolution
        epochs: Number of training epochs
        batch_size: Mini-batch size for training
        lr: Learning rate
        device: Computation device (CPU/GPU)
    
    Returns:
        Trained model and training losses
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    start_time = time.time()
    
    # Initialize model
    model = model_class(species_list, hidden_layers=3, neurons_per_layer=128).to(device)
    
    # Use Adam optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
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
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        
        # Get batch for this epoch
        start_idx = epoch * batch_size
        end_idx = min((epoch + 1) * batch_size, num_points)
        
        # Create tensors with requires_grad=True for autograd
        x_batch = x_points[start_idx:end_idx].clone().requires_grad_(True)
        y_batch = y_points[start_idx:end_idx].clone().requires_grad_(True)
        t_batch = t_points[start_idx:end_idx].clone().requires_grad_(True)
        
        # Forward pass
        species_outputs, fields = model(x_batch, y_batch, t_batch)
        
        # Split fields into components
        Ex, Ey, Ez, Bx, By, Bz = torch.split(fields, 1, dim=1)
        
        # Initialize loss
        total_loss = 0.0
        
        # Initial conditions: t=0
        mask_t0 = (t_batch == 0).squeeze()
        if mask_t0.any():
            for idx, species in enumerate(species_list):
                n, vx, vy, vz = torch.split(species_outputs[idx][mask_t0], 1, dim=1)
                
                # Density initialization
                n_init = species.density * (1.0 + 0.01 * torch.sin(2 * np.pi * x_batch[mask_t0] / Lx))
                total_loss += 10.0 * torch.mean((n - n_init)**2)
                
                # Velocity initialization with thermal perturbation
                vx_init = species.vx0 + species.vth_x * torch.randn_like(vx)
                vy_init = species.vy0 + species.vth_y * torch.randn_like(vy)
                
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
            n, vx, vy, vz = torch.split(species_outputs[idx], 1, dim=1)
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
                                for n, _, _, _ in [torch.split(species_outputs[idx], 1, dim=1)])
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
            n, vx, vy, vz = torch.split(species_outputs[idx], 1, dim=1)
            
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
            torch.save(model.state_dict(), f'{output_dir}/best_model.pt')
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
                plot_results(model, device, Lx, Ly, Tmax, output_dir, epoch)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(f'{output_dir}/best_model.pt'))
    
    # Final evaluation and plotting
    print("Training completed. Generating final results...")
    plot_results(model, device, Lx, Ly, Tmax, output_dir)
    
    # Generate energy evolution plots
    plot_energy_evolution(model, device, Lx, Ly, Tmax, output_dir)
    
    elapsed = time.time() - start_time
    print(f"Total simulation time: {elapsed:.2f} seconds")
    
    return model, losses


