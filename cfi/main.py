import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import os
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

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(neurons_per_layer, self.output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        out = self.network(inputs)

        # Split outputs
        outputs = []
        ptr = 0
        for _ in self.species:
            outputs.append(out[:, ptr:ptr+4])
            ptr += 4
        fields = out[:, ptr:ptr+6]
        return outputs, fields

def exponential_fit(t, a, b, c):
    """Exponential function for fitting growth rate: a * exp(b * t) + c"""
    return a * np.exp(b * t) + c

def run_simulation(species_list, Lx=100.0, Ly=20.0, Tmax=200.0, batch_size=1024, epochs=1000):
    start_time = time.time()
    print(f"Starting simulation with {len(species_list)} species")

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = PlasmaNet(species_list).to(device)

    # Training grid
    Nx, Ny, Nt = 32, 32, 50  # Reduced grid size for faster training
    x = torch.linspace(0, Lx, Nx, device=device)
    y = torch.linspace(0, Ly, Ny, device=device)
    t = torch.linspace(0, Tmax, Nt, device=device)

    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    x_train = X.reshape(-1,1)
    y_train = Y.reshape(-1,1)
    t_train = T.reshape(-1,1)

    # Create dataset
    train_data = torch.utils.data.TensorDataset(x_train, y_train, t_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Optimizer with learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    # Training history
    losses = []

    def loss_fn(model, x, y, t):
        # Make inputs require grad
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        species_outputs, fields = model(x, y, t)
        Ex, Ey, Ez, Bx, By, Bz = fields.split(1, dim=1)

        # Initialize loss
        total_loss = 0.0

        # Initial conditions
        mask_t0 = (t < 0.01).squeeze()
        if mask_t0.any():
            for idx, species in enumerate(species_list):
                n, ux, uy, uz = species_outputs[idx][mask_t0].split(1, dim=1)

                # Density initialization
                total_loss += torch.mean((n - species.density)**2)

                # Velocity initialization with thermal perturbation
                total_loss += torch.mean((ux - species.vx0)**2)
                total_loss += torch.mean((uy - species.vy0)**2)
                total_loss += torch.mean((uz - species.vz0)**2)

        # Maxwell's equations
        dBz_dt = torch.autograd.grad(Bz, t,
            grad_outputs=torch.ones_like(Bz), create_graph=True)[0]
        dEy_dx = torch.autograd.grad(Ey, x,
            grad_outputs=torch.ones_like(Ey), create_graph=True)[0]
        dEx_dy = torch.autograd.grad(Ex, y,
            grad_outputs=torch.ones_like(Ex), create_graph=True)[0]

        # Faraday's law for Bz
        faraday_loss = torch.mean((dBz_dt + dEy_dx - dEx_dy)**2)
        total_loss += faraday_loss

        # Current density calculation
        Jx, Jy, Jz = 0, 0, 0
        for idx, species in enumerate(species_list):
            n, ux, uy, uz = species_outputs[idx].split(1, dim=1)
            Jx += species.charge * n * ux
            Jy += species.charge * n * uy
            Jz += species.charge * n * uz

        # Ampere's law
        dBz_dy = torch.autograd.grad(Bz, y,
            grad_outputs=torch.ones_like(Bz), create_graph=True)[0]
        dBy_dz = torch.zeros_like(Bz)  # 2D simulation, no z-dependence
        dBx_dy = torch.autograd.grad(Bx, y,
            grad_outputs=torch.ones_like(Bx), create_graph=True)[0]
        dBy_dx = torch.autograd.grad(By, x,
            grad_outputs=torch.ones_like(By), create_graph=True)[0]
        
        # Add the missing gradient calculation
        dBz_dx = torch.autograd.grad(Bz, x,
            grad_outputs=torch.ones_like(Bz), create_graph=True)[0]
        dBx_dz = torch.zeros_like(Bz)  # 2D simulation, no z-dependence

        dEx_dt = torch.autograd.grad(Ex, t,
            grad_outputs=torch.ones_like(Ex), create_graph=True)[0]
        ampere_x_loss = torch.mean((dBy_dz - dBz_dy - Jx - dEx_dt)**2)

        dEy_dt = torch.autograd.grad(Ey, t,
            grad_outputs=torch.ones_like(Ey), create_graph=True)[0]
        ampere_y_loss = torch.mean((dBz_dx - dBx_dz - Jy - dEy_dt)**2)

        total_loss += ampere_x_loss + ampere_y_loss

        # Momentum equations for each species
        for idx, species in enumerate(species_list):
            n, ux, uy, uz = species_outputs[idx].split(1, dim=1)
            qm = species.qm_ratio

            # x-component
            Fx = qm*(Ex + uy*Bz - uz*By)
            dux_dt = torch.autograd.grad(ux, t,
                grad_outputs=torch.ones_like(ux), create_graph=True)[0]
            momentum_x_loss = torch.mean((dux_dt - Fx)**2)

            # y-component
            Fy = qm*(Ey + uz*Bx - ux*Bz)
            duy_dt = torch.autograd.grad(uy, t,
                grad_outputs=torch.ones_like(uy), create_graph=True)[0]
            momentum_y_loss = torch.mean((duy_dt - Fy)**2)

            total_loss += momentum_x_loss + momentum_y_loss

        return total_loss

    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    patience = 20  # Early stopping patience

    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch, t_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model, x_batch, y_batch, t_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # Learning rate scheduling
        scheduler.step(avg_loss)

        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch {epoch:04d}/{epochs} | Loss: {avg_loss:.2e} | Time: {time.time() - start_time:.1f}s')

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(torch.load(f"{output_dir}/best_model.pt"))

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(f"{output_dir}/training_loss.png", dpi=150)
    plt.close()

    print("Training completed. Starting analysis...")

    # Analysis grid (higher resolution for better visualization)
    Nx_plot, Ny_plot, Nt_plot = 50, 50, 100
    x_plot = torch.linspace(0, Lx, Nx_plot, device=device)
    y_plot = torch.linspace(0, Ly, Ny_plot, device=device)
    t_plot = torch.linspace(0, Tmax, Nt_plot, device=device)

    # Collect data for analysis
    bz_max_values = []
    bz_avg_values = []
    vx_avg_values = []
    vy_avg_values = []

    # Process in smaller batches to avoid memory issues
    t_values = t_plot.cpu().numpy()

    for t_idx, t_val in enumerate(t_plot):
        # Create 2D grid for this time step
        X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
        x_grid = X.reshape(-1, 1)
        y_grid = Y.reshape(-1, 1)
        t_grid = torch.full_like(x_grid, t_val)

        # Get model predictions
        with torch.no_grad():
            outputs, fields = model(x_grid, y_grid, t_grid)

            # Extract fields
            Ex, Ey, Ez, Bx, By, Bz = fields.split(1, dim=1)

            # Reshape to 2D grid
            Bz_2d = Bz.reshape(Nx_plot, Ny_plot).cpu().numpy()

            # Store max and average Bz
            bz_max_values.append(np.max(np.abs(Bz_2d)))
            bz_avg_values.append(np.mean(np.abs(Bz_2d)))

            # Process species data
            vx_species = []
            vy_species = []

            for idx, species in enumerate(species_list):
                n, ux, uy, uz = outputs[idx].split(1, dim=1)

                # Reshape to 2D grid
                ux_2d = ux.reshape(Nx_plot, Ny_plot).cpu().numpy()
                uy_2d = uy.reshape(Nx_plot, Ny_plot).cpu().numpy()

                # Store average velocities
                vx_species.append(np.mean(ux_2d))
                vy_species.append(np.mean(uy_2d))

            vx_avg_values.append(vx_species)
            vy_avg_values.append(vy_species)

    # Convert to numpy arrays
    vx_avg_values = np.array(vx_avg_values)
    vy_avg_values = np.array(vy_avg_values)

    # Calculate growth rate of Bz
    # Find the region of exponential growth (after initial transient,before saturation)
    growth_start_idx = int(Nt_plot * 0.1)  # Skip initial 10%
    growth_end_idx = int(Nt_plot * 0.7)    # Use up to 70% of time

    # Fit exponential to the growth phase
    try:
        growth_t = t_values[growth_start_idx:growth_end_idx]
        growth_bz = bz_max_values[growth_start_idx:growth_end_idx]

        # Initial guess for parameters
        p0 = [growth_bz[0], 0.1, 0]

        # Fit exponential function
        params, _ = curve_fit(exponential_fit, growth_t, growth_bz, p0=p0)

        # Extract growth rate
        growth_rate = params[1]
        print(f"Estimated growth rate: {growth_rate:.4f} ωp")

        # Generate fitted curve
        fit_t = np.linspace(t_values[0], t_values[-1], 200)
        fit_bz = exponential_fit(fit_t, *params)
    except Exception as e:
        print(f"Error fitting growth rate: {e}")
        growth_rate = None
        fit_t = []
        fit_bz = []

    # Plot Bz growth
    plt.figure(figsize=(12, 8))
    plt.semilogy(t_values, bz_max_values, 'b-', label='Max |Bz|')
    plt.semilogy(t_values, bz_avg_values, 'g--', label='Avg |Bz|')

    if growth_rate is not None:
        plt.semilogy(fit_t, fit_bz, 'r-', label=f'Fit: γ = {growth_rate:.4f} ωp')
        plt.axvspan(t_values[growth_start_idx], t_values[growth_end_idx], alpha=0.2,
        color='yellow', label='Fitting region')

    plt.xlabel(r'$t\ [1/\omega_p]$')
    plt.ylabel(r'$|B_z|$')
    plt.title('Magnetic Field Growth (Weibel Instability)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/bz_growth.png", dpi=150)
    plt.close()

    # Plot velocity components vs time
    plt.figure(figsize=(12, 8))
    for idx, species in enumerate(species_list):
        plt.plot(t_values, vx_avg_values[:, idx], '-',
label=f'{species.name} vx')
        plt.plot(t_values, vy_avg_values[:, idx], '--',
label=f'{species.name} vy')

    plt.xlabel(r'$t\ [1/\omega_p]$')
    plt.ylabel(r'Velocity $[c]$')
    plt.title('Species Velocities vs Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/velocities.png", dpi=150)
    plt.close()

    # Create 2D plots of Bz at different times
    plot_times = [0.0, 0.25*Tmax, 0.5*Tmax, 0.75*Tmax, Tmax]
    fig, axes = plt.subplots(1, len(plot_times), figsize=(20, 4))

    for i, t_val in enumerate(plot_times):
        t_idx = min(Nt_plot-1, int(t_val/Tmax * Nt_plot))

        # Create 2D grid for this time step
        X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
        x_grid = X.reshape(-1, 1)
        y_grid = Y.reshape(-1, 1)
        t_grid = torch.full_like(x_grid, t_plot[t_idx])

        # Get model predictions
        with torch.no_grad():
            _, fields = model(x_grid, y_grid, t_grid)
            Bz = fields[:, 5].reshape(Nx_plot, Ny_plot).cpu().numpy()

        # Plot
        im = axes[i].imshow(Bz.T, origin='lower', extent=[0, Lx, 0, Ly],
                           aspect='auto', cmap='RdBu_r')
        axes[i].set_title(f't = {t_values[t_idx]:.1f}')
        axes[i].set_xlabel('x')
        if i == 0:
            axes[i].set_ylabel('y')

    plt.colorbar(im, ax=axes, label='Bz')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bz_evolution.png", dpi=150)
    plt.close()

    # Save data for further analysis
    np.savez(f"{output_dir}/simulation_data.npz",
             t=t_values,
             bz_max=bz_max_values,
             bz_avg=bz_avg_values,
             vx=vx_avg_values,
             vy=vy_avg_values,
             growth_rate=growth_rate if growth_rate is not None else 0)

    print(f"Simulation completed in {time.time() - start_time:.1f} seconds")
    print(f"Results saved to {output_dir}/")

    return {
        'growth_rate': growth_rate,
        'bz_max': bz_max_values,
        'vx': vx_avg_values,
        'vy': vy_avg_values,
        't': t_values
    }

# Example usage: Counter-streaming electron beams (Weibel instability)
if __name__ == "__main__":
    # Create species with counter-streaming velocities
    species = [
        Species('Electrons+', 1.0, -1.0, 0.5, 0.2, vth_x=0.01, vth_y=0.01),
        Species('Electrons-', 1.0, -1.0, 0.5, -0.2, vth_x=0.01, vth_y=0.01)
    ]

    # Run simulation
    results = run_simulation(species, Lx=50.0, Ly=50.0, Tmax=100.0, epochs=500)

    print("Simulation completed successfully!")