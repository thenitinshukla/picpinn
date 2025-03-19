import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from utils.fitting import exponential_fit
import os

def plot_magnetic_field_2d(bz_field_history, x, y, t_vals, output_dir, num_plots=5, cmap='seismic', enhance_contrast=True):
    """
    Plot 2D magnetic field Bz(x,y) for different time steps with enhanced visualization
    to highlight filamentary structures.
    
    Args:
        bz_field_history: 3D array of Bz field data (time, y, x)
        x, y: Spatial coordinates
        t_vals: Time values
        output_dir: Directory to save results
        num_plots: Number of time steps to plot
        cmap: Colormap to use
        enhance_contrast: Whether to enhance contrast to highlight structures
    """
    Nt, Ny, Nx = bz_field_history.shape
    plot_indices = np.linspace(0, Nt-1, num_plots, dtype=int)
    
    # Create a directory for 2D plots
    bz_dir = f'{output_dir}/bz_2d_plots'
    os.makedirs(bz_dir, exist_ok=True)
    
    # Create a figure for all plots
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    for i, idx in enumerate(plot_indices):
        # Get the data for this time step
        bz_data = bz_field_history[idx]
        
        # Enhance contrast if requested
        if enhance_contrast:
            # Compute percentiles for better contrast
            vmin, vmax = np.percentile(bz_data, [2, 98])
            # Ensure symmetric colormap around zero
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
        else:
            vmin, vmax = None, None
        
        # Plot in the combined figure
        im = axes[i].imshow(bz_data, extent=[x[0], x[-1], y[0], y[-1]], 
                           origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].set_xlabel('x')
        if i == 0:
            axes[i].set_ylabel('y')
        axes[i].set_title(f't = {t_vals[idx]:.2f}')
        plt.colorbar(im, ax=axes[i], label='Bz')
        
        # Also create individual high-resolution plots
        fig_single, ax_single = plt.subplots(figsize=(10, 8))
        im_single = ax_single.imshow(bz_data, extent=[x[0], x[-1], y[0], y[-1]], 
                                    origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im_single, ax=ax_single, label='Bz')
        ax_single.set_xlabel('x')
        ax_single.set_ylabel('y')
        ax_single.set_title(f'Magnetic Field Bz(x,y) at t = {t_vals[idx]:.2f}')
        
        # Add contour lines to highlight structures
        # Ensure contour levels are strictly increasing
        vmin_contour = np.nanmin(bz_data) if vmin is None else vmin
        vmax_contour = np.nanmax(bz_data) if vmax is None else vmax
        
        # Ensure vmin_contour < vmax_contour
        if vmin_contour >= vmax_contour:
            # If they're equal or vmin > vmax, create a small range
            vmin_contour, vmax_contour = -1.0, 1.0
        
        # Create strictly increasing contour levels
        contour_levels = np.linspace(vmin_contour, vmax_contour, 10)
        
        try:
            ax_single.contour(bz_data, levels=contour_levels, colors='k', alpha=0.3, 
                             extent=[x[0], x[-1], y[0], y[-1]])
        except ValueError as e:
            print(f"Warning: Could not draw contours: {e}")
        
        plt.tight_layout()
        plt.savefig(f'{bz_dir}/bz_2d_t{idx:04d}.png', dpi=300)
        plt.close(fig_single)
    
    # Save the combined figure
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bz_2d_evolution.png', dpi=300)
    plt.close(fig)
    
    # Create a more detailed plot for the last time step to show filaments
    idx = plot_indices[-1]
    bz_data = bz_field_history[idx]
    
    # Create a high-resolution figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use a perceptually uniform colormap with enhanced contrast
    if enhance_contrast:
        vmin, vmax = np.percentile(bz_data, [1, 99])
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    else:
        vmin, vmax = None, None
    
    # Plot with high resolution
    im = ax.imshow(bz_data, extent=[x[0], x[-1], y[0], y[-1]], 
                  origin='lower', interpolation='bicubic', 
                  cmap='RdBu_r', vmin=vmin, vmax=vmax)
    
    # Add contour lines to highlight filamentary structures
    # Ensure contour levels are strictly increasing
    vmin_contour = np.nanmin(bz_data) if vmin is None else vmin
    vmax_contour = np.nanmax(bz_data) if vmax is None else vmax
    
    # Ensure vmin_contour < vmax_contour
    if vmin_contour >= vmax_contour:
        # If they're equal or vmin > vmax, create a small range
        vmin_contour, vmax_contour = -1.0, 1.0
    
    # Create strictly increasing contour levels
    contour_levels = np.linspace(vmin_contour, vmax_contour, 20)
    
    try:
        cs = ax.contour(bz_data, levels=contour_levels, colors='k', alpha=0.3, 
                       extent=[x[0], x[-1], y[0], y[-1]])
    except ValueError as e:
        print(f"Warning: Could not draw contours: {e}")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Bz Magnetic Field', fontsize=12)
    
    # Add labels and title
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title(f'Magnetic Field Filamentary Structures at t = {t_vals[idx]:.2f}', fontsize=16)
    
    # Add grid
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bz_filaments_detailed.png', dpi=600)
    plt.close()

def plot_energy_evolution(model, device, Lx, Ly, Tmax, output_dir, suffix=""):
    """
    Generate plots of electromagnetic energy and kinetic energy evolution.
    
    Args:
        model: The trained PlasmaNet model
        device: Computation device (CPU/GPU)
        Lx, Ly: System dimensions
        Tmax: Maximum simulation time
        output_dir: Directory to save results
        suffix: Suffix for output files
    """
    from utils.evaluation import calculate_energy_components
    
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
        popt, _ = curve_fit(exponential_fit, t_fit, bz_fit_data, p0=p0)
        A, gamma, C = popt
        
        # Plot theoretical growth rate
        ax1.semilogy(t_fit, exponential_fit(t_fit, A, gamma, C), 'r:', linewidth=1.5)
        
        # Add growth rate annotation
        ax1.annotate(f'$\\gamma = {gamma:.4f}\\omega_p$', 
                    xy=(t_fit[-1], exponential_fit(t_fit[-1], A, gamma, C)),
                    xytext=(t_fit[-1]+5, exponential_fit(t_fit[-1], A, gamma, C)*2),
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
    plt.savefig(f'{output_dir}/energy_evolution{suffix}.png', dpi=300)
    
    # Save data for further analysis
    np.savez(f'{output_dir}/energy_data{suffix}.npz', 
             t=t_vals, ex=ex_energy, ey=ey_energy, ez=ez_energy,
             bx=bx_energy, by=by_energy, bz=bz_energy,
             k_long=k_long, k_trans_y=k_trans_y, k_trans_z=k_trans_z)
    
    print(f"Energy evolution plots saved to '{output_dir}/energy_evolution{suffix}.png'")
    plt.close()

def plot_results(model, device, Lx, Ly, Tmax, output_dir, epoch=None):
    """
    Plot simulation results at the current state of the model.
    
    Args:
        model: The trained PlasmaNet model
        device: Computation device (CPU/GPU)
        Lx, Ly: System dimensions
        Tmax: Maximum simulation time
        output_dir: Directory to save results
        epoch: Current epoch number (optional, for intermediate results)
    """
    from utils.evaluation import evaluate_model
    
    suffix = f"_epoch{epoch}" if epoch is not None else ""
    print(f"Evaluating model{suffix}...")
    
    t_vals, bx_history, by_avg_history, bz_history, vx_history, vy_history, bz_field_history = evaluate_model(
        model, device, Lx, Ly, Tmax, Nx=64, Ny=64, Nt=100  # Higher resolution for better visualization
    )
    
    # Create plot for magnetic field evolution
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, bx_history, 'r-', label=r'$|B_x|$')
    plt.plot(t_vals, by_avg_history, 'g-', label=r'$|B_y|$')
    plt.plot(t_vals, bz_history, 'b-', label=r'$|B_z|$')
    plt.xlabel(r'$t$ [$\omega_p^{-1}$]', fontsize=12)
    plt.ylabel(r'Magnetic field magnitude', fontsize=12)
    plt.title('Magnetic Field Evolution', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/magnetic_field{suffix}.png', dpi=300)
    plt.close()
    
    # Create plot for velocity evolution (first species only for simplicity)
    if len(vx_history) > 0:
        plt.figure(figsize=(10, 6))
        for i in range(len(vx_history)):
            plt.plot(t_vals, vx_history[i], label=f'Species {i+1} - vx')
            plt.plot(t_vals, vy_history[i], '--', label=f'Species {i+1} - vy')
        plt.xlabel(r'$t$ [$\omega_p^{-1}$]', fontsize=12)
        plt.ylabel(r'Velocity [$c$]', fontsize=12)
        plt.title('Velocity Evolution', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/velocity{suffix}.png', dpi=300)
        plt.close()
    
    # Plot 2D magnetic field Bz(x,y) for different time steps
    x = np.linspace(0, Lx, bz_field_history.shape[2])
    y = np.linspace(0, Ly, bz_field_history.shape[1])
    plot_magnetic_field_2d(bz_field_history, x, y, t_vals, output_dir, num_plots=5, enhance_contrast=True)
    
    print(f"Results saved in {output_dir}")         