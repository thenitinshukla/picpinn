import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from utils.fitting import exponential_fit
import os

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
    
    t_vals, bx_history, by_history, bz_history, vx_history, vy_history = evaluate_model(
        model, device, Lx, Ly, Tmax, Nx=32, Ny=32, Nt=100
    )
    
    # Create plot for magnetic field evolution
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, bx_history, 'r-', label=r'$|B_x|$')
    plt.plot(t_vals, by_history, 'g-', label=r'$|B_y|$')
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
    
    print(f"Results saved in {output_dir}")


