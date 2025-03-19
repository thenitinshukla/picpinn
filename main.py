import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from species import Species
from models.plasma_net import PlasmaNet
from simulation.runner import run_simulation
from visualization.density_plots import plot_density_at_t0

def main():
    """
    Main function to set up and run the plasma physics simulation.
    """
    # Create output directory for results with timestamp
    output_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define species
    # Example: Create a pair plasma with electrons and positrons
    # Counter-propagating beams to observe Weibel instability and filamentary structures
    e1 = Species(name="e1", mass=1.0, charge=-1.0, density=0.5, 
                       vx0=0.1, vy0=0.0, vz0=0.0, vth_x=0.01, vth_y=0.01)
    p1 = Species(name="p1", mass=1.0, charge=1.0, density=0.5,
                       vx0=0.1, vy0=0.0, vz0=0.0, vth_x=0.01, vth_y=0.01)
    e2 = Species(name="e2", mass=1.0, charge=-1.0, density=0.5,
                       vx0=-0.1, vy0=0.0, vz0=0.0, vth_x=0.01, vth_y=0.01) 
    p2 = Species(name="p2", mass=1.0, charge=1.0, density=0.5,
                       vx0=-0.1, vy0=0.0, vz0=0.0, vth_x=0.01, vth_y=0.01)
    
    species_list = [e1, e2, p1, p2]
    
    # Run simulation with the specified parameters
    model, losses = run_simulation(
        species_list=species_list,
        model_class=PlasmaNet,
        output_dir=output_dir,
        Lx=100.0,
        Ly=70.0,
        Tmax=1000.0,
        Nx=1000,  # Reduced from 10000 for memory efficiency
        Ny=700,   # Reduced from 7000 for memory efficiency
        Nt=0.01, # Changed from 0.001 to match the prompt
        epochs=2048,
        batch_size=2000,
        lr=1e-5,
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
    plt.savefig(f'{output_dir}/training_loss.png', dpi=300)
    
    print("Simulation completed successfully!")
    print(f"Results saved in the '{output_dir}' directory")
    print("The simulation should show small-scale magnetic filamentary structures in Bz")
    print("These structures are characteristic of the Weibel instability in counter-streaming plasmas")

if __name__ == "__main__":
    main()
