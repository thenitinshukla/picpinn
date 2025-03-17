import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from species import Species
from models.plasma_net import PlasmaNet
from simulation.runner import run_simulation

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
    eb1 = Species(name="eb1", mass=1.0, charge=-1.0, density=0.5, 
                       vx0=0.1, vy0=0.0, vz0=0.0, vth_x=0.01, vth_y=0.01)
    
    eb2 = Species(name="eb2", mass=1.0, charge=1.0, density=0.5, 
                       vx0=-0.1, vy0=0.0, vz0=0.0, vth_x=0.01, vth_y=0.01)

    ions = Species(name="ions", mass=1836.0, charge=1.0, density=1.0,
            vx0=0.0, vy0=0.0, vz0=0.0, vth_x=0.001, vth_y=0.001)

    species_list = [eb1, eb2, ions]
    
    # Run simulation with the specified parameters
    model, losses = run_simulation(
        species_list=species_list,
        model_class=PlasmaNet,
        output_dir=output_dir,
        Lx=100.0,
        Ly=70.0,
        Tmax=1000.0,
        Nx=1000,
        Ny=700,
        Nt=1000,
        epochs=5000,
        batch_size=10000,
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

if __name__ == "__main__":
    main()


