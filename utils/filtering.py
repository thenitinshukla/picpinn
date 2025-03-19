import torch
import numpy as np

def apply_filter_1d(data, num_passes=5):
    """
    Apply a 1D binomial filter multiple times.
    
    Args:
        data: 1D tensor to filter
        num_passes: Number of filter passes
        
    Returns:
        Filtered data
    """
    # Simple binomial filter [1,2,1]/4
    filtered = data.clone()
    
    for _ in range(num_passes):
        # Use constant padding for 1D data
        padded = torch.nn.functional.pad(filtered, (1, 1), mode='constant', value=0)
        filtered = (padded[:-2] + 2*padded[1:-1] + padded[2:]) / 4.0
        
    return filtered

def apply_filter_2d(data, num_passes=5):
    """
    Apply a 2D binomial filter multiple times.
    
    Args:
        data: 2D tensor to filter
        num_passes: Number of filter passes
        
    Returns:
        Filtered data
    """
    # Convert to numpy for easier filtering
    data_np = data.cpu().numpy()
    filtered = data_np.copy()
    
    # Simple 3x3 binomial filter kernel
    kernel = np.array([[1, 2, 1], 
                       [2, 4, 2], 
                       [1, 2, 1]]) / 16.0
    
    # Apply filter multiple times
    for _ in range(num_passes):
        # Create padded array
        padded = np.pad(filtered, ((1, 1), (1, 1)), mode='edge')
        result = np.zeros_like(filtered)
        
        # Apply convolution manually
        for i in range(filtered.shape[0]):
            for j in range(filtered.shape[1]):
                result[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
        
        filtered = result
    
    # Convert back to torch tensor
    return torch.tensor(filtered, device=data.device)

def filter_fields(Ex, Ey, Ez, Bx, By, Bz, num_passes=5):
    """
    Apply a 5-pass filter to electromagnetic fields.
    
    Args:
        Ex, Ey, Ez, Bx, By, Bz: Field components
        num_passes: Number of filter passes
        
    Returns:
        Filtered field components
    """
    # Check dimensions and reshape if needed
    if Ex.dim() == 1:
        # Apply 1D filter
        Ex_filtered = apply_filter_1d(Ex, num_passes)
        Ey_filtered = apply_filter_1d(Ey, num_passes)
        Ez_filtered = apply_filter_1d(Ez, num_passes)
        Bx_filtered = apply_filter_1d(Bx, num_passes)
        By_filtered = apply_filter_1d(By, num_passes)
        Bz_filtered = apply_filter_1d(Bz, num_passes)
    else:
        # Apply 2D filter if the data is already 2D
        Ex_filtered = apply_filter_2d(Ex, num_passes)
        Ey_filtered = apply_filter_2d(Ey, num_passes)
        Ez_filtered = apply_filter_2d(Ez, num_passes)
        Bx_filtered = apply_filter_2d(Bx, num_passes)
        By_filtered = apply_filter_2d(By, num_passes)
        Bz_filtered = apply_filter_2d(Bz, num_passes)
        
    return Ex_filtered, Ey_filtered, Ez_filtered, Bx_filtered, By_filtered, Bz_filtered

def fourth_order_interpolation(grid_values, positions, grid_spacing):
    """
    Perform fourth-order interpolation.
    
    Args:
        grid_values: Values on the grid
        positions: Positions to interpolate to
        grid_spacing: Grid spacing
        
    Returns:
        Interpolated values
    """
    # Normalize positions to grid indices
    indices = positions / grid_spacing
    
    # Get integer indices
    i0 = torch.floor(indices).long()
    
    # Get fractional part
    dx = indices - i0
    
    # Get grid indices with boundary handling
    i_m1 = torch.clamp(i0 - 1, 0, len(grid_values) - 1)
    i_0 = torch.clamp(i0, 0, len(grid_values) - 1)
    i_p1 = torch.clamp(i0 + 1, 0, len(grid_values) - 1)
    i_p2 = torch.clamp(i0 + 2, 0, len(grid_values) - 1)
    
    # Fourth-order Lagrange interpolation weights
    w_m1 = -dx * (dx - 1) * (dx - 2) / 6
    w_0 = (dx + 1) * (dx - 1) * (dx - 2) / 2
    w_p1 = -(dx + 1) * dx * (dx - 2) / 2
    w_p2 = (dx + 1) * dx * (dx - 1) / 6
    
    # Interpolate
    return (w_m1 * grid_values[i_m1] + 
            w_0 * grid_values[i_0] + 
            w_p1 * grid_values[i_p1] + 
            w_p2 * grid_values[i_p2])


