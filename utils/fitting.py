import numpy as np
def exponential_fit(t, a, b, c):
    """
    Exponential function for fitting growth rate: a * exp(b * t) + c
    
    Args:
        t: Time values
        a: Amplitude
        b: Growth rate
        c: Offset
        
    Returns:
        Exponential function values
    """
    return a * np.exp(b * t) + c
