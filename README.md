# picpinn

This repository helps me understand Physics-Informed Neural Networks (PINNs).

## Problem Proposition

- **Scientific simulation** is well-established and grid-based, but it's often slow and rigid.
- Unlike traditional data-driven problems, **PINNs** are suitable for well-established physics problems.
- **Bridge the gap**: Combining physics-aware models with data to achieve scientific accuracy using less data.
- Can we enhance **scientific productivity** while complementing conventional scientific computing?
- To illustrate this, let's reflect on a well-established problem in **plasma physics**.

We aim to solve the **Vlasov-Maxwell equation** using PINNs and study phenomena like the **Weibel instability**.

## Steps

1. Define the number of species.
2. Define the charge density: \( n \).
3. Define the mass-to-charge ratio: \( q \).
4. Consider a two-dimensional system: (Lx, Ly).
5. Define the flow velocity: \( (v_x, v_y) \).
6. Define the thermal velocity: \( (v_{\text{thx}}, v_{\text{thy}}) \).

## Result Analysis

- Analyze the growth of magnetic fields over time for different velocities.
- Compute \( v_x \) vs time.

