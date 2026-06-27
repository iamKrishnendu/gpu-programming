# GPU-Accelerated 2D Heat Diffusion PDE Solver using Numba CUDA

## Overview

This project implements a **GPU-accelerated 2D Heat Diffusion Partial Differential Equation (PDE) solver** using **Numba CUDA**. The objective is to compare CPU and GPU implementations of a stencil-based finite difference solver and evaluate the performance improvements achieved through GPU parallelization.

---

## Team

- Sameer Sarkar (G25AIT1149)
- Krishnendu Halder (G25AIT1077)
- Samanwita Patra (G25AIT1148)
- Rajdeep Acharyya Chowdhury (G25AIT1128)

**Course:** GPU Programming  
**Institute:** Indian Institute of Technology Jodhpur (IITJ)

---

## Problem Statement

Implement a **2D Heat Diffusion PDE Solver** using a **5-point stencil** and compare its execution performance on:

- CPU (NumPy)
- GPU (Numba CUDA)

---

## Applications

- CPU thermal management
- Battery cooling in EVs
- Weather simulation
- Metal heating and cooling
- Building thermal analysis (HVAC)

---

## Mathematical Model

Heat Diffusion Equation:

```math
∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)
```

Finite Difference Update:

```math
u(i,j)^{t+1} = u(i,j)^t + r \left(
u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4u_{i,j}
\right)
```

where

```text
r = αΔt / Δx²
```

### Stability Condition

```text
r ≤ 0.25
```

---

## 5-Point Stencil

```text
        Up

Left   Center   Right

      Down
```

Each grid cell is updated using its four neighboring cells.

---

## Technologies Used

- Python 3
- NumPy
- Numba CUDA
- CUDA Toolkit
- Matplotlib

---

## GPU Implementation

The CUDA kernel assigns one GPU thread per grid cell.

Workflow:

1. Copy data from CPU to GPU
2. Launch CUDA kernel
3. Compute stencil update
4. Copy results back to CPU

GPU Concepts:

- CUDA Kernels
- Thread Blocks (16×16)
- Grid Configuration
- Device Memory
- Host-Device Memory Transfer

---

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Grid Sizes | 256², 512², 1024², 2048² |
| Iterations | 50–100 |
| Thermal Diffusivity (α) | 0.2 |
| Stability Parameter (r) | 0.2 |

---

## Performance Comparison

| Grid Size | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| 256 × 256 | 0.12 s | 0.008 s | 15× |
| 512 × 512 | 2.34 s | 0.18 s | 13× |
| 1024 × 1024 | 37.5 s | 1.2 s | 31× |
| 2048 × 2048 | 600 s | 9.6 s | 62× |

---

## Key Observations

- GPU achieves **13×–62×** speedup over CPU.
- Performance advantage increases with larger grid sizes.
- Numerical accuracy remains consistent between CPU and GPU implementations.
- GPU efficiently utilizes thousands of parallel CUDA threads.

---

## Why GPU Performs Better

- Massive parallelism
- Higher memory bandwidth
- Better latency hiding
- Superior floating-point throughput

---

## Results

### Performance

- Significant reduction in execution time
- Better scalability for larger computational grids

### Numerical Validation

- CPU and GPU outputs differ only within floating-point precision (`10⁻⁵` to `10⁻⁶`).
- CFL stability condition is maintained throughout the simulation.

---

## Future Enhancements

- 3D Heat Diffusion Solver
- Multi-GPU implementation
- Adaptive Mesh Refinement (AMR)
- Implicit PDE Solvers
- Support for Wave, Laplace and Poisson equations
- Mixed Precision (FP16)

---

## References

- CUDA C Programming Guide
- Numba Documentation
- NVIDIA Developer Documentation
- GPU Gems 3
- Numerical Recipes in C++
- Lawrence C. Evans – *Partial Differential Equations*

---

## Conclusion

This project demonstrates that stencil-based PDE solvers are highly suitable for GPU acceleration. By leveraging Numba CUDA, the GPU implementation significantly outperforms the CPU while maintaining numerical correctness. As the computational grid grows, GPU acceleration becomes increasingly beneficial, making it an effective solution for large-scale scientific and engineering simulations.