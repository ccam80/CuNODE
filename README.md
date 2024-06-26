<<<<<<< HEAD
# CuNODE
**Cuda (through Numba) ODE integrators**

ODE solvers for large parallel operations (e.g. grid searches) where the benefits of running solvers in parallel outweighs the penalty of the slightly slower runtime of a single solve.

**Requirements:**
Hardware:
NVIDIA GPU, Compute Capability > 2.* (I think)
Intelpython3 python environment
CUDA toolkit 12.x
numpy, numba, matplotlib

Intended functionality
- 
- You provide a dxdt function (fill in the function template), and select an integration type, step size, duration, and output sample frequency
- You provide a vector of variables to run (e.g. a 2D grid search looks like ((0, 0), (0,1), (1,0), (1,1))
- run euler, rk4, or other integration function
- reap rewards

