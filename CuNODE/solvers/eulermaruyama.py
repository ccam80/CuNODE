# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:31:43 2024

@author: cca79
"""

# -*- coding: utf-8 -*-
import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"



import numpy as np
from cupy import asarray, ascontiguousarray, get_default_memory_pool
from numba import cuda, from_dtype, literally
from numba import float32, float64, int32, int64, void, int16
from systems import  thermal_cantilever_ax_b # For testing code only, do not touch otherwise
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64, xoroshiro128p_dtype
from cupyx.scipy.signal import firwin, welch
from cupyx.scipy.fft import rfftfreq, rfft
from _utils import get_noise_64, get_noise_32, timing
from solvers.genericSolver import genericSolver


xoro_type = from_dtype(xoroshiro128p_dtype)

# global0 = 0

class Solver(genericSolver):

    def __init__(self,
                 precision = np.float32,
                 diffeq_sys = None
                 ):
        super().__init__(precision, diffeq_sys)

        self.precision = precision

    def build_kernel(self, system):
        dxdtfunc = system.dxdtfunc

        global zero
        zero = 0
        global nstates
        nstates = system.num_states
        global constants_length
        constants_length = len(system.constants_array)

        precision = self.numba_precision

        if precision == float32:
            get_noise = get_noise_32
        else:
            get_noise = get_noise_64

        @cuda.jit(opt=True, lineinfo=True) # Lazy compilation allows for literalisation of shared mem params.
        def eulermaruyamakernel(xblocksize,
                                output,
                                grid_values,
                                grid_indices,
                                constants,
                                inits,
                                step_size,
                                duration,
                                output_fs,
                                filtercoeffs,
                                RNG,
                                noise_sigmas,
                                warmup_time):


            #Figure out where we are on the chip
            tx = int16(cuda.threadIdx.x)
            block_index = int32(cuda.blockIdx.x)
            l_param_set = int32(xblocksize * block_index + tx)


            # Don't try and do a run that hasn't been requested.
            if l_param_set >= len(grid_values):
                return

            l_step_size = precision(step_size)
            l_ds_rate = int32(round(1 / (output_fs * l_step_size)))
            l_n_outer = int32(round((duration / l_step_size) / l_ds_rate))            #samples per output value
            l_warmup = int32(warmup_time * output_fs)

            litzero = literally(zero)
            litstates = literally(nstates)
            litconstantslength = literally(constants_length)

            # Declare arrays to be kept in shared memory - very quick access.
            dynamic_mem = cuda.shared.array(litzero, dtype=precision)
            s_sums = dynamic_mem[:xblocksize*nstates]
            s_state = dynamic_mem[xblocksize*nstates: 2*xblocksize*nstates]

            # vectorize local variables used in integration for convenience
            l_dxdt = cuda.local.array(
                shape=(litstates),
                dtype=precision)

            l_noise = cuda.local.array(
                shape=(litstates),
                dtype=precision)

            l_constants = cuda.local.array(
                shape=(litconstantslength),
                dtype=precision)

            c_sigmas = cuda.const.array_like(noise_sigmas)
            c_RNG = cuda.const.array_like(RNG)
            c_filtercoefficients = cuda.const.array_like(filtercoeffs)
            for i in range(len(constants)):
                l_constants[i] = constants[i]

            for i, index in enumerate(grid_indices):
                l_constants[index] = grid_values[l_param_set, i]

            #Initialise w starting states
            for i in range(nstates):
                s_state[tx*nstates + i] = inits[i]
                s_sums[tx*nstates + i] = precision(0.0)

            l_dxdt[:] = precision(0.0)
            l_t = precision(0.0)

            #Loop through output samples, one iteration per output
            for i in range(l_n_outer + l_warmup):

                #Loop through euler solver steps - smaller step than output for numerical accuracy reasons
                for j in range(l_ds_rate):
                    l_t += l_step_size

                    # Generate noise value for each state
                    get_noise(l_noise,
                              c_sigmas,
                              l_param_set,
                              c_RNG)

                    #Get current filter coefficient for the downsampling filter
                    filtercoeff = c_filtercoefficients[j]

                    # Calculate derivative at sample
                    dxdtfunc(l_dxdt,
                             s_state[tx*litstates: tx*litstates + 5],
                             l_constants,
                             l_t)

                    #Forward-step state using euler-maruyama eq
                    #Add sum*filter coefficient to a running sum for downsampler
                    for k in range(litstates):
                        s_state[tx*litstates + k] += l_dxdt[k] * l_step_size + l_noise[k]
                        s_sums[tx*litstates + k] += s_state[tx*litstates + k] * filtercoeff

                #Start saving only after warmup period (to get past transient behaviour)
                if i > (l_warmup - 1):

                #Grab completed output sample
                    output[i-l_warmup, l_param_set, 0] = s_sums[tx*litstates + 0]
                    output[i-l_warmup, l_param_set, 1] = s_sums[tx*litstates + 1]
                    output[i-l_warmup, l_param_set, 2] = s_sums[tx*litstates + 2]
                    output[i-l_warmup, l_param_set, 3] = s_sums[tx*litstates + 3]
                    output[i-l_warmup, l_param_set, 4] = s_sums[tx*litstates + 4]

                #Reset filters to zero for another run
                s_sums[tx*litstates:tx*litstates + 5] = precision(0)


        self.integratorKernel = eulermaruyamakernel




#%% Test Code
if __name__ == "__main__":
    precision = np.float32

    # #Setting up grid of params to simulate with
    a_gains = np.asarray([i * 0.01 for i in range(-500, 500)], dtype=precision)
    b_params = np.asarray([i * 0.02 for i in range(-500, 500)], dtype=precision)
    grid_params = [(a, b) for a in a_gains for b in b_params]
    grid_labels = ['omega_forcing', 'a']
    step_size = precision(0.001)
    fs = precision(1)
    duration = precision(10)
    sys = thermal_cantilever_ax_b.diffeq_system(precision = precision)
    inits = np.asarray([1.0, 0, 1.0, 0, 1.0], dtype=precision)

    ODE = Solver(precision=precision)
    ODE.load_system(sys)
    solutions = ODE.run(sys,
                        inits,
                        duration,
                        step_size,
                        fs,
                        grid_labels,
                        grid_params,
                        warmup_time=precision(100.0))
    psd, f_psd = ODE.get_psd(solutions, fs)
    phase, f_phase = ODE.get_fft_phase(solutions, fs)
