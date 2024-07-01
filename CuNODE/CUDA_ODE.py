# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:31:43 2024

@author: cca79
"""

# -*- coding: utf-8 -*-
import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"

import numpy as np
from numba import cuda, from_dtype, literally
from numba import float32, float64, int32, int64, void
from numba.types import Literal, literal
from dxdt import dxdt
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64, xoroshiro128p_dtype
from scipy.signal import firwin
BLOCKSIZE_X = 8 * 32        #Number of param sets per block
NUM_STATES = 5
xoro_type = from_dtype(xoroshiro128p_dtype)
global0 = 0

class CUDA_ODE(object):

    def __init__(self,
                 nstates,
                 dtype = np.float64,
                 blocksize_x = 8 * 32
                 ):

        self.nstates = nstates
        self.blocksize_x = blocksize_x

        #Initialise other attributes/members with None and None-check arguments
        #for more intelligible error handling

    def build_kernel(self):
        self.dxdt = dxdt()
        dxdtfunc = self.dxdt.dxdtfunc
        getnoisefunc = self.dxdt.getnoisefunc
        clipfunc = self.dxdt.clipfunc
        
        global zero
        zero = 0
        global nstates
        nstates = self.nstates
        
        @cuda.jit(
            # void(int32,
            #             int32,
            #             float64[:,:,::1],
            #             float64[:,::1],
            #             float64[::1],
            #             float64[::1],
            #             float64,
            #             int64,
            #             float64[::1],
            #             xoro_type[::1],
            #             float64[::1],
            #             float64[::1]
            #             ),
                       opt=True)
        def naiive_euler_kernel(xblocksize,
                                numstates,
                                output, 
                                grid_params,
                                constants,
                                inits,
                                duration,
                                output_fs,
                                filtercoeffs,
                                RNG,
                                noise_sigmas,
                                ref):

            #Figure out where we are on the chip
            tx = cuda.threadIdx.x
            block_index = cuda.blockIdx.x
            l_param_set = xblocksize * block_index + tx

            # Don't try and do a run that hasn't been requested.
            if l_param_set >= len(grid_params):
                return

            # store local params in a register - probably done by the compiler regardless
            # but would like to keep memory locations front-of-mind
            l_step_size = constants[-1]
            l_ds_rate = int32(1 / (output_fs * l_step_size))
            l_n_outer = int32((duration / l_step_size) / l_ds_rate)
            # if tx == 0 
            #     from pdb import set_trace
            #     set_trace()
            litzero = literally(zero)
            litstates = literally(nstates)

            dynamic_mem = cuda.shared.array(litzero, dtype=float64)
            
            # Declare arrays to be kept in shared memory - very quick access.
            s_sums = dynamic_mem[:xblocksize*numstates - 1]
            s_state = dynamic_mem[xblocksize*numstates: 2*xblocksize*numstates -1]
         
            # vectorize local variables used in integration for convenience
            l_dxdt = cuda.local.array(
                shape=(litstates),
                dtype=float64)

            l_noise = cuda.local.array(
                shape=(litstates),
                dtype=float64)

            l_sigmas = cuda.local.array(
                shape=(litstates),
                dtype=float64)

            l_RNG = cuda.local.array(
                shape=(litstates),
                dtype=xoro_type)

            c_filtercoefficients = cuda.const.array_like(filtercoeffs)
            c_filtercoefficients = filtercoeffs
            c_constants = cuda.const.array_like(constants[:9])
            l_a = grid_params[l_param_set, 0]
            l_b = grid_params[l_param_set, 1]
            l_cliplevel = constants[10]
            l_rhat = constants[9]
            l_sigmas = noise_sigmas

            #Initialise w starting states
            for i in range(numstates):
                s_state[tx*numstates + i] = inits[i]
                l_RNG[i] = RNG[l_param_set * numstates + i]


            s_sums[:] = 0.0
            l_dxdt[:] = 0.0

            #Loop through output samples, one iteration per output
            for i in range(l_n_outer):

                #Loop through euler solver steps - smaller step than output for numerical accuracy reasons
                for j in range(l_ds_rate):

                    # Get absolute index of current sample
                    abs_sample = i*l_ds_rate + j

                    # Generate noise value for each state
                    getnoisefunc(l_noise,
                                 l_sigmas,
                                 l_RNG)


                    #Get current filter coefficient for the downsampling filter
                    filtercoeff = c_filtercoefficients[j]

                    # Clip reference signal (to simulate a hardware limitation)
                    ref_i = clipfunc(ref[abs_sample] * l_rhat, l_cliplevel)

                    # Clip control signal (to simulate a hardware limitation)
                    control = clipfunc(l_a * s_state[tx*numstates + 4] + l_b, l_cliplevel)

                    ref_i = float64(0.0)
                    control = float64(1)
                    
                    # Calculate derivative at sample
                    dxdtfunc(l_dxdt,
                             s_state[tx*numstates: tx*numstates + 5],
                             c_constants,
                             control,
                             ref_i)
                    
                    l_dxdt[:] = 1

                    for k in range(numstates):
                        s_state[tx*numstates + k] += l_dxdt[k] * l_step_size + l_noise[k]
                        s_sums[tx*numstates + k] += s_state[tx*numstates + k] * filtercoeff

                #Grab completed output sample
                output[i, l_param_set, 0] = s_sums[tx*xblocksize + 0]
                output[i, l_param_set, 1] = s_sums[tx*xblocksize + 1]
                output[i, l_param_set, 2] = s_sums[tx*numstates + 2]
                output[i, l_param_set, 3] = s_sums[tx*numstates + 3]
                output[i, l_param_set, 4] = s_sums[tx*numstates + 4]
                #Reset filters to zero for another run
                s_sums[tx*numstates:tx*numstates + 5] = 0
                
                
        # del globals()['zero']
        # del globals()['nstates']

        self.eulermaruyamakernel = naiive_euler_kernel
                

       
    
    """ Below are system-specific equations, that I don't yet know how to make
    user-enterable while still compiling with jit"""
   
    def euler_maruyama(self,
                       y0,
                       constants=None,
                       duration=None,
                       step_size=None,
                       output_fs=None,
                       grid_params=None,
                       noise_sigmas=None,
                       reference = None):
        if reference is not None:
            self.reference = reference
        if constants is not None:
            self.constants = constants
        if duration is not None:
            self.duration = duration
        if step_size is not None:
            self.step_size = step_size
        if noise_sigmas is not None:
            self.noise_sigmas = noise_sigmas
        if output_fs is not None:
            self.output_fs = output_fs
        if grid_params is not None:
            self.grid_params = grid_params
            
        self.dxdt = dxdt()
        dxdtfunc = self.dxdt.dxdtfunc
        getnoisefunc = self.dxdt.getnoisefunc
        clipfunc = self.dxdt.clipfunc
        
        self.output_array = cuda.pinned_array((int(self.output_fs * self.duration), 
                                               len(grid_params), 
                                               self.nstates),
                                               dtype=np.float64)
        self.filtercoefficients = firwin(int((1/self.step_size) / self.output_fs),
                                         self.output_fs/3,
                                         window='hann',
                                         pass_zero='lowpass',
                                         fs = self.output_fs) # This is pretty arbitrary, trying to keep things smooth in the pass band
        
        d_outputstates = cuda.to_device(self.output_array)
        d_constants = cuda.to_device(self.constants)
        d_reference = cuda.to_device(self.reference)
        d_gridparams = cuda.to_device(self.grid_params)
        d_filtercoefficients = cuda.to_device(self.filtercoefficients)
        d_inits = cuda.to_device(y0)
        
        
        random_seed = 1
        #Indexing note - because we will not add noise to all states, index this like a 2d (state, tx) array
        d_noise = create_xoroshiro128p_states(len(self.grid_params)*self.nstates, random_seed)
        d_noisesigmas = cuda.to_device(noise_sigmas)
        
            
        self.output_array[:, :, :] = 0
        BLOCKSPERGRID = int(max(1, np.ceil(len(self.grid_params) / self.blocksize_x)))
        self.eulermaruyamakernel[BLOCKSPERGRID, self.blocksize_x, 0, 49152](self.blocksize_x,
                                 self.nstates,
                                 d_outputstates,
                                 d_gridparams,
                                 d_constants,
                                 d_inits,
                                 self.duration,
                                 self.output_fs,
                                 d_filtercoefficients,
                                 d_noise,
                                 d_noisesigmas,
                                 d_reference)
        cuda.synchronize()
        d_outputstates.copy_to_host(self.output_array)
        self.output_array = np.asarray(self.output_array)

        
    
    
    
#%% Test Code
from system_parallelisation import constants, inits, noise_sigmas, fs, duration, step_size

# #Setting up grid of params to simulate with
a_gains = np.asarray([i * 0.01 for i in range(-1, 1)], dtype=np.float64)
b_params = np.asarray([i * 0.02 for i in range(-1, 1)], dtype=np.float64)
grid_params = [(a, b) for a in a_gains for b in b_params]
step_size = constants[-1]
reference = np.sin(np.linspace(0,duration - step_size, int(duration / step_size)), dtype=np.float64)

ODE = CUDA_ODE(5)
ODE.build_kernel()
ODE.euler_maruyama(inits,
                    constants,
                    duration,
                    step_size,
                    fs,
                    grid_params,
                    noise_sigmas,
                    reference)
