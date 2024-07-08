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
from cupy import asarray, ascontiguousarray
from numba import cuda, from_dtype, literally
from numba import float32, float64, int32, int64, void
from diffeq_system import diffeq_system
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64, xoroshiro128p_dtype
from cupyx.scipy.signal import firwin, welch
from cupyx.scipy.fft import rfftfreq
from _utils import get_noise, timing


xoro_type = from_dtype(xoroshiro128p_dtype)

# global0 = 0

class CUDA_ODE(object):

    def __init__(self,
                 diffeq_sys                 
                 ):

        self.system = diffeq_sys

        #Initialise other attributes/members with None and None-check arguments
        #for more intelligible error handling

    def build_kernel(self):
        dxdtfunc = self.system.dxdtfunc
        
        global zero
        zero = 0
        global nstates
        nstates = self.system.num_states
        global constants_length
        constants_length = len(self.system.constants_array)
        
        
                    
        @cuda.jit(opt=True) # Lazy compilation allows for literalisation of shared mem params.
        def naiive_euler_kernel(xblocksize,
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
            tx = cuda.threadIdx.x
            block_index = cuda.blockIdx.x
            l_param_set = xblocksize * block_index + tx

        
            # Don't try and do a run that hasn't been requested.
            if l_param_set >= len(grid_values):
                return
    
            l_step_size = step_size
            l_ds_rate = int32(1 / (output_fs * l_step_size))
            l_n_outer = int32((duration / l_step_size) / l_ds_rate)            #samples per output value
            l_warmup = int32(warmup_time * output_fs)

            litzero = literally(zero)
            litstates = literally(nstates)
            litconstantslength = literally(constants_length)
            
            # Declare arrays to be kept in shared memory - very quick access.
            dynamic_mem = cuda.shared.array(litzero, dtype=float64)
            s_sums = dynamic_mem[:xblocksize*nstates]
            s_state = dynamic_mem[xblocksize*nstates: 2*xblocksize*nstates]
         
            # vectorize local variables used in integration for convenience
            l_dxdt = cuda.local.array(
                shape=(litstates),
                dtype=float64)

            l_noise = cuda.local.array(
                shape=(litstates),
                dtype=float64)
            
            l_constants = cuda.local.array(
                shape=(litconstantslength),
                dtype=float64)

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
                s_sums[tx*nstates + i] = 0.0

            l_dxdt[:] = 0.0
            l_t = float64(0.0)
            
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
                s_sums[tx*litstates:tx*litstates + 5] = 0
                
                
        self.eulermaruyamakernel = naiive_euler_kernel
                
   
    def euler_maruyama(self,
                       y0,             
                       duration,
                       step_size,
                       output_fs,
                       grid_labels,
                       grid_values,
                       noise_seed=1,
                       blocksize_x=64,
                       warmup_time=0.0):
     

        self.fs = output_fs
        self.duration = duration
        self.step_size = step_size
        
        self.output_array = cuda.pinned_array((int(output_fs * duration), 
                                               len(grid_values), 
                                               self.system.num_states),
                                               dtype=np.float64)
        self.output_array[:, :, :] = 0
        
        grid_indices = np.zeros(len(grid_labels), dtype=np.int32)
        
        for index, label in enumerate(grid_labels):
            grid_indices[index] = self.system.constant_indices[label]
        
        d_filtercoefficients = firwin(int(1 / (step_size * output_fs)),
                                      output_fs/2.01,
                                      window='hann',
                                      pass_zero='lowpass',
                                      fs = output_fs)
        
        d_outputstates = asarray(self.output_array)
        d_constants = asarray(self.system.constants_array)
        d_gridvalues = asarray(grid_values)
        d_gridindices = asarray(grid_indices)
        d_inits = asarray(y0)
        
        #one per system
        d_noise = create_xoroshiro128p_states(len(grid_values), noise_seed)
        d_noisesigmas = asarray(self.system.noise_sigmas)
        
        #total threads / threads per block (or 1 if 1 is greater) 
        BLOCKSPERGRID = int(max(1, np.ceil(len(grid_values) / blocksize_x)))
        
        #Size of shared allocation (n states per thread per block, times 2 (for sums) x 8 for float64)
        dynamic_sharedmem = 2 * blocksize_x * self.system.num_states * 8
        

        self.eulermaruyamakernel[BLOCKSPERGRID, blocksize_x, 
                                 0, dynamic_sharedmem](
                                     blocksize_x,
                                     d_outputstates,
                                     d_gridvalues,
                                     d_gridindices,
                                     d_constants,
                                     d_inits,
                                     step_size,
                                     duration,
                                     output_fs,
                                     d_filtercoefficients,
                                     d_noise,
                                     d_noisesigmas,
                                     warmup_time)
        cuda.synchronize()
        
        self.output_array = d_outputstates

        self.time_friendly_array = np.ascontiguousarray(self.output_array.T.get())

    def get_fft(self, window='hann'):
        
        nperseg = int(self.time_friendly_array.shape[2] / 4)
        noverlap = int(nperseg/2)
        nfft = nperseg*2

        # Manage space remaining in VRAM, this will get slow if we have little enough memory left.
        #If this becomes a sore point we wil need to delete the GPU array holding the output and write
        # back/forth between fft runs.
        total_mem = cuda.current_context().get_memory_info()[1]
        used_bytes = self.output_array.nbytes
        mem_remaining = total_mem - used_bytes
        
        num_segs = self.output_array.shape[0] / (nperseg - noverlap) - 1
        
        segsize = nperseg * self.output_array.shape[2] * self.output_array.shape[1] * 16
        
        total_mem_for_operation = segsize * num_segs
        total_chunks = int(np.ceil(total_mem_for_operation / mem_remaining) + 1)
        
        self.fft_array = np.zeros((self.output_array.shape[2], 
                                   self.output_array.shape[1], 
                                   int(nfft/2) + 1), 
                                  dtype=np.complex128)
        
        for i in range(total_chunks):
            chunksize = int(np.ceil(self.output_array.shape[1] / total_chunks))
            index = chunksize * i
            
            self.f, self.temp_fft_array = welch(ascontiguousarray(self.output_array[:,index:index + chunksize,:].T),
                                   fs=self.fs*2*np.pi,
                                       window=window,
                                       nperseg=nperseg,
                                       nfft=nfft,
                                       detrend='linear',
                                       scaling='spectrum',
                                       axis=2)
            self.fft_array[:, index:index+chunksize, :] = self.temp_fft_array.get()
            self.f = self.f.get()
        # self.f = self.f.get()
        # self.fft_array = self.fft_array.get()
        # self.f = rfftfreq(self.time_friendly_array.shape[2], d=1/(self.fs*2*np.pi)).get()
        

    
    
#%% Test Code
if __name__ == "__main__":
    
    # #Setting up grid of params to simulate with
    a_gains = np.asarray([i * 0.01 for i in range(-128, 128)], dtype=np.float64)
    b_params = np.asarray([i * 0.02 for i in range(-64, 64)], dtype=np.float64)
    grid_params = [(a, b) for a in a_gains for b in b_params]
    grid_labels = ['omega_forcing', 'a']
    step_size = 0.001
    fs = 10
    duration = np.float64(100)
    sys = diffeq_system()
    inits = np.asarray([1.0, 0, 1.0, 0, 1.0], dtype=np.float64)
    
    ODE = CUDA_ODE(sys)
    ODE.build_kernel()
    ODE.euler_maruyama(inits,
                       duration,
                       step_size,
                       fs,
                       grid_labels,
                       grid_params,
                       warmup_time=200.0)
    ODE.get_fft(fs=10.0)
