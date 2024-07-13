# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 20:31:43 2024

@author: cca79
"""

# -*- coding: utf-8 -*-
import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"

import numpy as np
from cupy import asarray, ascontiguousarray, get_default_memory_pool
from numba import cuda, from_dtype, literally
from numba import float64, float32, int32, int64, void, int8, int16
from diffeq_system import diffeq_system
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32, xoroshiro128p_dtype
from cupyx.scipy.signal import firwin, welch
from cupyx.scipy.fft import rfftfreq
from _utils import timing


xoro_type = from_dtype(xoroshiro128p_dtype)

# global0 = 0


# shared_indices = np.asarray([[1, 10, 10, 10, 10],
#                              [0,10,1,2,10],
#                              [3,3,2,10,10],
#                              [3,10,5,10,10],
#                              [4,10,1,10,10],
#                              [4,10,10,10,10],
#                              [10,10,10,10,10]], 
#                             dtype=np.int8)

# consts = np.asarray([[1,0,0,0],
#                      [0,-constants[2],constants[0],constants[7]],
#                      [constants[1],-constants[5],0,0],
#                      [-constants[6],constants[6],0,0],
#                      [-constants[5],constants[8],0,0],
#                      [constants[11],constants[12],0,0],
#                      [constants[9], 0, 0, 0]])
# c_shared_indices = cuda.const.array_like(shared_indices[tx])




                  
class CUDA_ODE(object):

    def __init__(self,
                 diffeq_sys,
                 precision=np.float32                 
                 ):
        self.precision = precision
        self.numba_precision = from_dtype(precision)
        self.system = diffeq_sys

        #Initialise other attributes/members with None and None-check arguments
        #for more intelligible error handling

    def build_kernel(self):
        
        global zero
        zero = 0
        global nstates
        nstates = 7
        global constants_length
        constants_length = len(self.system.constants_array)

        precision = self.numba_precision
                    
        @cuda.jit(opt=True,
                    # debug=True)
                   lineinfo=True,
                   max_registers = 54) # Lazy compilation allows for literalisation of shared mem params.
        def naiive_euler_kernel(xblocksize,
                                yblocksize,
                                output, 
                                grid_values,
                                grid_indices,
                                constants,
                                constants_array,
                                d_state_indices,
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
            ty = int16(cuda.threadIdx.y)
            block_index = int32(cuda.blockIdx.x)
            l_param_set = int32(yblocksize * block_index + ty)

        
            # Don't try and do a run that hasn't been requested.
            if l_param_set >= len(grid_values):
                return
    
            l_step_size = precision(step_size)
            l_ds_rate = int32(round(1 / (output_fs * l_step_size)))
            l_n_outer = int32(round((duration / l_step_size) / l_ds_rate))            #samples per output value
            l_warmup = int32(round(warmup_time * output_fs))
            l_cliplevel = precision(constants[10])
            litzero = literally(zero)
            litstates = literally(nstates)
            litconstantslength = literally(constants_length)
            
            for i, index in enumerate(grid_indices):
                constants[index] = grid_values[l_param_set, i]
            
            #Hard coded for a, b in this specific test, would need to automate
            
            #10 means don't care, should'nt be hit due to selps.
            s_shared_indices = cuda.shared.array(shape=(7,5),
                                                 dtype=int8)
                    
            # Declare arrays to be kept in shared memory - very quick access.
            dynamic_mem = cuda.shared.array(litzero, dtype=precision)
            s_sums = dynamic_mem[:yblocksize*(xblocksize)]
            s_state = dynamic_mem[yblocksize*(xblocksize): 2*(yblocksize*xblocksize)]
            
            s_constants = cuda.shared.array(shape=(7,4),
                                           dtype=precision)
    
            
            for i in range(4):
                s_constants[tx, i] = constants_array[tx,i]
            #Hard coded for a, b in this specific test, would need to automate
            if tx == 5:
                s_constants[tx, 0] = grid_values[l_param_set,0]
                s_constants[tx, 1] = grid_values[l_param_set,1]
                
            for i in range(5):
                s_shared_indices[tx, i] = d_state_indices[tx,i]

            
            c_sigmas = cuda.const.array_like(noise_sigmas)
            c_RNG = cuda.const.array_like(RNG)
            c_filtercoefficients = cuda.const.array_like(filtercoeffs)
            
            
            #Initialise w starting states
            for i in range(nstates):
                s_state[ty*nstates + tx] = inits[tx]
                s_sums[ty*nstates + tx] = precision(0.0)

            l_t = precision(0.0)
            l_ctrl = precision(0.0)
            l_ref_samp = int8(7)
            s_ref = cuda.shared.array(
                shape=(litstates),
                dtype=precision)
            l_noise = precision(0.0)
            
            #Loop through output samples, one iteration per output
            for i in range(l_n_outer + l_warmup):

                #Loop through euler solver steps - smaller step than output for numerical accuracy reasons
                for j in range(l_ds_rate):
                    
                    if l_ref_samp == 7:
                        # s_ref[tx] = cuda.libdevice.cos(precision(constants[13]*(l_t + tx)))
                        s_ref[tx] = np.cos(precision(constants[13]*(l_t + tx)))           #Use np fro debug
                        l_ref_samp = int8(0)
                        
                    l_t += l_step_size
                    
                    # Generate noise value for each state
                    if c_sigmas[tx] == precision(0.0):
                        l_noise == precision(0.0)
                    else:
                        l_noise = xoroshiro128p_normal_float32(RNG, ty) * c_sigmas[tx]
                    
                    #Get current filter coefficient for the downsampling filter
                    filtercoeff = c_filtercoefficients[j]

                    cuda.syncthreads()
                    # Calculate derivative at sample
                    s_state[ty*nstates + tx] += ((cuda.selp(tx==6, s_ref[l_ref_samp], s_state[s_shared_indices[tx, 0]]) 
                                                  * cuda.selp((tx==1), s_state[s_shared_indices[tx, 1]], precision(1.0)) * s_constants[tx, 0] + 
                                                  cuda.selp((tx < 1 or tx > 4), precision(1.0), s_state[s_shared_indices[tx, 2]]) * s_constants[tx, 1] + 
                                                  cuda.selp(tx==1, s_state[s_shared_indices[tx, 3]], precision(0.0)) * s_constants[tx, 2] +
                                                  cuda.selp(tx==1, s_state[s_shared_indices[tx, 4]], precision(0.0)) * s_constants[tx, 3]) * l_step_size + l_noise)
                    # cuda.syncthreads()
                    if tx==5:
                        l_ctrl = s_state[ty*nstates+ int32(5)]
                 
                        s_state[ty*nstates + int32(5)] = cuda.selp(l_ctrl > l_cliplevel, l_cliplevel, l_ctrl)
                    
                        l_ctrl = s_state[ty*nstates+int32(5)] 
                        s_state[ty*nstates + int32(5)] = cuda.selp(l_ctrl < -l_cliplevel, -l_cliplevel, l_ctrl)

        
                    s_sums[ty*litstates + tx] += s_state[ty*litstates + tx] * filtercoeff
                        
                    l_ref_samp += int8(1)
                    
                #Start saving only after warmup period (to get past transient behaviour)
                if i > (l_warmup - int8(1)):
                    
                #Grab completed output sample
                    if tx < 5:    
                        output[i-l_warmup, l_param_set, tx] = s_sums[ty*litstates + tx]
                    
                #Reset filters to zero for another run
                s_sums[ty*litstates+tx] = precision(0.0)
                
                
        self.eulermaruyamakernel = naiive_euler_kernel
    
    def euler_maruyama(self,
                       y0,             
                       duration,
                       step_size,
                       output_fs,
                       grid_labels,
                       grid_values,
                       noise_seed=1,
                       blocksize_x=7,
                       blocksize_y=9,
                       warmup_time=0.0):
     

        self.fs = output_fs
        self.duration = self.precision(duration)
        self.step_size = self.precision(step_size)
        
        self.output_array = cuda.pinned_array((int(output_fs * duration), 
                                               len(grid_values), 
                                               self.system.num_states),
                                               dtype=self.precision)
        self.output_array[:, :, :] = 0
        constants = self.system.constants_array
        
        const_array = np.asarray([[1,0,0,0],
                                [-1,-constants[2],constants[0],constants[7]],
                                [constants[1],-constants[5],0,0],
                                [-constants[6],constants[6],0,0],
                                [-constants[5],constants[8],0,0],
                                [constants[11],constants[12],0,0],
                                [constants[9], 0, 0, 0]], dtype=self.precision
                                 )
        
                
        state_index_array = np.asarray([[1,0,0,0,0],
                                        [0,0,1,2,0],
                                        [3,3,2,0,0],
                                        [3,0,5,0,0],
                                        [4,0,1,0,0],
                                        [4,0,0,0,0],
                                        [0,0,0,0,0]], 
                                       dtype=np.int8
                                       )
        
        
        grid_indices = np.zeros(len(grid_labels), dtype=np.int32)
        
        for index, label in enumerate(grid_labels):
            grid_indices[index] = self.system.constant_indices[label]
        
        d_filtercoefficients = firwin(int(1 / (step_size * output_fs)),
                                      output_fs/2.01,
                                      window='hann',
                                      pass_zero='lowpass',
                                      fs = output_fs)
        d_filtercoefficients = asarray(d_filtercoefficients, dtype=self.precision)
        
        d_outputstates = asarray(self.output_array, dtype=self.precision)
        d_constants = asarray(self.system.constants_array, dtype=self.precision)
        d_constants_array = asarray(const_array, dtype=self.precision)
        d_gridvalues = asarray(grid_values, dtype=self.precision)
        d_gridindices = asarray(grid_indices)
        d_inits = asarray(y0, dtype=self.precision)
        d_state_indices = asarray(state_index_array)
        
        #one per system
        d_noise = create_xoroshiro128p_states(len(grid_values), noise_seed)
        d_noisesigmas = asarray(self.system.noise_sigmas, dtype=self.precision)
        
        #total threads / threads per block (or 1 if 1 is greater) 
        BLOCKSPERGRID = int(max(1, np.ceil(len(grid_values) / blocksize_y)))
        
        #Size of shared allocation (n states per thread per block, times 2 (for sums) x 8 for float32)
        if self.numba_precision == float32:
            bytes_per_val = 4
        else:
            bytes_per_val = 8
        dynamic_sharedmem = 2 * blocksize_x * blocksize_y * bytes_per_val
        
        cuda.profile_start()
        self.eulermaruyamakernel[BLOCKSPERGRID, (blocksize_x,blocksize_y), 
                                 0, dynamic_sharedmem](
                                     blocksize_x,
                                     blocksize_y,
                                     d_outputstates,
                                     d_gridvalues,
                                     d_gridindices,
                                     d_constants,
                                     d_constants_array,
                                     d_state_indices,
                                     d_inits,
                                     step_size,
                                     duration,
                                     output_fs,
                                     d_filtercoefficients,
                                     d_noise,
                                     d_noisesigmas,
                                     self.precision(warmup_time))
        cuda.synchronize()
        cuda.profile_stop()

        
        self.output_array = np.ascontiguousarray(d_outputstates.get().T)
        

    def get_fft(self, window='hann'):
        
        nperseg = int(self.output_array.shape[2] / 4)
        noverlap = int(nperseg/2)
        nfft = nperseg*2

        mem_remaining = self.get_free_memory()
        
        
        num_segs = (self.output_array.shape[2] - noverlap) // (nperseg - noverlap)
        segsize = nperseg * self.output_array.shape[0] * self.output_array.shape[1] * 16
        
        total_mem_for_operation = segsize * num_segs
        total_chunks = int(np.ceil(total_mem_for_operation / mem_remaining))
        
        self.fft_array = np.zeros((self.output_array.shape[0], 
                                   self.output_array.shape[1], 
                                   int(nfft/2) + 1), 
                                  dtype=np.complex128)
        
        for i in range(total_chunks):
            chunksize = int(np.ceil(self.output_array.shape[1] / total_chunks))
            index = chunksize * i
            
            self.f, self.temp_fft_array = welch(asarray(self.output_array[:,index:index + chunksize,:], order='C'),
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
        

    def get_free_memory(self):
        total_mem = cuda.current_context().get_memory_info()[1] - 1024**2  # Leave 1G for misc overhead
        allocated_mem = get_default_memory_pool().used_bytes()
        
        return total_mem - allocated_mem

    
#%% Test Code
if __name__ == "__main__":
    precision = np.float64
    # #Setting up grid of params to simulate with
    a_gains = np.asarray([i * 0.01 for i in range(-1, 1)], dtype=precision)
    b_params = np.asarray([i * 0.02 for i in range(-1, 1)], dtype=precision)
    grid_params = [(a, b) for a in a_gains for b in b_params]
    grid_labels = ['a', 'b']
    step_size = precision(0.001)
    fs = 1
    duration = precision(1000)
    sys = diffeq_system(precision=precision)
    sys.noise_sigmas = np.asarray([0, 0, 0, 0, 0, 0, 0], dtype=precision)
    inits = np.asarray([1.0, 0, 1.0, 0, 1.0, 0.0, 0.0], dtype=precision)
    
    ODE = CUDA_ODE(sys)
    ODE.build_kernel()
    ODE.euler_maruyama(inits,
                       duration,
                       step_size,
                       fs,
                       grid_labels,
                       grid_params,
                       warmup_time=1000.0)
    # ODE.get_fft()
