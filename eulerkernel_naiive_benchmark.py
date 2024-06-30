# -*- coding: utf-8 -*-

""" Working array layout

|        | 0                | 1           | 2         | 3          |   | 4                 | 5            | 6         | 7           |   | 8          | 9      | 10       | 11       |
|--------|------------------|-------------|-----------|------------|---|-------------------|--------------|-----------|-------------|---|------------|--------|----------|----------|
| FMA    | op1              | op2         | op3       | res        |   | op1               | op2          | op3       | res         |   | op1        | op2    | op3      | res      |
| 0      | omL*-step        | CTRL        | 0         | ctrl_a     |   | step              | VEL          | DISP      | **disp**    |   | DISP       | FIL    | DISP_SUM | disp_sum |
| 1      | FIL              | VEL         | VEL_SUM   | vel_sum    |   | delta*omega*-step | VEL          | RREF      | vel_a       |   | VELA       | 1      | VELB     | **vel**  |
| 2      | gamma*step       | CTRL2       | TEMP      | gctrl2     |   | beta*-step        | TEMP         | GCTRL2    | **temp**    |   | TEMP       | FIL    | TEMP_SUM | temp_sum |
| 3      | omL*step         | AXB(clip)   | CTRL      | ctrl_b     |   | 1                 | CTRLA        | CTRL_B    | **ctrl**    |   | CTRL       | FIL    | CTRL_SUM | ctrl_sum |
| 4      | rhat*sign*step   | REF         | 0         | rref       |   | step*sign         | VEL          | fhpf      | **HPF**     |   | HPF        | FIL    | HPF_SUM  | HPF_SUM  |
| 5      | (-)step*omega^2  | DISP        | VEL       | dispstep   |   | alpha*step        | TEMP         | DISP_STEP | vel_b       |   | CTRL       | CTRL   | 0        | ctrl2    |
| 6      | omH*-step        | HPF         | HPF       | fhpf       |   |                   |              |           |             |   | a          | HPF    | b        | axb      |
| 7      |                  |             |           |            |   |                   |              |           |             |   |            |        |          |          |

"""

import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"


from numba import cuda, float64, int64, void, int32, int16, uint64, from_dtype
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float64, xoroshiro128p_dtype
import numpy as np
from filter_coefficients import ds_100_filter
from time import time
from system_parallelisation import constants, inits, fs, duration, noise_sigmas


BLOCKSIZE_X = 512        #Threads per solver/operations per param set
                            # should be equal to (64kb - reference_chun/ SM allocation per thread maxish
NUM_STATES = 5
NUM_CONSTANTS = 14

# WORKINGARRAY_LENGTH = BLOCKSIZE_X
# WORKINGARRAY_WIDTH = 12 + 1 # 4 per math stage -op1, op2, op3, op4, then one extra to lessen bank conflicts (these will still occur between y-coord thread groups - one every two groups I think)
# OUTPUTBUFFER_DEPTH = 4

a_gains = np.asarray([i * 0.2 for i in range(-128, 128)], dtype=np.float64)
b_params = np.asarray([i * 0.1 for i in range(-128, 128)], dtype=np.float64)
grid_params = np.asarray([(a, b) for a in a_gains for b in b_params])

#TODO: Consider scaling.
filtercoefficients = np.asarray(ds_100_filter, dtype=np.float64)
step_size = constants[-1]


# working_array_3d = np.repeat(working_array_constants[:,:,np.newaxis], BLOCKSIZE_Y, axis=2)
# cuda.profile_start()

xoro_type = from_dtype(xoroshiro128p_dtype)

@cuda.jit(
                void(float64[:,:,:],
                float64[:,:],
                float64[:],
                float64[:],
                float64,
                int64,
                float64[:],
                xoro_type[:],
                float64[:],
                float64[:]
                ),
                opt=True)
def naiive_euler_kernel(output,
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

    if tx > len(grid_params):
        return

    block_index = cuda.blockIdx.x
    l_param_set = BLOCKSIZE_X * block_index + tx
    # store local params in a register - probably done by the compiler regardless
    # but would like to keep memory locations front-of-mind
    l_alpha, l_beta, l_gamma, l_delta, l_omega, l_omH, l_omL, l_pz_sign, l_hpf_sign, l_rhat, l_cliplevel, l_a, l_b, l_step_size = constants

    l_ds_rate = int32(1 / (output_fs * l_step_size))
    l_n_outer = int32((duration / l_step_size) / l_ds_rate)

    s_sums = cuda.shared.array(
        shape=(BLOCKSIZE_X,
               NUM_STATES),
        dtype=float64)

    s_state = cuda.shared.array(
        shape=(BLOCKSIZE_X,
               NUM_STATES),
        dtype=float64)

    for i in range(NUM_STATES):
        s_state[tx, i] = inits[i]

    s_sums[:] = 0

    l_a = grid_params[l_param_set, 0]
    l_b = grid_params[l_param_set, 1]

    c_filtercoefficients = cuda.const.array_like(filtercoeffs)
    c_filtercoefficients = filtercoeffs

    for i in range(l_n_outer):

        for j in range(l_ds_rate):
            abs_sample = i*l_ds_rate + j

            thread_noise0 = xoroshiro128p_normal_float64(RNG, tx + 0) * noise_sigmas[0]
            thread_noise1 = xoroshiro128p_normal_float64(RNG, tx + 1) * noise_sigmas[1]
            thread_noise2 = xoroshiro128p_normal_float64(RNG, tx + 2) * noise_sigmas[2]
            thread_noise3 = xoroshiro128p_normal_float64(RNG, tx + 3) * noise_sigmas[3]
            thread_noise4 = xoroshiro128p_normal_float64(RNG, tx + 4) * noise_sigmas[4]
            filtercoeff = c_filtercoefficients[j]

            ref_i = ref[abs_sample] * l_rhat
            if ref_i < l_cliplevel and ref_i > -l_cliplevel:
                clipped_ref = ref_i
            if ref_i > l_cliplevel:
                clipped_ref = l_cliplevel
            elif ref_i < -l_cliplevel:
                clipped_ref = -l_cliplevel

            control = l_a * s_state[tx, 4] + l_b
            if control < l_cliplevel and control > -l_cliplevel:
                clipped_control = control
            if control > l_cliplevel:
                clipped_control = l_cliplevel
            elif control < -l_cliplevel:
                clipped_control = -l_cliplevel

            dxdt0 = s_state[tx,0] + s_state[tx, 1]
            dxdt1 = (-s_state[tx,0] + l_delta*s_state[tx, 1] + l_alpha * s_state[tx, 2] + l_pz_sign * clipped_ref)
            dxdt2 = (-l_beta * s_state[tx, 2] + l_gamma * s_state[tx, 3]**2)
            dxdt3 = (-l_omL*s_state[tx, 3] + l_omL*clipped_control)
            dxdt4 = (-l_omH * s_state[tx,4] + l_hpf_sign * s_state[tx, 1])

            s_state[tx,0] += dxdt0*l_step_size + thread_noise0
            s_state[tx,1] += dxdt1*l_step_size + thread_noise1
            s_state[tx,2] += dxdt2*l_step_size + thread_noise2
            s_state[tx,3] += dxdt3*l_step_size + thread_noise3
            s_state[tx,4] += dxdt4*l_step_size + thread_noise4


            s_sums[tx, 0] += s_state[tx,0] * filtercoeff
            s_sums[tx, 1] += s_state[tx,1] * filtercoeff
            s_sums[tx, 2] += s_state[tx,2] * filtercoeff
            s_sums[tx, 3] += s_state[tx,3] * filtercoeff
            s_sums[tx, 4] += s_state[tx,4] * filtercoeff


        output[i, l_param_set, 0] = s_sums[tx,0]
        output[i, l_param_set, 1] = s_sums[tx,1]
        output[i, l_param_set, 2] = s_sums[tx,2]
        output[i, l_param_set, 3] = s_sums[tx,3]
        output[i, l_param_set, 4] = s_sums[tx,4]


        s_sums[tx, 0] = 0
        s_sums[tx, 1] = 0
        s_sums[tx, 2] = 0
        s_sums[tx, 3] = 0
        s_sums[tx, 4] = 0

durations = np.asarray([1, 10, 100, 1000, 10000])
num_parallel = np.asarray([32, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
naiive_timings = np.zeros((len(durations), len(num_parallel)))
for j, duration in enumerate(durations):
    for k, parallels in enumerate(num_parallel):
        sampled_combinations = np.random.choice(len(grid_params), size=parallels, replace=(parallels > len(grid_params)))
        sampled_combinations = grid_params[sampled_combinations]

        if (fs * duration * (parallels * NUM_STATES + 1)*8) > (7168 * 1024**2):
            naiive_timings[j, k] = 0
        else:
            NUMTHREADS = len(sampled_combinations)
            BLOCKSPERGRID = max(1, np.ceil(len(sampled_combinations) / BLOCKSIZE_X))

            outputstates = cuda.pinned_array((int(fs * duration), parallels, NUM_STATES), dtype=np.float64)
            outputstates[:, :, :] = 0
            reference = np.sin(np.linspace(0,duration - step_size, int(duration / step_size)), dtype=np.float64)

            d_outputstates = cuda.to_device(outputstates)
            d_constants = cuda.to_device(constants)
            d_reference = cuda.to_device(reference)
            d_gridparams = cuda.to_device(sampled_combinations)
            d_filtercoefficients = cuda.to_device(filtercoefficients)
            d_inits = cuda.to_device(inits)

            random_seed = 1
            d_noise = create_xoroshiro128p_states(NUMTHREADS, random_seed)
            d_noisesigmas = cuda.to_device(noise_sigmas)




            # Test timing loop
            start = cuda.event()
            end = cuda.event()
            _s = 0
            _e = 0
            timing_nb = 0
            timing_nb_wall = 0
            for i in range(4):
                if i > 0:
                    start.record()
                    _s = time()

                naiive_euler_kernel[BLOCKSPERGRID,
                                        BLOCKSIZE_X](
                    d_outputstates,
                    d_gridparams,
                    d_constants,
                    d_inits,
                    duration,
                    fs,
                    d_filtercoefficients,
                    d_noise,
                    d_noisesigmas,
                    d_reference)

                cuda.synchronize()
                # print("{} loops done".format(i))
                d_outputstates.copy_to_host(outputstates)
                outputstates = np.asarray(outputstates)
                if i > 0:
                    end.record()
                    end.synchronize()
                    _e = time()
                    timing_nb += cuda.event_elapsed_time(start, end)
                    timing_nb_wall += (_e - _s)

            naiive_timings[j, k] = timing_nb_wall / 3 * 1000
            print('run', j*len(num_parallel) + k, ' out of ', len(num_parallel) * len(durations))
            # print('numba events:', timing_nb / 3, 'ms')
            print('numba wall  :', timing_nb_wall / 3 * 1000, 'ms')