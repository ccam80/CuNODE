# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:08:37 2024

@author: cca79
"""
import numpy as np
from _utils import get_readonly_view


class modelController():
    """Helper class that takes a set of requested variables, slices, and scales,
    sorts them and fetches the data to arrange a request that makes sense to the
    plotter. """

    def __init__(self, messaging_service):
        pass
        self.system = None
        self.solver = None

        self.solutions = np.zeros((1,1,1))
        self.psd = np.zeros(1)
        self.fft_phase = np.zeros(1)
        self.messenger = None

    def register_messaging_service(self, messaging_service):
        """Connect a message passing service to communicate between widgets

        Args:
            messaging_service(class): A class with publish, subscribe, unsubscribe methods"""
        self.messenger = messaging_service

    def load_solver(self, solver_class, precision=np.float64):
        self.solver = solver_class(precision = precision)

    def load_system(self, system):
        self.system = system
        self.solver.build_kernel(self.system)

    def update_system_parameters(self, params_dict):
        self.system.update_constants(params_dict)

    def run_solver(self, system, request, sweeps,
                   noise_seed=1, blocksize_x=64):

        y0 = request['y0']
        duration = request['duration']
        output_fs = request['fs']
        step_size = request['dt']
        warmup = request['warmup']
        grid_labels = sweeps['vars']
        grid_values = sweeps['values']

        self.solutions = self.solver.run(system,
                                         y0,
                                         duration,
                                         step_size,
                                         output_fs,
                                         grid_labels,
                                         grid_values,
                                         noise_seed,
                                         blocksize_x,
                                         warmup)

        self.psd, _ = self.solver.get_psd(self.solutions, output_fs)
        self.phase, _ = self.solver.get_fft_phase(self.solutions, output_fs)
        # self.t = np.linspace(0,  duration -  1/output_fs, int( duration * output_fs))


    def trim_data(self, data, slice_ends):
        lower_bound = slice_ends[0]
        upper_bound = slice_ends[1]
        return data[data >= lower_bound and data <= upper_bound]


    def get_data(self, data_type, state,
                 grid_indices=None,
                 freq_indices=None,
                 time_indices=None):
        """Return a 1D vector of results - sliced either by optional grid,
        time, or frequency indices. If no indices are given, or incorrect ones
        for the data type, the data will be reshaped to a very long vector, silently."""

        # See if the data type is a state label, and get that state if so. If the
        #data type is not in the dict, carry on to find it in the non-system-defined
        # types.
        try:
            state = self.system.state_labels[data_type]
            data = self.solutions[state, grid_indices, time_indices]
        except:

            if data_type == 'psd':
                data = self.psd[state, grid_indices, freq_indices]
            elif data_type == 'phase':
                data = self.phase[state, grid_indices, freq_indices]
            elif data_type == 'amplitude':
                data = self.solutions[state, grid_indices, time_indices]

            elif data_type == 'rms':
                grid_length = self.solutions.shape[1]
                data = np.zeros(grid_length)
                for i in range(grid_length):
                    solution_vector = self.solutions[state,i,:]
                    data[i] = np.sqrt(np.mean(solution_vector**2))
                data = data[grid_indices, freq_indices]

        data = data.reshape(-1,1)
        data.flags.writeable = False

        return data