# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:08:37 2024

@author: cca79
"""
import numpy as np
from pubsub import PubSub
from _utils import get_readonly_view


class modelController():
    """Helper class that takes a set of requested variables, slices, and scales,
    sorts them and fetches the data to arrange a request that makes sense to the
    plotter. """

    def __init__(self, messaging_service=None):
        pass
        self.system = None
        self.solver = None

        self.solutions = np.zeros((1,1,1))
        self.psd = np.zeros(1)
        self.fft_phase = np.zeros(1)
        if messaging_service:
            self.register_messaging_service(messaging_service)
        else:
            self.messenger = None

    def register_messaging_service(self, messaging_service):
        """Connect a message passing service to communicate between widgets

        Args:
            messaging_service(class): A class with publish, subscribe, unsubscribe methods"""

        messaging_service.subscribe("precision", self.update_precision)
        messaging_service.subscribe("model_request", self.serve_data)

        self.messenger = messaging_service

    def publish(self, topic, data):
        self.messenger.publish(topic, data)

    def update_precision(self, precision):
        self.precision = precision

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

    def rms(self, data, axis=-1):
        return np.sqrt(np.mean(np.square(data), axis=axis))

    def serve_data(self, request):
        """Return a 1D vector of results - sliced either by optional grid,
        time, or frequency indices. If no indices are given, or incorrect ones
        for the data type, the data will be reshaped to a very long vector, silently."""

        variable = request['variable']
        states = request['state_indices']
        param_indices = request['param_indices']
        time_or_freq_indices = request['time_or_freq_indices']
        aggregation = request['aggregation']

        if variable == 'Amplitude':
            selected_data = self.solutions
        elif variable == 'PSD':
            selected_data = self.psd
        elif variable == 'FFT Phase':
            selected_data = self.phase
        selected_data = get_readonly_view(selected_data)
        try:
            selected_data = np.squeeze(selected_data[states, param_indices, time_or_freq_indices])
        except:
            pass
        if aggregation == 'RMS':
            selected_data = self.rms(selected_data)

        self.publish("requested_data", selected_data)