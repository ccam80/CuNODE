# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:08:37 2024

@author: cca79
"""
import numpy as np

class modelController():
    """Helper class that takes a set of requested variables, slices, and scales,
    sorts them and fetches the data to arrange a request that makes sense to the
    plotter. """

    def __init__(self):
        pass
        #What state does this thing need?
        self.plotter_request = {'axes': '3d',
                                'type': 'surface',
                                'dataformat': 'mesh',
                                'xdata': np.zeros(1),
                                'ydata': np.zeros(1),
                                'zdata': np.zeros(1),
                                'xlabel': '',
                                'ylabel': '',
                                'zlabel': '',
                                'xscale': '',
                                'yscale': '',
                                'zscale': ''}

        self.plotController_return = {'xbounds': [0, 100],
                                      'ybounds': [0,100],
                                      'zbounds': [0, 100],}

        self.state_bounds = {'time': [0,1000],
                             'psd_freq': [0,1000],
                             'fft_freq': [0,1000]}

        self.solutions = np.zeros((1,1,1))
        self.psd = np.zeros(1,1)
        self.fft_phase = np.zeros(1,1)

    def load_solver(self, solver_class, precision=np.float64):
        self.solver = solver_class(precision = precision)

    def load_system(self, system_class, precision=np.float64):
        self.system = system_class(precision = precision)
        self.solver.build_kernel(self.system)

    def run_solver(self, request, constants):
        #self.system.system_constants =
        #self.system.y0
        self.solutions = self.solver.run() # get params in here else she won't go... will she?
        self.psd, self.f_psd = self.solver.get_psd()
        self.phase, self.f_phase = self.solver.get_fft_phase()

    def trim_data(self, data, slice_ends):
        lower_bound = slice_ends[0]
        upper_bound = slice_ends[1]
        return data[data >= lower_bound and data <= upper_bound]


    def interpret(self, request):
        if request['plot_type'] == 'grid3d':
            self.plot_grid3d(request)
        elif request['plot_type'] == 'time3d':
            self.plot_time3d(request)
        elif request['plot_type'] == 'spec3d':
            self.plot_spec3d(request)
        elif request['plot_type'] == 'singlestate':
            if request['singleplotstyle'] == 'Spectrogram':
                self.plot_spec2d(request)
            elif request['singleplotstyle'] == 'Time-domain':
                self.plot_time2d(request)
            elif request['singleplotstyle'] == '2D Phase Diagram':
                self.plot_phase2d(request)
            elif request['singleplotstyle'] == '3D Phase Diagram':
                self.plot_phase3d(request)

    def plot_grid3d(self, request):
        self.plotter_request['axes'] = '3d'
        self.plotter_request['type'] = 'surface'
        self.plotter_request['dataformat'] = 'vector'
