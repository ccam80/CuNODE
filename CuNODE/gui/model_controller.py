# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:08:37 2024

@author: cca79
"""
import numpy as np

class plot_request_interpreter():
    """Helper class that takes a set of requested variables, slices, and scales,
    sorts them and fetches the data to arrange a request that makes sense to the
    plotter. """

    def __init__(self):
        pass
        #What state does this thing need?
        self.plotter_request = {'axes': '3d',
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

    def self.plot_grid3d(self, request):
