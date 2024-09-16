# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 20:30:54 2024

@author: cca79
"""
import numpy as np
from _utils import round_sf
from time import sleep
from threading import Event, Thread

class plotstyle_interfaces(dict):
    def __init__(self, precision=np.float64, sig_figs=4, messaging_service=None):
        # Initialize the dictionary with your pre-filled keys
        super().__init__({
            'grid3d': grid3d_style(precision, sig_figs, messaging_service),
            'time3d': time3d_style(precision, sig_figs, messaging_service),
            'spec3d': spec3d_style(precision, sig_figs, messaging_service),
            'singlestate_style': singlestate_style(precision, sig_figs, messaging_service)
        })

class data_request(dict):
    def __init__(self,
                 variable = "",
                 state_indices = None,
                 param_indices = None,
                 time_or_freq_indices = None,
                 aggregation = None):

        super().__init__({'variable': variable,
                        'state_indices': state_indices,
                        'param_indices': param_indices,
                        'time_or_freq_indices': time_or_freq_indices,
                        'aggregation': aggregation})

class generic_plot_style(object):
    """Template parent class with common functionality and template methods to
    overwrite when subclassing. In it's raw form it mostly returns junk, but will
    get some data from the model."""

    axis_labels = {'x': "",
                'y': "",
                'z': ""}

    def __init__(self, precision, sig_figs, messaging_service=None):
        self.precision = precision
        self.display_sig_figs = sig_figs
        self.data_pending = False
        self.style = 'surface'

        if messaging_service:
            self.register_messaging_service(messaging_service)
        else:
            self.messenger = None

    def register_messaging_service(self, messaging_service):
        """Connect a message passing service to communicate between widgets.
        Subscribe all setter functions that don't generate their own data or get
        it from the user interface.

        Args:
            messaging_service(class): A class with publish, subscribe, unsubscribe methods"""

        self.messenger = messaging_service
        self.messenger.subscribe("requested_data", self.receive_data_from_model)


    def trim_xy(self, xdata, ydata, plot_state):

        [xmin, xmax] = plot_state['slices']['x']
        [ymin, ymax] = plot_state['slices']['y']

        xdata = xdata[(xdata >= xmin) & (xdata <= xmax)]
        ydata = ydata[(ydata >= ymin) & (ydata <= ymax)]

        return xdata, ydata

    def param_lists_to_tuples(self, p1, p2):
        return [(p1val, p2val) for p1val in p1 for p2val in p2]

    def map_tuple_to_index(self, param_tuple, tuple_map):
        rounded_tuple = (self.precision(round_sf(param_tuple[0], self.display_sig_figs)),
                        self.precision(round_sf(param_tuple[1], self.display_sig_figs)))
        return tuple_map.get(rounded_tuple)

    def map_tuple_list_to_indices(self, param_tuples, tuple_map):
        return [self.map_tuple_to_index(param_tuple, tuple_map) for param_tuple in param_tuples]

    def get_param_indices_from_vectors(self, p1, p2, tuple_map):
        tuples = self.param_lists_to_tuples(p1, p2)
        indices = self.map_tuple_list_to_indices(tuples, tuple_map)
        return indices

    def get_x_bounds(self, plot_state, external_variables):
        xdata = self.get_xdata(plot_state, external_variables)
        return [np.amin(xdata), np.amax(xdata)]

    def get_y_bounds(self, plot_state, external_variables):
        ydata = self.get_ydata(plot_state, external_variables)
        return [np.amin(ydata), np.amax(ydata)]

    def update_axis_labels(self, plot_state):
        if plot_state['variables']['x'] == 'Parameter 1':
            self.axis_labels['x'] = plot_state['param_labels'][0]
        else:
            self.axis_labels['x'] = plot_state['param_labels'][1]
        self.axis_labels['y'] = 'Frequency (hertz-like)'
        self.axis_labels['z'] = plot_state['variables']['z']

    def generate_data_request(self, plot_state, external_variables, average=True):
        """PLACEHOLDER - OVERLOAD WITH THE STYLE'S LOGIC"""


        request = data_request(variable = 'Amplitude',
                               state_indices = 0,
                               param_indices = 0,
                               time_or_freq_indices =None,
                               aggregation = None)

        return request

    def generate_animation_requests(self, plot_state,
                                    external_variables):
        """PLACEHOLDER - OVERLOAD WITH THE STYLE'S LOGIC"""
        frame_requests = []
        for p in [0,1]:
            request = self.generate_data_request(plot_state, external_variables)
            frame_requests = frame_requests.append(request)

        return frame_requests

    def request_from_model(self, request):
        self.messenger.publish("model_request", request)

    def receive_data_from_model(self, data):
        self.requested_data = data

    def get_xy_grid(self, plot_state, external_variables):
        return -1, -1

    def get_param_indices(self, plot_state, external_variables):
        return -1

    def get_z(self, plot_state, external_variables):
        request = self.generate_data_request(plot_state, external_variables)
        self.request_from_model(request)
        return self.requested_data

    def get_animate_zlist(self, plot_state, external_variables):
        zlist = []
        values = self.get_animation_values(plot_state, external_variables)
        for value in values:
            plot_state['fixed_value'] = value
            plot_state['param_indices'] = self.get_param_indices(plot_state, external_variables)
            Z = self.get_z(plot_state, external_variables)
            zlist.append(Z)
        return zlist

    def get_plot_data(self, plot_state, external_variables, animate=False):

        X, Y = self.get_ordinate_data(plot_state, external_variables)
        plot_state['param_indices'] = self.get_param_indices(plot_state, external_variables)
        if animate:
            Z = self.get_animate_zlist(plot_state, external_variables)
        else:
            Z = self.get_z( plot_state, external_variables)
        return X, Y, Z

    def request_update_plot(self, X, Y, Z, plot_state, animate=False):

        plot_data = {'data': {'x': X,
                              'y': Y,
                              'z': Z},
                     'scales': plot_state['scales'],
                     'axis_labels': self.axis_labels,
                     'animate': animate,
                     'style': self.style}

        self.messenger.publish("update_plot", plot_data)

    def plot(self, plot_state, external_variables, animate=False):
        self.update_axis_labels(plot_state, external_variables)
        X, Y, Z = self.get_plot_data(plot_state, external_variables, animate)
        self.request_update_plot(X, Y, Z, plot_state, animate)





class grid3d_style(generic_plot_style):

    def __init__(self, *args):
        super().__init__(*args)
        self.style = 'surface'

    def update_axis_labels(self, plot_state, external_variables):
        self.axis_labels['x'] = external_variables['param_labels'][0]
        self.axis_labels['y'] = external_variables['param_labels'][1]
        self.axis_labels['z'] = plot_state['variables']['z']



    def generate_data_request(self, plot_state, external_variables, average=True):
        """Generate a request to send to the model to get grid3d data at a fixed
        frequency"""
        f = external_variables['frequency_bins']
        variable = plot_state['variables']['z']
        state = [plot_state['state_to_plot']]
        param_indices = plot_state['param_indices']
        fixed_value = plot_state['fixed_value']
        aggregation = None

        if variable == 'Amplitude':
            if average == True:
                aggregation = 'RMS'
                time_or_freq_indices = None
            else:
                time_or_freq_indices = [fixed_value]
        else:
            time_or_freq_indices = [np.argmin(np.abs(fixed_value - f))]

        request = data_request(variable = variable,
                               state_indices = state,
                               param_indices = param_indices,
                               time_or_freq_indices = time_or_freq_indices,
                               aggregation = aggregation)

        return request

    def get_animation_values(self, plot_state, external_variables):
        """Build a list of data requests, one per frame of the requested animation"""
        f = external_variables['frequency_bins']
        t = external_variables['t']
        variable = plot_state['variables']['z']

        if variable == 'Amplitude':
            iteration_variable = t
        else:
            iteration_variable = f

        return iteration_variable


    def get_ordinate_data(self, plot_state, external_variables):
        #Consider generalising - get ordinate data- group such that this can just return x, and the request: z.
        xdata = external_variables['param1_values']
        ydata = external_variables['param2_values']
        xdata, ydata = self.trim_xy(xdata, ydata, plot_state)

        X, Y = np.meshgrid(xdata, ydata)

        return X, Y

    def get_param_indices(self, plot_state, external_variables):
        p1_full = external_variables['param1_values']
        p2_full = external_variables['param2_values']
        X, Y = np.meshgrid(p1_full, p2_full)
        tuple_map = external_variables['param_index_map']

        param_indices_flat = [self.map_tuple_to_index((x, y), tuple_map) for x, y in zip(X.flatten(), Y.flatten())]
        param_indices = np.array(param_indices_flat).reshape(X.shape)
        return param_indices



class time3d_style(generic_plot_style):

    def __init__(self, *args):
        super().__init__(*args)
        self.style = 'surface'

    def update_axis_labels(self, plot_state, external_variables):
        if plot_state['variables']['x'] == 'Parameter 1':
            self.axis_labels['x'] = external_variables['param_labels'][0]
        else:
            self.axis_labels['x'] = external_variables['param_labels'][1]
        self.axis_labels['y'] = 'Time (seconds-like)'
        self.axis_labels['z'] = plot_state['variables']['z']

    def generate_data_request(self, plot_state, external_variables, average=True):
        """Generate a request to send to the model to get grid3d data at a fixed
        frequency"""
        [ymin, ymax] = plot_state['slices']['y']
        t = external_variables['t']
        variable = plot_state['variables']['z']
        state = [plot_state['state_to_plot']]
        param_indices = plot_state['param_indices']
        aggregation = None

        time_or_freq_indices = np.argwhere((t >= ymin) & (t <= ymax))

        request = data_request(variable = variable,
                               state_indices = state,
                               param_indices = param_indices,
                               time_or_freq_indices = time_or_freq_indices,
                               aggregation = aggregation)

        return request

    def get_animation_values(self, plot_state,
                                    external_variables):
        """Get vector of "fixed values" for animation """

        xvar = plot_state['variables']['x']
        p1_values = external_variables['param1_values']
        p2_values = external_variables['param2_values']
        if xvar == 'Parameter 1':
            iterated_variable = p2_values
        else:
            iterated_variable = p1_values
        return iterated_variable

    def get_ordinate_data(self, plot_state, external_variables):
        #Consider generalising - get ordinate data- group such that this can just return x, and the request: z.
        [xmin, xmax] = plot_state['slices']['x']
        xvar = plot_state['variables']['x']
        p1_full = external_variables['param1_values']
        p2_full = external_variables['param2_values']

        if xvar == 'Parameter 1':
            xdata = p1_full[(p1_full >= xmin) & (p1_full <= xmax)]
        else:
            xdata = p2_full[(p2_full >= xmin) & (p2_full <= xmax)]

        ydata = external_variables['t']
        xdata, ydata = self.trim_xy(xdata, ydata, plot_state)
        X, Y = np.meshgrid(xdata, ydata)

        return X, Y

    def get_param_indices(self, plot_state, external_variables):
        [xmin, xmax] = plot_state['slices']['x']
        xvar = plot_state['variables']['x']
        p_fixed = plot_state['fixed_value']
        p1_full = external_variables['param1_values']
        p2_full = external_variables['param2_values']
        tuple_map = external_variables['param_index_map']

        if xvar == 'Parameter 1':
            p1 = p1_full[(p1_full >= xmin) & (p1_full <= xmax)]
            p2 = [p_fixed]
        else:
            p1 = [p_fixed]
            p2 = p2_full[(p2_full >= xmin) & (p2_full <= xmax)]

        param_indices = self.get_param_indices_from_vectors(p1, p2, tuple_map)

        return param_indices

class spec3d_style(generic_plot_style):

    def __init__(self, *args):
        super().__init__(*args)
        self.style = 'surface'

    def update_axis_labels(self, plot_state, external_variables):
        if plot_state['variables']['x'] == 'Parameter 1':
            self.axis_labels['x'] = external_variables['param_labels'][0]
        else:
            self.axis_labels['x'] = external_variables['param_labels'][1]
        self.axis_labels['y'] = 'Frequency (hertz-like)'
        self.axis_labels['z'] = plot_state['variables']['z']

    def generate_data_request(self, plot_state, external_variables, average=True):
        [ymin, ymax] = plot_state['slices']['y']
        f = external_variables['frequency_bins']
        variable = plot_state['variables']['z']
        state = [plot_state['state_to_plot']]
        param_indices = plot_state['param_indices']
        aggregation = None

        time_or_freq_indices = np.argwhere((f >= ymin) & (f <= ymax))

        request = data_request(variable = variable,
                               state_indices = state,
                               param_indices = param_indices,
                               time_or_freq_indices = time_or_freq_indices,
                               aggregation = aggregation)

        return request

    def get_animation_values(self, plot_state,
                                    external_variables):
        """Get vector of "fixed values" for animation """

        xvar = plot_state['variables']['x']
        p1_values = external_variables['param1_values']
        p2_values = external_variables['param2_values']
        if xvar == 'Parameter 1':
            iterated_variable = p2_values
        else:
            iterated_variable = p1_values
        return iterated_variable

    def get_ordinate_data(self, plot_state, external_variables):
        #Consider generalising - get ordinate data- group such that this can just return x, and the request: z.
        [xmin, xmax] = plot_state['slices']['x']
        xvar = plot_state['variables']['x']
        p1_full = external_variables['param1_values']
        p2_full = external_variables['param2_values']

        if xvar == 'Parameter 1':
            xdata = p1_full[(p1_full >= xmin) & (p1_full <= xmax)]
        else:
            xdata = p2_full[(p2_full >= xmin) & (p2_full <= xmax)]

        ydata = external_variables['frequency_bins']
        xdata, ydata = self.trim_xy(xdata, ydata, plot_state)
        X, Y = np.meshgrid(xdata, ydata)

        return X, Y

    def get_param_indices(self, plot_state, external_variables):
        [xmin, xmax] = plot_state['slices']['x']
        xvar = plot_state['variables']['x']
        p_fixed = plot_state['fixed_value']
        p1_full = external_variables['param1_values']
        p2_full = external_variables['param2_values']
        tuple_map = external_variables['param_index_map']

        if xvar == 'Parameter 1':
            p1 = p1_full[(p1_full >= xmin) & (p1_full <= xmax)]
            p2 = [p_fixed]
        else:
            p1 = [p_fixed]
            p2 = p2_full[(p2_full >= xmin) & (p2_full <= xmax)]

        param_indices = self.get_param_indices_from_vectors(p1, p2, tuple_map)

        return param_indices

class singlestate_style(generic_plot_style):
    """Four plots in a trenchcoat, so this one is big and ugly. Break into four ,
    as these are only lumped into one for display/legacy reasons."""


    def __init__(self, *args):
        super().__init__(*args)
        self.style = '2d line'


    def update_axis_labels(self, plot_state, state_labels = None):
        plot = plot_state['singleplot_style']
        xstate = plot_state['variables']['x']
        ystate = plot_state['variables']['y']
        zstate = plot_state['variables']['z']

        if plot == 'Spectrogram':
            self.axis_labels['x'] = 'Frequency (hertz-like)'
            self.axis_labels['y'] = 'Amplitude (V^2/root(Hz)'
        elif plot == '2D Phase Diagram':
            self.axis_labels['x'] = state_labels[xstate]
            self.axis_labels['y'] = state_labels[ystate]
        elif plot == '3D Phase Diagram':
            self.axis_labels['x'] = state_labels[xstate]
            self.axis_labels['y'] = state_labels[ystate]
            self.axis_labels['z'] = state_labels[zstate]
        elif plot == 'Time-domain':
            self.axis_labels['x'] = 'Time (seconds-like)'
            self.axis_labels['y'] = state_labels[ystate]



    # def get_parameter_tuples(self, plot_state, external_variables):
    #     p1_single = self.plot_state['single_param1_selection']
    #     p2_single = self.plot_state['single_param2_selection']
    #     p1 = [p1_single]
    #     p2 = [p2_single]
    #     return self.param_lists_to_tuples(p1, p2)

    def generate_data_request(self, plot_state, external_variables, average=True):
        """Generate a request to send to the model to get grid3d data at a fixed
        frequency"""

        plot = plot_state['singleplot_style']

        xstate = plot_state['variables']['x']
        ystate = plot_state['variables']['y']
        zstate = plot_state['variables']['z']


        if plot == 'Spectrogram':
            raise NotImplementedError
        elif plot == '2D Phase Diagram':
            states = [xstate, ystate]
        elif plot == '3D Phase Diagram':
            states = [xstate, ystate, zstate]
        elif plot == 'Time-domain':
            states = [ystate]

        variable = 'Amplitude'
        param_indices = plot_state['param_indices']
        aggregation = None
        time_or_freq_indices = None

        request = data_request(variable = variable,
                               state_indices = states,
                               param_indices = param_indices,
                               time_or_freq_indices = time_or_freq_indices,
                               aggregation = aggregation)

        return request

    def generate_animation_requests(self, plot_state,
                                    external_variables,
                                    param_to_animate=1):
        """Build a list of data requests, one per frame of the requested animation"""
        p1_values = external_variables['param1_values']
        p2_values = external_variables['param2_values']
        frame_requests = []
        tuple_map = external_variables['param_index_map']
        if param_to_animate == 2:
            for p2 in p2_values:
                plot_state['single_param2_selection'] = p2
                param_tuples = self.get_parameter_values(plot_state, external_variables)
                plot_state['param_indices'] = [self.map_tuple_to_index(param_tuples)]
                request = self.generate_data_request(plot_state, external_variables)
                frame_requests = frame_requests.append(request)
        else:
            for p1 in p1_values:
                plot_state['single_param1_selection'] = p1
                param_tuple = self.get_parameter_values(plot_state, external_variables)
                plot_state['param_indices'] = self.map_tuple_to_index(param_tuple, tuple_map)
                request = self.generate_data_request(plot_state, external_variables)
                frame_requests = frame_requests.append(request)

        return frame_requests

    def get_xdata(self, plot_state, external_variables):
        return external_variables['param1_values']

    def get_ydata(self, plot_state, external_variables):
        return external_variables['param2_values']


    def get_plot_data(self, xdata, ydata, requested_data):
        X, Y = np.meshgrid(xdata, ydata)

        #Get indices of all parameter tuples one unique index per cell in the 3d grid
        indices = np.array([self.get_param_index((x, y)) for x, y in zip(X.ravel(), Y.ravel())])
        indices = indices.reshape(X.shape)  # Reshape to match meshgrid

        #requested data should have size one on the "state" and time/freq dimensions,
        # So if we index with a 2D array of indices, we shoudl get a 2D scalar array back.
        Z = requested_data[0,indices,0]

        return X, Y, Z