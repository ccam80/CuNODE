# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:02:51 2024

@author: cca79
"""

from qtpy.QtWidgets import QFrame, QGridLayout, QLabel, QLineEdit, QGroupBox, QWidget, QSizePolicy, QSpacerItem
from qtpy.QtCore import Slot, Signal
from _utils import get_readonly_view
from pubsub import PubSub
from numpy import zeros
from gui.resources.widgets.qtdesigner.sim_controller import Ui_simController
import logging
import numpy as np

class sim_controller_widget(QFrame, Ui_simController):

    solve_request = Signal()

    def __init__(self,
                 parent=None,
                 messaging_service=None,
                 precision=np.float64):

        super(sim_controller_widget, self).__init__(parent)
        self.setupUi(self)

        self.precision = precision
        if messaging_service:
            self.register_messaging_service(messaging_service)
        else:
            self.messenger = None

        self.sim_state = {
            'dt': 0.0,
            'duration': 0.0,
            'fs': 0.0,
            'param1_sweep_bounds': [0.0,1.00],
            'param1_sweep_scale': 'Linear',
            'param1_num_values': 100,
            'param1_var': '',
            'param1_values': zeros(1),
            'param2_sweep_bounds': [0.0,1.0],
            'param2_sweep_scale': 'Linear',
            'param2_num_values': 100,
            'param2_var': '',
            'param2_values': zeros(1),
            'warmup': 0.0,
            'y0': zeros(1),
            'noise_sigmas': zeros(1)
        }

        self.fft_params = {'psd_nperseg': 256,
                           'psd_segments': 5,
                           'psd_window': 'hann',
                           'fft_window': 'hann'}
        self.system_params_local = {}
        self.get_tab_values(self.SimulationSettings_tab)


    def register_messaging_service(self, messaging_service):
        """Connect a message passing service to communicate between widgets

        Args:
            messaging_service(class): A class with publish, subscribe, unsubscribe methods"""
        messaging_service.subscribe("precision", self.update_precision)

        self.messenger = messaging_service

    def update_precision(self, precision):
        self.precision = precision

    def publish(self, topic, label):
        self.messenger.publish(topic, label)

    @Slot(str)
    def update_p1_from(self, _from):
        self.sim_state['param1_sweep_bounds'][0] = float(_from)
        logging.debug(f"P1 from updated: {_from}")

    @Slot(str)
    def update_p1_to(self, to):
        self.sim_state['param1_sweep_bounds'][1] = float(to)
        logging.debug(f"P1 to updated: {to}")

    @Slot(str)
    def update_p1_n(self, n):
        self.sim_state['param1_num_values'] = int(n)
        logging.debug(f"P1 n updated: {n}")

    @Slot(str)
    def update_p1_scale(self, scale):
        self.sim_state["param1_sweep_scale"] = scale
        logging.debug(f"P1 scale updated: {scale}")

    @Slot(str)
    def update_p1_var(self, var):
        self.sim_state["param1_var"] = var
        logging.debug(f"P1 var updated: {var}")

    @Slot(str)
    def update_p2_from(self, _from):
        self.sim_state['param2_sweep_bounds'][0] = float(_from)
        logging.debug(f"P2 from updated: {_from}")

    @Slot(str)
    def update_p2_to(self, to):
        self.sim_state['param2_sweep_bounds'][1] = float(to)
        logging.debug(f"P2 to updated: {to}")

    @Slot(str)
    def update_p2_n(self, n):
        self.sim_state['param2_num_values'] = int(n)
        logging.debug(f"P2 n updated: {n}")

    @Slot(str)
    def update_p2_scale(self, scale):
        self.sim_state["param2_sweep_scale"] = scale
        logging.debug(f"P2 scale updated: {scale}")

    @Slot(str)
    def update_p2_var(self, var):
        self.sim_state["param2_var"] = var
        logging.debug(f"P2 var updated: {var}")

    @Slot(str)
    def update_duration(self, duration):
        self.sim_state['duration'] = float(duration)
        logging.debug("Duration updated: {duration}")

    @Slot(str)
    def update_warmup(self, warmup):
        self.sim_state['warmup'] = float(warmup)
        logging.debug("Warmup updated: {warmup}")

    @Slot(str)
    def update_fs(self, fs):
        self.sim_state['fs'] = float(fs)
        logging.debug("Fs updated: {fs}")

    @Slot(str)
    def update_dt(self, dt):
        self.sim_state['dt'] = float(dt)
        logging.debug("Dt updated: {dt}")

    @Slot(str, str)
    def update_param(self, param, value):
        self.system_params_local[param] = float(value)
        logging.debug(f"Param updated: {param}, {value}")

    @Slot(str, str)
    def update_init(self, index, value):
        self.sim_state['y0'][index] =  float(value)
        logging.debug(f"Init updated: {index}, {value}")

    @Slot(str, str)
    def update_noise(self, index, value):
        self.noise[index] =  float(value)

        logging.debug(f"Noise updated: {index}, {value}")

    def load_system(self, sysparams, noise_sigmas):
        self.local_sysparams_dict = sysparams
        self.sim_state['noise_sigmas'] = noise_sigmas
        self.populate_sysParams_tab(sysparams, noise_sigmas)
        self.sim_state['y0'] = zeros(len(noise_sigmas))
        self.param1Sweep_options.varDdItems = sysparams.keys()
        self.param2Sweep_options.varDdItems = sysparams.keys()
        self.populate_groupbox_with_array(self.inits_box, self.sim_state['y0'])

    def populate_sysParams_tab(self, sysparams_dict, noise_array):
        if self.SystemParameters_tab.layout():
            for i in reversed(range(self.SystemParameters_tab.layout().count())):
                widget = self.SystemParameters_tab.layout().itemAt(i).widget()
                if widget:
                    widget.deleteLater()

        layout = QGridLayout()
        keys = list(sysparams_dict.keys())

        numcols = 2
        for row in range((len(keys) + 1) // numcols):
            col = 0
            for i in range(numcols):
                idx = row * numcols + i
                if idx >= len(keys):
                    break
                key = keys[idx]
                value = sysparams_dict[key]

                label = QLabel(key)
                entry = QLineEdit()
                entry.setText(f"{value:.2f}")
                entry.textChanged.connect(lambda text, k=key: self.update_sysparam(k, text))
                label.setStyleSheet("font-size: 12pt;")
                entry.setStyleSheet("font-size: 12pt;")

                layout.addWidget(label, row, col)
                layout.addWidget(entry, row, col + 1)
                col += numcols

        noise_groupbox = QGroupBox("Gaussian noise (std dev)")
        self.populate_groupbox_with_array(noise_groupbox, noise_array)
        layout.addWidget(noise_groupbox, (len(keys) + 1) // numcols, 0, 1, numcols * 2)

        # Add a vertical spacer to the bottom of a QVBoxLayout called "layout"
        spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacerItem)

        container = QWidget()
        container.setLayout(layout)
        self.SystemParameters_tab.layout().addWidget(container)

    def populate_groupbox_with_array(self, groupbox, array):
        layout = groupbox.layout()
        if layout is not None:
            for i in reversed(range(layout.count())):
                widget = layout.itemAt(i).widget()
                if widget:
                    widget.deleteLater()
        else:
            layout = QGridLayout()
            groupbox.setLayout(layout)

        for i, value in enumerate(array):
            label = QLabel(str(i))
            entry = QLineEdit()
            entry.setText(f"{value:.2f}")
            entry.textChanged.connect(lambda text, idx=i: self.update_array(idx, text, array))
            label.setStyleSheet("font-size: 12pt;")
            entry.setStyleSheet("font-size: 12pt;")

            layout.addWidget(label, i, 0)
            layout.addWidget(entry, i, 1)

    def update_sysparam(self, key, text):
        try:
            self.local_sysparams_dict[key] = float(text)
        except ValueError:
            pass

    def update_array(self, index, text, array):
        try:
            array[index] = float(text)
        except ValueError:
            pass

    def get_tab_values(self, tab):
        """
        Triggers the textChanged signal for all entry boxes on
        on the specified tab.

        Args:
            tab (QWidget): The tab on which to trigger signals.
        """
        # Trigger textChanged for all QLineEdit and its subclasses
        line_edits = tab.findChildren(QLineEdit)
        for line_edit in line_edits:
            line_edit.textChanged.emit(line_edit.text())

    def get_sysparams(self):
        return self.system_params_local.copy()

    def get_sim_state(self):
        return self.sim_state.copy()

    def update_independent_variables(self):
        self.update_parameter_sweeps()
        self.update_sweep_labels()
        self.update_time_vector()
        self.update_freq_vectors()


    def solve(self):
        #TODO: protect against pushing the solve button without a system loaded.
        #Do this by deactivating buttons, but also add a pubsubbed error that the
        #top GUI makes an errorbox for.

        self.update_independent_variables()

        self.solve_request.emit()

    def get_swept_parameters(self, param):
        bounds = self.sim_state[param + "_sweep_bounds"]
        n = self.sim_state[param + "_num_values"]
        numpyspace_args = (bounds[0], bounds[1], n)
        scale = self.sim_state[param + "_sweep_scale"]

        return scale, numpyspace_args

    def generate_sweep(self, param):
        scale, space_args = self.get_swept_parameters(param)

        if scale == 'Linear':
            sweep = np.linspace(*space_args, dtype=self.precision)
        elif scale == 'Logarithmic':
            try:
                sweep = np.logspace(np.log10(*space_args, dtype=self.precision))
            except:
                logging.warning(f"Bounds: {space_args} unable to create a log space")
                self.displayerror("The requested logarithmic sweep contains un-loggable values, change bounds or try a linear sweep.")
        return sweep

    def update_time_vector(self):
        duration = self.sim_state['duration']
        fs = self.sim_state['fs']
        warmup = self.sim_state['warmup']

        self.t = np.linspace(warmup, warmup + duration, int(fs*duration))
        self.publish("t",get_readonly_view(self.t))

    # def get_time_vector(self):
    #     return get_readonly_view(self.t)

    def update_freq_vectors(self):
        spectr_fs = self.sim_state['fs'] * 2 * np.pi # this is a hangove of the nondimensionalising of Seigan's model, and should be removed once we get to a working version

        t_axis_length = len(self.t)
        nperseg = int(t_axis_length / 4) # These are a copy from generic solver, incorporate them into a settings dict modifiable through the gui
        noverlap = int(nperseg/2)
        nfft = nperseg*2
        max_f = spectr_fs / 2


        self.psd_freq = np.linspace(0, max_f, nperseg)
        self.fft_freq = np.linspace(0, max_f, int(t_axis_length/2))

        self.publish("fft_freq", get_readonly_view(self.fft_freq))
        self.publish("psd_freq", get_readonly_view(self.psd_freq))

    # def get_psd_freq_vector(self):
    #     return get_readonly_view(self.psd_freq)

    # def get_fft_freq_vector(self):
    #     return get_readonly_view(self.fft_freq)


    def update_parameter_sweeps(self):
        self.param1_values = self.generate_sweep('param1')
        self.param2_values = self.generate_sweep('param2')
        self.publish("param1_values", get_readonly_view(self.param1_values))
        self.publish("param2_values", get_readonly_view(self.param2_values))



    # def get_parameter_sweeps(self):
    #     p1view = get_readonly_view(self.param1_values)
    #     p2view = get_readonly_view(self.param2_values)
    #     return p1view, p2view

    def update_sweep_labels(self):
        p1_var = self.sim_state['param1_var']
        p2_var = self.sim_state['param2_var']
        self.sweep_labels = [p1_var, p2_var]
        self.publish('param_labels', self.sweep_labels)

    # def get_sweep_labels(self):
    #     return self.sweep_labels.copy()