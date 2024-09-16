# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 21:00:07 2024

@author: cca79
"""
from qtpy.QtWidgets import QFrame, QAbstractButton, QMenu, QLineEdit, QComboBox
from qtpy.QtCore import Slot, Signal
from qtpy.QtGui import QAction
from gui.resources.widgets.qtdesigner.plot_controller import Ui_plotController
from pubsub import PubSub
from gui.resources.plotstyle_data_interactions import plotstyle_interfaces
import logging
import numpy as np
from time import sleep
from _utils import round_sf, round_list_sf
class plot_controller_widget(QFrame, Ui_plotController):

    updatePlot = Signal()
    plotstyles = {0: 'grid3d',
                  1: 'time3d',
                  2: 'spec3d',
                  3: 'singlestate'}

    variable_labels = {'Parameter 1': 'param1_values',
                       'Parameter 2': 'param2_values',
                       'Time': 't',
                       'Frequency': 'frequency_bins'}
    def __init__(self,
                 parent=None,
                 precision=np.float64,
                 messaging_service=None,
                 display_sig_figs = 4):
        self.precision = precision


        self.plot_state = {
            'state_to_plot': 0,
            'plot_type': 'grid3d',
            'single_param1_selection': 1.0,
            'single_param2_selection': 1.0,
            'fixed_value': 1.5,
            'single_plot_style': '2D Phase Plot',
            'scales': {'x': 'Linear',
                       'y': 'Linear',
                       'z': 'Linear'},
            'slices': {'x': [0, 0],
                       'y': [0, 0],
                       'z': [0, 0]},
            'variables': {'x': 'Parameter 1',
                          'y': 'Parameter 2',
                          'z': 'PSD'},
            'param_indices': [],
        }

        self.external_variables = {"t": np.zeros(1, dtype=self.precision),
                                   "psd_freq": np.zeros(1, dtype=self.precision),
                                   "fft_freq": np.zeros(1, dtype=self.precision),
                                   'frequency_bins': np.zeros(1, dtype=self.precision),
                                   "param1_values": np.zeros(1, dtype=self.precision),
                                   "param2_values": np.zeros(1, dtype=self.precision),
                                   "param_labels": ["",""],
                                   "param_index_map": {},
                                   "display_sig_figs": 4}

        self.plotstyle_interfaces = plotstyle_interfaces(precision = self.precision,
                                                         sig_figs=self.external_variables['display_sig_figs'])
        self.state_labels = {'': 0}

        self.yboxes = []
        self.xboxes = []
        self.zboxes = []

        super().__init__(parent)
        self.setupUi(self)
        self.populate_toolbutton_menus()

        self.xboxes = [self.paramAxisTime3D_options,
                       self.paramAxisSpec3d_options,
                       self.xAxisGridPlotOptions,
                       self.xAxisSinglePlotOptions]

        self.yboxes = [self.timeAxis3d_options,
                       self.freqAxisSpec3d_options,
                       self.yAxisGridPlotOptions,
                       self.yAxisSinglePlotOptions]

        self.zboxes = [self.amplAxisTime3d_options,
                       self.amplAxisSpec3d_options,
                       self.zAxisGridPlotOptions,
                       self.zAxisSinglePlotOptions]

        if messaging_service:
            self.register_messaging_service(messaging_service)
        else:
            self.messenger = None

        self.requested_data = np.zeros(1, dtype=self.precision)
        self.plotMode_select(0)





    def register_messaging_service(self, messaging_service):
        """Connect a message passing service to communicate between widgets.
        Subscribe all setter functions that don't generate their own data or get
        it from the user interface.

        Args:
            messaging_service(class): A class with publish, subscribe, unsubscribe methods"""

        messaging_service.subscribe("t", self.update_t)
        messaging_service.subscribe("psd_freq", self.update_psd_freq)
        messaging_service.subscribe("fft_freq", self.update_fft_freq)
        messaging_service.subscribe("param1_values", self.update_param1_values)
        messaging_service.subscribe("param2_values", self.update_param2_values)
        messaging_service.subscribe("param_labels", self.update_param_labels)
        messaging_service.subscribe("precision", self.update_precision)
        messaging_service.subscribe("param_index_map", self.update_param_index_map)
        messaging_service.subscribe("animation_step", self.update_animation_step)

        self.messenger = messaging_service
        for key, plotstyle in self.plotstyle_interfaces.items():
            plotstyle.register_messaging_service(messaging_service)


    #************************ PubSub sibscriber callbacks ********************#
    def update_param_index_map(self, param_index_map):
        self.external_variables['param_index_map'] = param_index_map

    def update_t(self, t):
        self.external_variables["t"] = t

    def update_psd_freq(self, f):
        self.external_variables["psd_freq"] = f

    def update_fft_freq(self, f):
        self.external_variables["fft_freq"] = f

    def update_param1_values(self, p1):
        self.external_variables["param1_values"] = p1

    def update_param2_values(self, p2):
        self.external_variables["param2_values"] = p2

    def update_param_labels(self, labels):
        self.external_variables["param_labels"] = labels

    def update_precision(self, precision):
        self.precision = precision

    def update_animation_step(self, step):
        self.fixedParamTime_slider.setSliderPosition(step)
        self.fixedParamSpec_slider.setSliderPosition(step)
        self.frequency_slider.setSliderPosition(step)


    @Slot(int)
    def plotMode_select(self, index):
        """Change plot style and "kick" all widgets on that tab into updating
        the plot_state dict with their current values"""

        current_tab = self.plotSettingsTabs.widget(index)
        self.get_tab_values(current_tab)
        logging.debug(f"Plot mode selected: {index}")
        plot_state = self.plotstyles[index]
        self.plot_state['plot_type'] = plot_state
        self.data_interface = self.plotstyle_interfaces[plot_state]

    @Slot(str, int)
    def update_xSlice(self, value, index):
        self.plot_state['slices']['x'][index] = float(value)
        logging.debug("X Slice updated")

    @Slot(str, int)
    def update_ySlice(self, value, index):
        self.plot_state['slices']['y'][index] = float(value)
        logging.debug("Y Slice updated")

    @Slot(str, int)
    def update_zSlice(self, value, index):
        self.plot_state['slices']['z'][index] = float(value)
        logging.debug("Z Slice updated")

    @Slot(str)
    def update_xScale(self, scale):
        self.plot_state['scales']['x'] = scale
        logging.debug("X Scale updated")

    @Slot(str)
    def update_yScale(self, scale):
        logging.debug("Y Scale updated")
        self.plot_state['scales']['y'] = scale

    @Slot(str)
    def update_zScale(self, scale):
        logging.debug("Z Scale updated")
        self.plot_state['scales']['z'] = scale

    @Slot(str)
    def update_xVar(self, variable):
        self.plot_state['variables']['x'] = variable
        self.update_variable_range(self.xboxes, variable)
        self.update_fixed_slider()

    @Slot(str)
    def update_yVar(self, variable):
        self.plot_state['variables']['y'] = variable
        self.update_variable_range(self.yboxes, variable)

    @Slot(str)
    def update_zVar(self, variable):
        logging.debug("Z Variable updated")
        self.plot_state['variables']['z'] = variable
        if variable == 'PSD':
            freq = self.external_variables["psd_freq"]
            self.external_variables['frequency_bins'] = freq
            self.update_frequency_slider(freq)

        elif variable == "FFT Phase":
            freq = self.external_variables["fft_freq"]
            self.external_variables['frequency_bins'] = freq
            self.update_frequency_slider(freq)


    def update_display_sigfigs(self, sf):
        self.external_variables['display_sig_figs'] = sf

    @Slot()
    def on_updatePlot(self):
        logging.debug("Update plot")
        self.data_interface.plot(self.plot_state, self.external_variables)
        # self.updatePlot.emit()

    @Slot(int, int)
    def set_current_params_from_plotter(self, param1, param2):
        param1index = np.argmin(np.abs(self.external_variables["param1_values"] - param1))
        param2index = np.argmin(np.abs(self.external_variables["param2_values"] - param2))
        self.param1ValSelect_dd.setCurrentIndex(param1index)
        self.param2ValSelect_dd.setCurrentIndex(param2index)

    @Slot(int)
    def set_slice_frequency(self, index):
        freq = self.external_variables['frequency_bins'][index]
        self.plot_state['fixed_value'] = freq
        self.currentFrequency_l.display(freq)
        logging.debug(f"Slice frequency set to: {freq}")

    @Slot(int)
    def set_fixedParam(self, index):
        if self.plot_state['variables']['x'] == 'Parameter 1':
            param_val = self.external_variables["param2_values"][index]
        else:
            param_val = self.external_variables["param1_values"][index]

        self.plot_state['fixed_value'] = param_val
        self.fixedParamTime_display.display(param_val)
        self.fixedParamSpec_entry.setText(str(param_val))

        logging.debug(f"Fixed parameter set to: {param_val}")

    # @Slot(str)
    # def update_slider_from_entrybox(self, value):

    @Slot()
    def animate_fixedparam(self):

        self.data_interface.plot(self.plot_state, self.external_variables, animate=True)

        logging.debug("Fixed parameter animation started")


    @Slot(QAbstractButton)
    def update_singlePlotStyle(self, button):
        logging.debug(f"Single plot style updated: {button}")
        self.plot_state['single_plot_style'] = button.text()

    @Slot(int)
    def select_single_param1(self, index):
        param = self.external_variables["param1_values"][index]
        self.plot_state['single_param1_selection'] = self.external_variables["param1_values"][index]
        logging.debug(f"Parameter 1 selected: {param}")


    @Slot(int)
    def select_single_param2(self, index):
        param = self.external_variables["param2_values"][index]
        self.plot_state['single_param2_selection'] = self.external_variables["param2_values"][index]
        logging.debug(f"Parameter 2 selected: {param}")

    @Slot()
    def animate_param1_singleSet(self):
        logging.debug("Parameter 1 single set animation started")
        #TODO: figurre out animation logic

    @Slot()
    def animate_param2_singleSet(self):
        logging.debug("Parameter 2 single set animation started")
        #TODO: figurre out animation logic

    @Slot(str)
    def update_state_to_plot(self, state):
        if state in self.state_labels.keys():
            state_index = self.state_labels[state]
            self.plot_state['state_to_plot'] = state_index
            state_to_plot_comboboxes = [self.plot_state_time3D,
                                        self.plot_state_spec3D,
                                        self.plot_state_grid3D]

            for box in state_to_plot_comboboxes:
                box.setCurrentIndex(state_index)
            logging.debug("Plot state set to {state_index}")

    # def populate_swept_parameter_values(self):
    #     p1_values = self.plot_state['param1_values']
    #     p2_values = self.plot_state['param2_values']

    #     self.param1ValSelect_dd.clear()
    #     self.param2ValSelect_dd.clear()
    #     self.param1ValSelect_dd.addItems(p1_values.astype(str))
    #     self.param2ValSelect_dd.addItems(p2_values.astype(str))
    #     self.param1ValSelect_dd.setCurrentIndex(0)
    #     self.param2ValSelect_dd.setCurrentIndex(0)


    def update_fixed_slider(self):
        if self.plot_state['variables']['x'] == 'Parameter 2':
            sliderlength = len(self.external_variables["param1_values"]) - 1
        else:
            sliderlength = len(self.external_variables["param2_values"]) - 1
        self.fixedParamSpec_slider.setMaximum(sliderlength)
        self.fixedParamTime_slider.setMaximum(sliderlength)
        self.fixedParamSpec_slider.setSliderPosition(0)
        self.fixedParamTime_slider.setSliderPosition(0)

    def update_frequency_slider(self, frequencies):
        self.frequency_slider.setMaximum(len(frequencies) - 1)
        self.frequency_slider.setSliderPosition(0)

    def update_variable_range(self, boxes, variable):
        #fail silently if a bogus variable requested
        if variable in self.variable_labels.keys():
            variable_name = self.variable_labels[variable]
            data = self.external_variables[variable_name]
            _min = np.amin(data)
            _max = np.amax(data)

            for box in boxes:
                box.update_range(_min, _max)



    def load_state_labels(self, state_labels):

        self.state_labels = state_labels

        state_label_boxes = [self.plot_state_grid3D,
                             self.plot_state_time3D,
                             self.plot_state_spec3D]

        time_and_state_boxes = [self.xAxisSinglePlotOptions,
                                self.yAxisSinglePlotOptions,
                                self.zAxisSinglePlotOptions]

        for box in state_label_boxes:
            box.clear()
            box.addItems(list(state_labels.keys()))

        time_and_state = state_labels.copy()
        time_and_state['Time'] = -1                                            #NOTE: This magic number is probably not the ideal way to denote time.
        for from_to_widget in time_and_state_boxes:
            from_to_widget.varDdItems = time_and_state.keys()

    # def load_solution_values(self, param1_values,
    #                          param2_values,
    #                          frequency_bins):
    #     """
    #     Updates sliders and dropdowns with frequency bins and parameter values.

    #     Args:
    #         param1_values (list or array): Values for parameter 1.
    #         param2_values (list or array): Values for parameter 2.
    #         frequency_bins (list or array): Values for frequency bins.
    #     """
    #     self.external_variables["param1_values"] = param1_values
    #     self.external_variables["param2_values"] = param2_values
    #     self.frequency_bins = frequency_bins

    #     n_param1 = len(self.external_variables["param1_values"])
    #     n_param2 = len(self.external_variables["param2_values"])
    #     n_freqs = len(self.frequency_bins)

    #     # Update the frequency slider to have as many positions as len(frequency_bins)
    #     self.frequency_slider.setRange(0, n_freqs - 1)

    #     # Update the fixedParamSpec_slider and fixedParamTime_slider based on the value of x_var
    #     if self.x_var == 'Parameter 1':
    #         self.fixedParamSpec_slider.setRange(0, n_param2 - 1)
    #         self.fixedParamTime_slider.setRange(0, n_param2 - 1)
    #     else:
    #         self.fixedParamSpec_slider.setRange(0, n_param1 - 1)
    #         self.fixedParamTime_slider.setRange(0, n_param1 - 1)

    #     self.fill_paramVal_lists(param1_values, param2_values)

    def populate_toolbutton_menus(self):
        """
        Populates the tool buttons with animation speed menus.

        """
        animate_buttons = [self.param1Animate_button,
                           self.param2Animate_button,
                           self.fixedParamSpecAnimate_button,
                           self.fixedParamTimeAnimate_button]

        speed_options = ['0.5x', '1.0x', '1.5x', '2.0x', '2.5x']

        def create_action(speed_text):
            action = QAction(speed_text, self)
            action.triggered.connect(lambda: self.set_animation_speed(speed_text))
            return action

        for button in animate_buttons:
            menu = QMenu(button)
            for speed in speed_options:
                menu.addAction(create_action(speed))
            button.setMenu(menu)

    def set_animation_speed(self, speed_string):
        """
        Sets the animation speed based on the selected menu option.

        Args:
            speed_string (str): The text value from the menu option (e.g., '1.0x').
        """
        self.plot_state['animation_speed'] = float(speed_string[:-1])


    def get_tab_values(self, tab):
        """
        Triggers the textChanged signal for all entry and combo boxes on
        on the specified tab.

        Args:
            tab (QWidget): The tab on which to trigger signals.
        """
        # Trigger textChanged for all QLineEdit and its subclasses
        line_edits = tab.findChildren(QLineEdit)
        for line_edit in line_edits:
            line_edit.textChanged.emit(line_edit.text())

        # Trigger currentTextChanged for all QComboBox widgets
        combo_boxes = tab.findChildren(QComboBox)
        for combo_box in combo_boxes:
            combo_box.currentTextChanged.emit(combo_box.currentText())

    # def set_independent_variables(self, variables_dict):
    #     self.independent_variables = variables_dict