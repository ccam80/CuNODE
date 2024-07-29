# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 21:00:07 2024

@author: cca79
"""
from qtpy.QtWidgets import QFrame, QAbstractButton, QMenu, QLineEdit, QComboBox
from qtpy.QtCore import Slot, Signal
from qtpy.QtGui import QAction
from gui.resources.widgets.qtdesigner.plot_controller import Ui_plotController
import logging
import numpy as np

class plot_controller_widget(QFrame, Ui_plotController):

    updatePlot = Signal()
    plotstyles = {0: 'grid3d',
                  1: 'time3d',
                  2: 'spec3d',
                  3: 'singlestate'}


    def __init__(self, parent=None):
        super(plot_controller_widget, self).__init__(parent)
        self.plot_state = {
            'state_to_plot': 0,
            'plot_type': 'grid3d',
            'param1_selection': 1.0,
            'param2_selection': 1.0,
            'fixed_value': 1.5,
            'single_plot_style': '2D Phase Plot',
            'x_var': 'Parameter 1',
            'y_var': 'Parameter 2',
            'z_var': 'PSD Magnitude',
            'x_slice': [0,0],
            'y_slice': [0,0],
            'z_slice': [0,0],
            'x_scale': 'Linear',
            'y_scale': 'Linear',
            'z_scale': 'Linear',
        }

        self.setupUi(self)
        self.populate_toolbutton_menus()
        self.state_labels = {'': 0}
        #Keep a copy of these bins in this class, as otherwise the int slider
        #would need to request the data each time it's updated.
        self.frequency_bins = np.zeros(1)
        self.param1_values = np.zeros(1)
        self.param2_values = np.zeros(1)





    @Slot(int)
    def plotMode_select(self, index):
        """Change plot style and "kick" all widgets on that tab into updating
        the plot_state dict with their current values"""

        current_tab = self.plotSettingsTabs.widget(index)
        self.get_tab_values(current_tab)
        logging.debug(f"Plot mode selected: {index}")
        self.plot_state['plot_type'] = self.plotstyles[index]

    @Slot(str, int)
    def update_xSlice(self, value, index):
        self.plot_state['x_slice'][index] = value
        logging.debug("X Slice updated")

    @Slot(str, int)
    def update_ySlice(self, value, index):
        self.plot_state['y_slice'][index] = value
        logging.debug("Y Slice updated")

    @Slot(str, int)
    def update_zSlice(self, value, index):
        self.plot_state['z_slice'][index] = value
        logging.debug("Z Slice updated")

    @Slot(str)
    def update_xScale(self, scale):
        self.plot_state['x_scale'] = scale
        logging.debug("X Scale updated")

    @Slot(str)
    def update_yScale(self, scale):
        logging.debug("Y Scale updated")
        self.plot_state['y_scale'] = scale

    @Slot(str)
    def update_zScale(self, scale):
        logging.debug("Z Scale updated")
        self.plot_state['z_scale'] = scale

    @Slot(str)
    def update_xVar(self, variable):
        logging.debug("X Variable updated")
        self.plot_state['x_var'] = variable

    @Slot(str)
    def update_yVar(self, variable):
        logging.debug("Y Variable updated")
        self.plot_state['y_var'] = variable

    @Slot(str)
    def update_zVar(self, variable):
        logging.debug("Z Variable updated")
        self.plot_state['z_var'] = variable

    @Slot()
    def on_updatePlot(self):
        logging.debug("Update plot")
        # for key, item in self.plot_state.items():
        #     print(key + ": " + str(item))
        # print("")
        self.updatePlot.emit()

    @Slot(int, int)
    def set_current_params_from_plotter(self, param1, param2):
        param1index = np.argmin(np.abs(self.param1_values - param1))
        param2index = np.argmin(np.abs(self.param2_values - param2))
        self.param1ValSelect_dd.setCurrentIndex(param1index)
        self.param2ValSelect_dd.setCurrentIndex(param2index)
        logging.debug(f"params selected: {self.param1_values[param1index], self.param2_values[param2index]}")


    @Slot(int)
    def set_slice_frequency(self, index):
        freq = self.frequency_bins[index]
        self.plot_state['fixed_value'] = freq
        self.currentFrequency_l.display(freq)
        logging.debug(f"Slice frequency set to: {freq}")

    @Slot(int)
    def set_fixedParam(self, index):
        if self.plot_state['x_var'] == 'Parameter 1':
            param_val = self.param2_values[index]
        else:
            param_val = self.param1_values[index]

        self.plot_state['fixed_value'] = param_val
        self.fixedParamTime_display.display(param_val)
        logging.debug(f"Fixed parameter set to: {param_val}")

    @Slot()
    def animate_fixedparam(self):
        logging.debug("Fixed parameter animation started")
        #TODO: figurre out animation logic

    @Slot(QAbstractButton)
    def update_singlePlotStyle(self, button):
        logging.debug(f"Single plot style updated: {button}")
        self.plot_state['single_plot_style'] = button.text()

    @Slot(int)
    def select_single_param1(self, index):
        param = self.param1_values[index]
        self.plot_state['param1_selection'] = self.param1_values[index]
        logging.debug(f"Parameter 1 selected: {param}")


    @Slot(int)
    def select_single_param2(self, index):
        param = self.param2_values[index]
        self.plot_state['param2_selection'] = self.param2_values[index]
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
        state_index = self.state_labels[state]
        self.plot_state['state_to_plot'] = state_index
        state_to_plot_comboboxes = [self.plot_state_time3D,
                                    self.plot_state_spec3D,
                                    self.plot_state_grid3D]

        for box in state_to_plot_comboboxes:
            box.setCurrentIndex(state_index)
        logging.debug("Plot state set to {state_index}")

    def populate_swept_parameter_values(self, param1_values, param2_values):
        self.param1ValSelect_dd.clear()
        self.param2ValSelect_dd.clear()
        self.param1ValSelect_dd.addItems(param1_values)
        self.param2ValSelect_dd.addItems(param2_values)
        self.param1ValSelect_dd.setCurrentIndex(0)
        self.param2ValSelect_dd.setCurrentIndex(0)
        self.sim_state['param1_values'] = param1_values
        self.sim_state['param2_values'] = param2_values

    def update_fixed_sliders(self, param):
        if param == 'param1':
            sliderlength = len(self.sim_s) - 1
        elif param == 'param2':
            sliderlength = len(self.sim_s) - 1
        self.fixedParamSpec_slider.setMaximum(sliderlength)
        self.fixedParamTime_slider.setMaximum(sliderlength)

    def update_frequency_slider(self, frequencies):
        self.frequency_slider.setMaximum(len(frequencies) - 1)

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

    def load_solution_values(self, param1_values,
                             param2_values,
                             frequency_bins):
        """
        Updates sliders and dropdowns with frequency bins and parameter values.

        Args:
            param1_values (list or array): Values for parameter 1.
            param2_values (list or array): Values for parameter 2.
            frequency_bins (list or array): Values for frequency bins.
        """
        self.param1_values = param1_values
        self.param2_values = param2_values
        self.frequency_bins = frequency_bins

        n_param1 = len(self.param1_values)
        n_param2 = len(self.param2_values)
        n_freqs = len(self.frequency_bins)

        # Update the frequency slider to have as many positions as len(frequency_bins)
        self.frequency_slider.setRange(0, n_freqs - 1)

        # Update the fixedParamSpec_slider and fixedParamTime_slider based on the value of x_var
        if self.x_var == 'Parameter 1':
            self.fixedParamSpec_slider.setRange(0, n_param2 - 1)
            self.fixedParamTime_slider.setRange(0, n_param2 - 1)
        else:
            self.fixedParamSpec_slider.setRange(0, n_param1 - 1)
            self.fixedParamTime_slider.setRange(0, n_param1 - 1)

        self.fill_paramVal_lists(param1_values, param2_values)

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
