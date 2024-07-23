# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:42:23 2024

@author: cca79
"""

import sys
from os import environ, makedirs
environ["QT_API"] = "pyside6"

from os.path import splitext, basename
import logging
sys.path.append("..") # This is an easily-broken, dangerous way to get at the solver files until the project directory is refactored
from _utils import round_sf
from Euler_maruyama_solver import Solver
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec

from qtpy.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QGridLayout, QWidget, QFileDialog, QGroupBox, QErrorMessage
from qtpy.QtGui import QAction, QActionGroup
from qtpy import QtCore
from QT_simGUI import Ui_MainWindow  # Import the generated class from your_file.py


class ODE_Explorer_Controller(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ODE_Explorer_Controller, self).__init__()
        self.init_logging()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.precision = np.float64
        self.init_precision_action_group()
        self.diffeq_system_file_path = None

        logging.info("ODE_Explorer_Controller initialized")

        self.plot_state = {
            'param1': None,
            'param2': None,
            'current_fft_freq': None,
            'single_plot_style': None,
            'x_var': None,
            'y_var': None,
            'z_var': None,
            'x_slice': None,
            'y_slice': None,
            'z_slice': None,
            'x_scale': None,
            'y_scale': None,
            'z_scale': None,
        }

        self.sim_state = {
            'dt': None,
            'duration': None,
            'fs': None,
            'param1_sweep_bounds': None,
            'param1_sweep_scale': None,
            'param1_num_values': 100,
            'param1_var': None,
            'param1_values': [],
            'param2_sweep_bounds': None,
            'param2_sweep_scale': None,
            'param2_num_values': 100,
            'param2_var': None,
            'param2_values': [],
            'warmup': None,

            'y0': None
        }
        self.param_index_map = {}
        self.display_sig_figs = 4
    def init_logging(self):
        # Create logs directory if it doesn't exist
        makedirs("logs", exist_ok=True)

        # Configure the logger
        logging.basicConfig(
            filename='logs/GUIlog.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )

        logging.debug('Logfile initialised')

    def init_precision_action_group(self):
        """Group 32-64 bit precision buttons as QTDesigner has dropped support for
        doing this inside the GUI"""

        self.action_64bit = self.findChild(QAction, "action64_bit_2")
        self.action_32bit = self.findChild(QAction, "action32_bit_2")

        self.precision_group = QActionGroup(self)
        self.precision_group.addAction(self.action_64bit)
        self.precision_group.addAction(self.action_32bit)
        self.precision_group.setExclusive(True)

        self.action_64bit.triggered.connect(lambda: self.set_precision(self.action_64bit))
        self.action_32bit.triggered.connect(lambda: self.set_precision(self.action_32bit))

        def set_precision(self, action):
            if action.text() == '64-bit':
                self.precision = np.float64
            elif action.text() == '32-bit':
                self.precision = np.float32
            print(f"Precision set to {self.precision}")

    def load_system_from_filedialog(self):
        """Open file dialog to select diffeq_system file, instantiate and start solver"""

        self.diffeq_system_file_path, _ = QFileDialog.getOpenFileName(self, "Select Python File", "", "Python Files (*.py);;All Files (*)")
        if self.diffeq_system_file_path:
            system = self.get_system_from_file(self.diffeq_system_file_path)
        else:
            logging.warning("diffeq system file path not valid")

        self.load_system(system)


    def get_system_from_file(self, file_path):
        module_name = splitext(basename(file_path))[0]
        spec = spec_from_file_location(module_name, file_path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, 'diffeq_system'):
            system = getattr(module, 'diffeq_system')(precision = self.precision)
            return system
        else:
            logging.warning("The module you selected has no diffeq_system class")
            raise AttributeError("The module you selected has no diffeq_system class")


    def load_system(self, system):
        """ Start a new solver instance, and load a provided ODE system into it.
        Populate the system paramaters box, the inits and noise sigmas fields,
        fill the param select lists, load default sweep if present. """

        self.solver = Solver(self.precision)
        self.solver.load_system(system)
        self.populate_sysParams_tab(self.solver.system.constants_dict)
        self.populate_groupbox_with_array(self.ui.inits_box, np.zeros(self.solver.system.num_states))
        self.fill_paramSelect_lists()


    def reload_system(self):
        """ Assumes we have already located the ODE_system file, and are just
        reloading to either change solvers of modify precision"""
        system = self.get_system_from_file(self.diffeq_file_path)
        self.load_system(system)

    def fill_paramSelect_lists(self):
        # Clear existing items
        self.ui.p1Select_dd.clear()
        self.ui.p2Select_dd.clear()

        # Add keys to QComboBox
        self.ui.p1Select_dd.addItems(self.solver.system.constants_dict.keys())
        self.ui.p2Select_dd.addItems(self.solver.system.constants_dict.keys())


    def animate_3D(self):
        pass

    def save_results(self):
        pass

    def select_param1(self):
        self.plot_state['param1_val'] = self.ui.param1ValSelect_dd.currentText()
        self.update_selected_params()

    def select_param2(self):
        self.plot_state['param2_val'] = self.ui.param2ValSelect_dd.currentText()
        self.update_selected_params()

    def update_selected_params(self):
        p1 = self.plot_state['param1_val']
        p2 = self.plot_state['param2_val']
        self.solver.param_index = self.solver.get_param_index((p1, p2))
        self.single_state = self.solver.solved_ODE.output_array[:, self.solver.param_index, :]

    def set_current_fft_freq(self):
        pass

    def set_duration(self):
        self.sim_state['duration'] = self.ui.duration_e.value()

    def set_fs(self):
        self.sim_state['fs'] = self.ui.fs_e.value()

    def set_dt(self):
        self.sim_state['dt'] = self.ui.stepsize_e.value()

    def set_warmup(self):
        self.sim_state['warmup'] = self.ui.warmup_e.value()

    def set_param1Sweep_bounds(self):
        lower_bound = self.ui.p1SweepLower_entry.value()
        upper_bound = self.ui.p1SweepUpper_entry.value()
        self.sim_state['param1_sweep_bounds'] = (lower_bound, upper_bound)

    def set_param1Sweep_scale(self, index):
        if index == 1:
            scale = 'log'
        else:
            scale = 'lin'
        self.sim_state['param1_sweep_scale'] = scale

    def set_param1_var(self):
        self.sim_state['param1_var'] = self.ui.p1Select_dd.value()

    def set_param2Sweep_bounds(self):
        lower_bound = self.ui.p2SweepLower_entry.value()
        upper_bound = self.ui.p2SweepUpper_entry.value()
        self.sim_state['param2_sweep_bounds'] = (lower_bound, upper_bound)

    def set_param2Sweep_scale(self, index):
        if index == 1:
            scale = 'log'
        else:
            scale = 'lin'
        self.sim_state['param1_sweep_scale'] = scale

    def set_param2_var(self):
        self.sim_state['param2_var'] = self.ui.p2select_dd.value()

    def build_sweeps(self):
        #TODO: Add num values to GUI and logic.
        param1_bounds = self.sim_state['param1_sweep_bounds']
        param1_scale = self.sim_state['param1_sweep_scale']
        param1_num_values = self.sim_state['param1_num_values']

        param2_bounds = self.sim_state['param2_sweep_bounds']
        param2_scale = self.sim_state['param2_sweep_scale']
        param2_num_values = self.sim_state['param2_num_values']

        if param1_bounds and param1_scale and param1_num_values:
            if param1_scale == 'lin':
                self.sim_state['param1_values'] = np.linspace(param1_bounds[0],
                                                              param1_bounds[1],
                                                              param1_num_values,
                                                              dtype=self.precision)
            elif param1_scale == 'log':
                try:
                    self.sim_state['param1_values'] = np.logspace(np.log10(param1_bounds[0]),
                                                                  np.log10(param1_bounds[1]),
                                                                  param1_num_values,
                                                                  dtype=self.precision)
                except:
                    logging.warning(f"P1 bounds: {param1_bounds} unable to create a log space")
                    self.displayerror("Param 1 values are unloggable, try a linear sweep or change values")

        if param2_bounds and param2_scale and param2_num_values:
            if param2_scale == 'lin':
                self.sim_state['param2_values'] = np.linspace(param2_bounds[0],
                                                              param2_bounds[1],
                                                              param2_num_values,
                                                              dtype=self.precision)
            elif param2_scale == 'log':
                try:
                    self.sim_state['param2_values'] = np.logspace(np.log10(param2_bounds[0]),
                                                                  np.log10(param2_bounds[1]),
                                                                  param2_num_values,
                                                                  dtype=self.precision)
                except:
                    logging.warning(f"P1 bounds: {param1_bounds} unable to create a log space")
                    self.displayerror("Param 1 values are unloggable, try a linear sweep or change values")

    def generate_index_map(self):
        """Set up a dict mapping parameter sets to their index in the output
        array, cutting some compute time when populating the z mesh for plotting.

        Saves results to self.param_index_map.
        """
        self.param_index_map = {}
        grid_values = [(p1, p2) for p1 in self.sim_state['param1_values'] for p2 in self.sim_state['param2_values']]
        for idx, (p1, p2) in enumerate(grid_values):
            p1_round = self.precision(round_sf(p1, self.display_sig_figs))
            p2_round = self.precision(round_sf(p2, self.display_sig_figs))
            self.param_index_map[(p1_round, p2_round)] = idx

    def set_singlePlot_style(self):
        pass

    #TODO: fix widget names from here down.
    def set_xScale(self):
        self.plot_state['x_scale'] = self.ui.x_scale.currentText()

    def set_yScale(self):
        self.plot_state['y_scale'] = self.ui.y_scale.currentText()

    def set_zScale(self):
        self.plot_state['z_scale'] = self.ui.z_scale.currentText()

    def set_y0(self):
        self.sim_state['inits'] = [self.ui.inits_box.itemAt(i).widget().value() for i in range(self.ui.inits_box.count())]

    def set_noise_sigmas(self):
        self.sim_state['noise_sigmas'] = [self.ui.noise_sigmas_box.itemAt(i).widget().value() for i in range(self.ui.noise_sigmas_box.count())]

    def solve_ODE(self):
        self.build_sweeps()

        # Update solver with simulation parameters
        self.solver.duration = self.sim_state['duration']
        self.solver.fs = self.sim_state['fs']
        self.solver.step_size = self.sim_state['dt']
        self.solver.warmup_time = self.sim_state['warmup']
        self.t = np.linspace(0,  self.duration -  1/self.fs, int( self.duration * self.fs))

    def update_x_var(self):
        self.plot_state['x_var'] = self.ui.x_var_select.currentText()

    def update_y_var(self):
        self.plot_state['y_var'] = self.ui.y_var_select.currentText()

    def update_z_var(self):
        self.plot_state['z_var'] = self.ui.z_var_select.currentText()

    def update_plot(self):
        pass
    def solve_ODE(self):
        self.build_sweeps()

        # Update solver with simulation parameters
        self.solver.duration = self.sim_state['duration']
        self.solver.fs = self.sim_state['fs']
        self.solver.step_size = self.sim_state['dt']
        self.solver.warmup_time = self.sim_state['warmup']

    def update_x_slice(self):
        self.plot_state['x_slice'] = (self.ui.x_slice_min.value(), self.ui.x_slice_max.value())

    def update_y_slice(self):
        self.plot_state['y_slice'] = (self.ui.y_slice_min.value(), self.ui.y_slice_max.value())

    def update_z_slice(self):
        self.plot_state['z_slice'] = (self.ui.z_slice_min.value(), self.ui.z_slice_max.value())

    def populate_sysParams_tab(self, data_dict):
    # Clear existing widgets
        if self.ui.sysParams_tab.layout():
            for i in reversed(range(self.ui.sysParams_tab.layout().count())):
                widget = self.ui.sysParams_tab.layout().itemAt(i).widget()
                if widget:
                    widget.deleteLater()

        layout = QGridLayout()
        keys = list(data_dict.keys())

        numcols = 2
        for row in range((len(keys) + 1) // numcols):
            col = 0
            for i in range(numcols):
                idx = row * numcols + i
                if idx >= len(keys):
                    break
                key = keys[idx]
                value = data_dict[key]

                label = QLabel(key)
                entry = QLineEdit()
                entry.setText(f"{value:.2f}")

                layout.addWidget(label, row, col)
                layout.addWidget(entry, row, col + 1)
                col += numcols

        noise_groupbox = QGroupBox("Gaussian noise (std dev)")
        self.populate_groupbox_with_array(noise_groupbox, self.solver.system.noise_sigmas)
        layout.addWidget(noise_groupbox, (len(keys) + 1) // numcols, 0, 1, numcols * 2)

        container = QWidget()
        container.setLayout(layout)
        self.ui.sysParams_tab.layout().addWidget(container)



    def populate_groupbox_with_array(self, groupbox, array):
        # Clear existing widgets
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

            layout.addWidget(label, i, 0)
            layout.addWidget(entry, i, 1)


    def displayerror(self, message):
        error_dialog = QErrorMessage()
        error_dialog.showMessage(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ODE_Explorer_Controller()
    window.show()
    sys.exit(app.exec())
